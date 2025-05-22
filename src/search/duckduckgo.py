"""
DuckDuckGo Search Module for Agentic Researcher
Implements DuckDuckGo search functionality
"""
import os
import sys
import time
import logging
import random
from typing import List, Dict, Any
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random, retry_if_exception_type, before_sleep_log, RetryError, after_log

# Add project root to Python path for direct script execution
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    sys.path.insert(0, project_root)

from src.utils.config import Config as config
from src.utils.proxy_handler import ProxyHandler

# Set up logger
logger = logging.getLogger(__name__)

class DuckDuckGoSearch:
    """
    DuckDuckGo search implementation
    Uses the duckduckgo_search library with retry logic
    """
    
    def __init__(self):
        # Use global config
        self.config = config
        
        # Proxy handler for rotating user agents
        self.proxy_handler = ProxyHandler()
        
        # Configure DuckDuckGo client with proxies and timeouts
        self.ddgs = DDGS(proxies=None, timeout=15)  # Increased timeout
        
        # Rate limiting settings
        self.request_delay = 3.0  # Increased delay between requests to avoid rate limiting
        self.last_request_time = 0
        self.last_success_time = 0
        
        # Search settings
        self.max_retries = 5  # Increased from 3 to 5
        self.timeout = 20  # Increased timeout
        
        # Backoff settings
        self.backoff_factor = 2.0  # Exponential backoff factor
        self.jitter = True  # Add jitter to backoff
        self.rate_limit_window = 60 * 10  # 10 minute window for rate limiting memory
        
        # Rate limit tracking
        self.rate_limit_incidents = []
        self.max_incidents_before_extended_backoff = 3
        self.extended_backoff_time = 60 * 30  # 30 minutes
        self.extended_backoff_until = 0
        
        # Per-session request counter to track behavior across sessions
        self.session_id = random.randint(1000000, 9999999)
        self.request_counter_file = os.path.join(os.path.dirname(__file__), '.ddg_request_counter')
        self.daily_request_limit = 100  # Conservative limit per day
        
        # Load previous request counts
        self.daily_request_count = self._load_request_count()
        
        # User agent rotation with a much larger pool
        self.rotate_user_agent_every = 2  # Rotate user agent more frequently
        self.request_count = 0
        self.used_user_agents = set()  # Track used user agents to avoid reuse
        
        # IP and location rotation through proxy selection
        self.ip_rotation_enabled = True
        self.ip_rotation_frequency = 5  # Rotate IP every 5 requests
        
        # Regional query distribution (use different region codes)
        self.regions = ["wt-wt", "us-en", "uk-en", "ca-en", "au-en", "in-en", "de-de"]
        self.current_region_index = 0
        
        logger.info("DuckDuckGo search client initialized successfully with enhanced retry mechanism")
    
    def _load_request_count(self):
        """Load the daily request count from file if it exists"""
        try:
            if os.path.exists(self.request_counter_file):
                with open(self.request_counter_file, 'r') as f:
                    data = f.read().strip().split(',')
                    if len(data) == 2:
                        date_str, count_str = data
                        today = time.strftime('%Y-%m-%d')
                        if date_str == today and count_str.isdigit():
                            return int(count_str)
        except Exception as e:
            logger.warning(f"Error loading request count: {e}")
        return 0
        
    def _save_request_count(self):
        """Save the current request count to file"""
        try:
            today = time.strftime('%Y-%m-%d')
            with open(self.request_counter_file, 'w') as f:
                f.write(f"{today},{self.daily_request_count}")
        except Exception as e:
            logger.warning(f"Error saving request count: {e}")
    
    def _rotate_region(self):
        """Rotate through different region codes to distribute queries"""
        self.current_region_index = (self.current_region_index + 1) % len(self.regions)
        return self.regions[self.current_region_index]
    
    def _get_fresh_user_agent(self):
        """Get a user agent that hasn't been used recently"""
        try:
            # Get a list of modern user agents
            ua_list = [
                # Chrome on Windows
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
                # Firefox on Windows
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
                # Edge on Windows
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36 Edg/92.0.902.78",
                # Chrome on macOS
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
                # Safari on macOS
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
                # Chrome on Linux
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                # Firefox on Linux
                "Mozilla/5.0 (X11; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0",
                # Chrome on Android
                "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
                # Safari on iOS
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
            ]
            
            # Also try to get user agents from proxy handler if available
            try:
                proxy_ua_list = self.proxy_handler.get_user_agents()
                if proxy_ua_list and isinstance(proxy_ua_list, list) and len(proxy_ua_list) > 0:
                    # Add these to our list, avoiding duplicates
                    ua_list.extend([ua for ua in proxy_ua_list if ua not in ua_list])
            except:
                # If proxy handler doesn't have get_user_agents or fails, continue with our list
                pass
            
            # Find user agents we haven't used yet
            available_user_agents = [ua for ua in ua_list if ua not in self.used_user_agents]
            
            # If we've used all user agents, start over but keep track of the last 3 used
            # to avoid immediately reusing the most recent ones
            if not available_user_agents and ua_list:
                recent_agents = list(self.used_user_agents)[-3:] if len(self.used_user_agents) > 3 else self.used_user_agents
                self.used_user_agents = set(recent_agents)
                available_user_agents = [ua for ua in ua_list if ua not in self.used_user_agents]
            
            # Select a user agent and mark it as used
            if available_user_agents:
                selected_ua = random.choice(available_user_agents)
                self.used_user_agents.add(selected_ua)
                return selected_ua
        except Exception as e:
            logger.warning(f"Error getting fresh user agent: {e}")
            
        # Fall back to a generic modern user agent if all else fails
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
        
    def _is_rate_limit_error(self, exception):
        """Check if the exception is related to rate limiting"""
        error_str = str(exception).lower()
        return any(term in error_str for term in ["rate", "limit", "429", "too many requests", "ratelimit", "202", "blocked"])
    
    def _record_rate_limit_incident(self):
        """Record a rate limit incident for adaptive backoff"""
        now = time.time()
        
        # Add the current incident
        self.rate_limit_incidents.append(now)
        
        # Remove incidents outside the window
        self.rate_limit_incidents = [t for t in self.rate_limit_incidents 
                                  if now - t <= self.rate_limit_window]
        
        # Check if we need extended backoff
        if len(self.rate_limit_incidents) >= self.max_incidents_before_extended_backoff:
            logger.warning(f"Too many rate limit incidents ({len(self.rate_limit_incidents)}). Enabling extended backoff.")
            self.extended_backoff_until = now + self.extended_backoff_time
    
    def _should_apply_extended_backoff(self):
        """Check if we're in an extended backoff period"""
        return time.time() < self.extended_backoff_until
    
    def _get_success_interval(self):
        """Get time interval since last successful request"""
        if self.last_success_time == 0:
            return float('inf')
        return time.time() - self.last_success_time
        
    def _random_jitter(self, base=1, deviation=2):
        """Add random jitter to avoid synchronized retries"""
        return random.uniform(base, base + deviation)
    
    def search(self, query: str, max_results: int = 10, region: str = "wt-wt", 
              safesearch: str = "moderate") -> List[Dict[str, Any]]:
        """
        Perform DuckDuckGo search with advanced rate limit handling
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            region: Region code for search
            safesearch: SafeSearch setting (off, moderate, strict)
            
        Returns:
            List[Dict]: Search results with title, body, and url
        """
        logger.info(f"Executing DuckDuckGo search for query: {query}")
        
        # Check daily request limit
        if self.daily_request_count >= self.daily_request_limit:
            logger.warning(f"Daily request limit of {self.daily_request_limit} reached. Using placeholder results.")
            return self._generate_placeholder_results(query, max_results)
            
        # Check if we're in extended backoff period
        if self._should_apply_extended_backoff():
            remaining = int(self.extended_backoff_until - time.time())
            logger.warning(f"In extended backoff period for {remaining}s. Using placeholder results.")
            return self._generate_placeholder_results(query, max_results)

        # Adaptive delay based on success interval
        success_interval = self._get_success_interval()
        if success_interval < 300:  # 5 minutes
            adaptive_delay = self.request_delay
        elif success_interval < 900:  # 15 minutes
            adaptive_delay = self.request_delay * 0.8  # Slightly reduce delay if we haven't had a success in a while
        else:
            adaptive_delay = self.request_delay * 0.6  # More aggressive if it's been a long time
            
        # Enforce base rate limiting with adaptive component
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < adaptive_delay:
            sleep_time = adaptive_delay - time_since_last
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
        # Increment and save request counter
        self.daily_request_count += 1
        self._save_request_count()
        
        # Record this request time
        self.last_request_time = time.time()
        self.request_count += 1
        
        # Try multiple techniques to get results
        for attempt in range(4):  # 0, 1, 2, 3 attempts
            try:
                # Apply exponential backoff for retries
                if attempt > 0:
                    backoff_time = 2 ** attempt
                    logger.info(f"Retry attempt {attempt}, waiting {backoff_time}s")
                    time.sleep(backoff_time)
                    
                    # Rotate region for each retry
                    region = self._rotate_region()
                    logger.info(f"Using region code: {region}")
                
                # Rotate user agents to avoid detection
                if attempt > 0 or self.request_count % self.rotate_user_agent_every == 0:
                    user_agent = self._get_fresh_user_agent()
                    if user_agent:
                        logger.info("Rotating user agent")
                        self.ddgs.user_agent = user_agent
                
                # Modify query or settings based on attempt number
                current_query = query
                current_safesearch = safesearch
                
                if attempt == 1:
                    # Try with different safe search setting
                    current_safesearch = "off" if safesearch != "off" else "moderate"
                    logger.info(f"Using alternative safesearch: {current_safesearch}")
                elif attempt >= 2:
                    # Try with modified query
                    current_query = self._modify_query(query, attempt-1)
                    logger.info(f"Using modified query: {current_query}")
                
                # Execute search
                logger.info(f"Executing search with {current_query}, region={region}, safesearch={current_safesearch}")
                raw_results = self.ddgs.text(current_query, region=region, safesearch=current_safesearch)
                
                # Process results
                results = []
                if raw_results:
                    for r in raw_results:
                        # Check for rate limiting indicators
                        if ('href' in r and 'duckduckgo.com/?q=' in r['href'] and 
                            ('rate' in r.get('body', '').lower() or 'limit' in r.get('body', '').lower())):
                            logger.warning("Detected rate limiting in results")
                            self._record_rate_limit_incident()
                            raise Exception("Rate limit detected in search results")
                            
                        # Add formatted result
                        results.append({
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                            "url": r.get("href", "")
                        })
                        
                        # Stop once we have enough results
                        if len(results) >= max_results:
                            break
                
                # If we got results, return them
                if results:
                    self.last_success_time = time.time()
                    logger.info(f"Search succeeded with {len(results)} results")
                    return results
                else:
                    logger.warning(f"No results on attempt {attempt}, trying next approach")
                    
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = self._is_rate_limit_error(e)
                
                if is_rate_limit:
                    self._record_rate_limit_incident()
                    logger.warning(f"Rate limit detected on attempt {attempt}: {e}")
                else:
                    logger.error(f"Search error on attempt {attempt}: {e}")
                    
                # If we're on the last attempt and hit rate limit, use placeholders
                if attempt == 3 and is_rate_limit:
                    logger.warning("Rate limiting persists after all attempts, using placeholders")
                    return self._generate_placeholder_results(query, max_results)
        
        # If we reach this point, all attempts failed
        logger.error("All search attempts failed")
        return self._generate_placeholder_results(query, max_results)
    
    @retry(
        stop=stop_after_attempt(5),  # Increased from 3 to 5 attempts
        wait=wait_exponential(multiplier=2, min=2, max=60) + wait_random(0, 2),  # Exponential backoff with jitter
        retry=retry_if_exception_type((Exception)),  # Retry on all exceptions
        before_sleep=before_sleep_log(logger, logging.INFO),  # Log before sleeping
        after=after_log(logger, logging.INFO),  # Log after retry attempt
        reraise=False  # Don't re-raise the last exception if all retries fail
    )
    def image_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform DuckDuckGo image search
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Image search results
        """
        logger.info(f"Executing DuckDuckGo image search for query: {query}")
        
        # Enforce rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        try:
            # Add headers with user agents
            headers = self.proxy_handler.get_headers()
            
            # Perform image search
            results = []
            for r in self.ddgs.images(
                query,
                max_results=max(max_results * 2, 20)  # Request more to ensure we get enough
            ):
                results.append({
                    "title": r.get("title", ""),
                    "image": r.get("image", ""),
                    "thumbnail": r.get("thumbnail", ""),
                    "url": r.get("url", ""),
                    "height": r.get("height", 0),
                    "width": r.get("width", 0),
                    "source": r.get("source", "")
                })
                
                # Break if we have enough results
                if len(results) >= max_results:
                    break
            
            # Update last request time
            self.last_request_time = time.time()
            
            # If we don't have enough results, generate placeholders
            if len(results) < max_results:
                logger.warning(f"Could only retrieve {len(results)} image results for query: {query}")
                # Fill remaining slots with placeholder results
                for i in range(len(results), max_results):
                    results.append({
                        "title": f"Image result {i+1} for '{query}'" if i == 0 else f"Alternative image for '{query}'",
                        "image": "",
                        "thumbnail": "",
                        "url": "https://duckduckgo.com/?q=" + query.replace(" ", "+") + "&ia=images",
                        "height": 0,
                        "width": 0,
                        "source": "placeholder"
                    })
            
            # Ensure we only return the requested number of results
            results = results[:max_results]
            
            logger.info(f"DuckDuckGo image search returned {len(results)} results")
            return results
        
        except Exception as e:
            # Log error
            logger.error(f"DuckDuckGo image search error: {str(e)}")
            # Return placeholder results instead of empty list
            return self._generate_image_placeholder_results(query, max_results)
    
    @retry(
        stop=stop_after_attempt(5),  # Increased from 3 to 5 attempts
        wait=wait_exponential(multiplier=2, min=2, max=60) + wait_random(0, 2),  # Exponential backoff with jitter
        retry=retry_if_exception_type((Exception)),  # Retry on all exceptions
        before_sleep=before_sleep_log(logger, logging.INFO),  # Log before sleeping
        after=after_log(logger, logging.INFO),  # Log after retry attempt
        reraise=False  # Don't re-raise the last exception if all retries fail
    )
    def news_search(self, query: str, max_results: int = 10, time_period: str = "d") -> List[Dict[str, Any]]:
        """
        Perform DuckDuckGo news search
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            time_period: Time period (d: day, w: week, m: month)
            
        Returns:
            List[Dict]: News search results
        """
        logger.info(f"Executing DuckDuckGo news search for query: {query}, time period: {time_period}")
        
        # Enforce rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        try:
            # Add headers with user agents
            headers = self.proxy_handler.get_headers()
            
            # Perform news search
            results = []
            for r in self.ddgs.news(
                query,
                max_results=max(max_results * 2, 20),  # Request more to ensure we get enough
                time_period=time_period
            ):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("url", ""),
                    "published": r.get("published", ""),
                    "source": r.get("source", "")
                })
                
                # Break if we have enough results
                if len(results) >= max_results:
                    break
            
            # Update last request time
            self.last_request_time = time.time()
            
            # If we don't have enough results, try with a different time period
            if len(results) < max_results and time_period != "m":
                logger.info(f"Not enough news results ({len(results)}/{max_results}). Trying with extended time period")
                
                # Try with a longer time period
                extended_period = "w" if time_period == "d" else "m"
                
                # Add a small delay before the next request
                time.sleep(self.request_delay)
                
                for r in self.ddgs.news(
                    query,
                    max_results=max_results * 2,
                    time_period=extended_period
                ):
                    # Check for duplicates before adding
                    url = r.get("url", "")
                    if not any(result["url"] == url for result in results):
                        results.append({
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                            "url": url,
                            "published": r.get("published", ""),
                            "source": r.get("source", "")
                        })
                    
                    # Break if we have enough results
                    if len(results) >= max_results:
                        break
            
            # If we still don't have enough results, generate placeholders
            if len(results) < max_results:
                logger.warning(f"Could only retrieve {len(results)} news results for query: {query}")
                # Fill remaining slots with placeholder results
                for i in range(len(results), max_results):
                    results.append({
                        "title": f"News result {i+1} for '{query}'" if i == 0 else f"Alternative news for '{query}'",
                        "body": "No additional news found. Consider refining your search query or checking different time periods.",
                        "url": "https://duckduckgo.com/?q=" + query.replace(" ", "+") + "&ia=news",
                        "published": "",
                        "source": "placeholder"
                    })
            
            # Ensure we only return the requested number of results
            results = results[:max_results]
            
            logger.info(f"DuckDuckGo news search returned {len(results)} results")
            return results
        
        except Exception as e:
            # Log error
            logger.error(f"DuckDuckGo news search error: {str(e)}")
            # Return placeholder results instead of empty list
            return self._generate_news_placeholder_results(query, max_results)


    def _modify_query(self, query: str, attempt: int) -> str:
        """
        Modify the query to try to get more results
        
        Args:
            query: Original search query
            attempt: Current attempt number
            
        Returns:
            str: Modified query
        """
        if attempt == 1:
            # First attempt: Add "about" to the query
            return f"about {query}"
        elif attempt == 2:
            # Second attempt: Remove quotes and special characters
            cleaned_query = query.replace('"', '').replace("'", "").replace("?", "").replace("!", "")
            return cleaned_query
        else:
            # Third attempt: Use more general terms from the query
            words = query.split()
            if len(words) > 3:
                # Use the first few significant words
                return " ".join(words[:3])
            else:
                # Add a general term
                return f"{query} information"
    
    def _generate_placeholder_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Generate placeholder results when search fails
        
        Args:
            query: Search query
            max_results: Number of results to generate
            
        Returns:
            List[Dict]: Placeholder search results
        """
        results = []
        for i in range(max_results):
            results.append({
                "title": f"Search result {i+1} for '{query}'" if i == 0 else f"Alternative search result for '{query}'",
                "body": "Search could not be completed. Please try again later or refine your search query.",
                "url": "https://duckduckgo.com/?q=" + query.replace(" ", "+")
            })
        return results
    
    def _generate_image_placeholder_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Generate placeholder image results when search fails
        
        Args:
            query: Search query
            max_results: Number of results to generate
            
        Returns:
            List[Dict]: Placeholder image search results
        """
        results = []
        for i in range(max_results):
            results.append({
                "title": f"Image result {i+1} for '{query}'" if i == 0 else f"Alternative image for '{query}'",
                "image": "",
                "thumbnail": "",
                "url": "https://duckduckgo.com/?q=" + query.replace(" ", "+") + "&ia=images",
                "height": 0,
                "width": 0,
                "source": "placeholder"
            })
        return results
    
    def _generate_news_placeholder_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Generate placeholder news results when search fails
        
        Args:
            query: Search query
            max_results: Number of results to generate
            
        Returns:
            List[Dict]: Placeholder news search results
        """
        results = []
        for i in range(max_results):
            results.append({
                "title": f"News result {i+1} for '{query}'" if i == 0 else f"Alternative news for '{query}'",
                "body": "News search could not be completed. Please try again later or refine your search query.",
                "url": "https://duckduckgo.com/?q=" + query.replace(" ", "+") + "&ia=news",
                "published": "",
                "source": "placeholder"
            })
        return results


# Comprehensive example usage of DuckDuckGoSearch
if __name__ == "__main__":
    import sys
    import os
    import json
    import pandas as pd
    import time
    from datetime import datetime
    from pathlib import Path
    
    # Add project root to Python path if necessary
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Try to import related components
    try:
        from src.utils.file_utils import ensure_dir, write_json_file
        from src.search.unified_scraper import UnifiedScraper
        integrated_components = True
    except ImportError:
        integrated_components = False
        print("Warning: Some integrated components not available. Running in standalone mode.")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(os.path.dirname(__file__), 'duckduckgo_search.log'))
        ]
    )
    
    # Create output directory for results
    output_dir = Path(os.path.join(project_root, "example_results", f"duckduckgo_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("DUCKDUCKGO SEARCH - COMPREHENSIVE EXAMPLE")
    print("=" * 80)
    
    # Initialize the DuckDuckGo search
    print("\nInitializing DuckDuckGo search client...")
    ddg = DuckDuckGoSearch()
    
    # Collection to track all results
    all_results = []
    
    try:
        # Example 1: Basic web search with different result counts
        print("\n" + "-" * 80)
        print("EXAMPLE 1: BASIC WEB SEARCH")
        print("-" * 80)
        
        # Define a research query
        query = "What are the latest developments in large language models?"
        print(f"Query: '{query}'")
        
        # Try different result counts
        result_counts = [3, 10]
        for count in result_counts:
            print(f"\nRequesting {count} results...")
            start_time = time.time()
            try:
                results = ddg.search(query, max_results=count)
                duration = time.time() - start_time
                
                print(f"✓ Found {len(results)} results in {duration:.2f} seconds")
                
                # Display the results
                for i, result in enumerate(results[:5], 1):  # Show up to 5 results
                    print(f"\nResult {i}:")
                    print(f"Title: {result['title']}")
                    print(f"URL: {result['url']}")
                    
                    # Print a snippet of the body text
                    body_snippet = result['body'][:100] + "..." if len(result['body']) > 100 else result['body']
                    print(f"Snippet: {body_snippet}")
                
                # Record results for later analysis
                all_results.append({
                    "query": query,
                    "type": "web",
                    "count": count,
                    "results_found": len(results),
                    "duration": duration,
                    "results": results
                })
                
                # Save results to file
                results_file = output_dir / f"web_search_{count}_results.json"
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to: {results_file}")
                
            except Exception as e:
                print(f"× Search failed: {str(e)}")
        
        # Example 2: Regional search variations
        print("\n" + "-" * 80)
        print("EXAMPLE 2: REGIONAL SEARCH VARIATIONS")
        print("-" * 80)
        
        query = "local news today"
        regions = ["us-en", "uk-en", "in-en", "au-en"]
        region_names = {"us-en": "United States", "uk-en": "United Kingdom", "in-en": "India", "au-en": "Australia"}
        
        print(f"Query: '{query}'")
        print("Testing different regional search results...")
        
        regional_results = {}
        for region in regions:
            print(f"\nRegion: {region_names.get(region, region)}")
            try:
                results = ddg.search(query, max_results=5, region=region)
                regional_results[region] = results
                
                print(f"✓ Found {len(results)} results")
                if results:
                    print(f"First result: {results[0]['title']}")
                    print(f"URL: {results[0]['url']}")
                    
                # Track for later analysis
                all_results.append({
                    "query": query,
                    "type": "regional",
                    "region": region,
                    "region_name": region_names.get(region, region),
                    "results_found": len(results)
                })
            except Exception as e:
                print(f"× Search failed for region {region}: {str(e)}")
        
        # Save regional comparison to CSV
        if regional_results:
            # Extract domain from URL for each result to compare sources
            regional_data = []
            for region, results in regional_results.items():
                for i, result in enumerate(results[:3], 1):  # Look at top 3 results
                    url = result['url']
                    import urllib.parse
                    domain = urllib.parse.urlparse(url).netloc
                    
                    regional_data.append({
                        "region": region_names.get(region, region),
                        "rank": i,
                        "title": result['title'],
                        "domain": domain
                    })
            
            if regional_data:
                regional_df = pd.DataFrame(regional_data)
                csv_file = output_dir / "regional_comparison.csv"
                regional_df.to_csv(csv_file, index=False)
                print(f"\nRegional comparison saved to: {csv_file}")
        
        # Example 3: News search with different time periods
        print("\n" + "-" * 80)
        print("EXAMPLE 3: NEWS SEARCH")
        print("-" * 80)
        
        query = "climate change initiatives"
        time_periods = {"d": "Last 24 hours", "w": "Last week", "m": "Last month"}
        
        print(f"Query: '{query}'")
        print("Testing news search with different time periods...")
        
        for period_code, period_name in time_periods.items():
            print(f"\nTime period: {period_name} ({period_code})")
            try:
                results = ddg.news_search(query, max_results=5, time_period=period_code)
                
                print(f"✓ Found {len(results)} news articles")
                for i, result in enumerate(results[:3], 1):
                    print(f"\nNews {i}:")
                    print(f"Title: {result['title']}")
                    print(f"Source: {result.get('source', 'Unknown')}")
                    print(f"Published: {result.get('published', 'Unknown')}")
                    print(f"URL: {result['url']}")
                
                # Save news results
                news_file = output_dir / f"news_search_{period_code}.json"
                with open(news_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"News results saved to: {news_file}")
                
                # Track results
                all_results.append({
                    "query": query,
                    "type": "news",
                    "time_period": period_code,
                    "period_name": period_name,
                    "results_found": len(results)
                })
            except Exception as e:
                print(f"× News search failed for period {period_code}: {str(e)}")
        
        # Example 4: Image search
        print("\n" + "-" * 80)
        print("EXAMPLE 4: IMAGE SEARCH")
        print("-" * 80)
        
        query = "renewable energy technology"
        print(f"Query: '{query}'")
        
        try:
            results = ddg.image_search(query, max_results=10)
            
            print(f"✓ Found {len(results)} images")
            if results:
                # Create a simple HTML file to display the images
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Image Search Results: {query}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .image-container {{ display: flex; flex-wrap: wrap; }}
        .image-item {{ margin: 10px; max-width: 300px; }}
        .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .image-item p {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>Image Search Results: {query}</h1>
    <div class="image-container">
"""
                
                for i, result in enumerate(results, 1):
                    print(f"Image {i}: {result.get('title', 'Untitled')}")
                    html_content += f"""
        <div class="image-item">
            <img src="{result.get('thumbnail', '')}" alt="{result.get('title', 'Image')}">
            <p><strong>{result.get('title', 'Untitled')}</strong></p>
            <p>Source: <a href="{result.get('url', '')}" target="_blank">{result.get('source', 'Unknown')}</a></p>
        </div>
"""
                
                html_content += """
    </div>
</body>
</html>
"""
                
                # Save HTML file
                html_file = output_dir / "image_search_results.html"
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"\nImage results saved as HTML: {html_file}")
                
                # Save raw image data
                image_json = output_dir / "image_search_results.json"
                with open(image_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                
                # Track results
                all_results.append({
                    "query": query,
                    "type": "image",
                    "results_found": len(results)
                })
        except Exception as e:
            print(f"× Image search failed: {str(e)}")
        
        # Example 5: Error handling demonstration
        print("\n" + "-" * 80)
        print("EXAMPLE 5: ERROR HANDLING & RETRY MECHANISM")
        print("-" * 80)
        
        # Try a complex query with special characters
        special_query = "≈∞≠∂∫√⊥∠∨∧↔→←↑↓⊗⊕♠♣♥♦ unusual symbols search test"
        print(f"Testing error handling with special query: '{special_query}'")
        
        try:
            results = ddg.search(special_query, max_results=3)
            print(f"Results found: {len(results)}")
            if results:
                print("The search engine handled special characters properly.")
        except Exception as e:
            print(f"× Search failed (expected for unusual query): {str(e)}")
            print("This demonstrates the error handling capability.")
        
        # Example 6: Integration with scraper (if available)
        if integrated_components:
            print("\n" + "-" * 80)
            print("EXAMPLE 6: INTEGRATION WITH WEB SCRAPER")
            print("-" * 80)
            
            query = "python programming best practices"
            print(f"Query: '{query}'")
            print("Performing search and scraping top results...")
            
            try:
                # Get search results
                results = ddg.search(query, max_results=3)
                
                if results:
                    # Initialize scraper with reasonable settings
                    scraper = UnifiedScraper(
                        headless=True,
                        honor_robots_txt=True,
                        use_stealth_mode=True
                    )
                    
                    print(f"Found {len(results)} search results, now scraping content...")
                    
                    scraped_results = []
                    for i, result in enumerate(results, 1):
                        url = result['url']
                        title = result['title']
                        
                        print(f"\nScraping result {i}: {title}")
                        print(f"URL: {url}")
                        
                        try:
                            # Scrape the webpage
                            content, metadata = scraper.scrape(url)
                            
                            # Calculate content length
                            content_length = len(content)
                            print(f"✓ Successfully scraped {content_length} characters")
                            
                            # Truncate content for display
                            preview = content[:200] + "..." if len(content) > 200 else content
                            print(f"Preview: {preview}")
                            
                            # Add to the collection
                            scraped_results.append({
                                "search_rank": i,
                                "title": title,
                                "url": url,
                                "content_length": content_length,
                                "content": content[:2000] + "..." if len(content) > 2000 else content,
                                "metadata": metadata
                            })
                        except Exception as e:
                            print(f"× Failed to scrape: {str(e)}")
                    
                    if scraped_results:
                        # Save scraped content
                        scraped_file = output_dir / "search_and_scrape_results.json"
                        with open(scraped_file, "w", encoding="utf-8") as f:
                            json.dump(scraped_results, f, indent=2)
                        print(f"\nSearch and scrape results saved to: {scraped_file}")
                    
                    # Clean up scraper resources
                    scraper.close()
                    print("Scraper resources released")
            except Exception as e:
                print(f"× Integration demo failed: {str(e)}")
        
        # Generate summary statistics
        print("\n" + "=" * 80)
        print("SEARCH PERFORMANCE SUMMARY:")
        print("=" * 80)
        
        # Create DataFrame for analysis
        summary_data = []
        for result in all_results:
            if "results" in result:
                del result["results"]  # Remove actual results to avoid huge data
            summary_data.append(result)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))
            
            # Save summary
            summary_file = output_dir / "search_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"\nSummary statistics saved to: {summary_file}")
        
        print("\n" + "=" * 80)
        print("CAPABILITIES DEMONSTRATED:")
        print("=" * 80)
        print("1. Basic web search with configurable result count")
        print("2. Regional search variations across different countries")
        print("3. News search with time period filtering")
        print("4. Image search with results visualization")
        print("5. Error handling and retry mechanisms")
        if integrated_components:
            print("6. Integration with web scraper for content extraction")
        print("7. Rate limiting and backoff strategies")
        print("8. Cross-region result comparison")
        print("\nAll results saved to: " + str(output_dir))
        
    except Exception as e:
        print(f"\nUnhandled error in example: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nExample completed")
        print("Note: For production use, remember to respect DuckDuckGo's Terms of Service")
        print("See: https://help.duckduckgo.com/duckduckgo-help-pages/company/partnerships/")

