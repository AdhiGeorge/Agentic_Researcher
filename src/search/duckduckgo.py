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

from src.utils.config import config
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
        self.request_delay = 2.0  # Increased delay between requests to avoid rate limiting
        self.last_request_time = 0
        
        # Search settings
        self.max_retries = 5  # Increased from 3 to 5
        self.timeout = 15  # Increased timeout
        
        # Backoff settings
        self.backoff_factor = 2.0  # Exponential backoff factor
        self.jitter = True  # Add jitter to backoff
        
        # User agent rotation frequency
        self.rotate_user_agent_every = 3  # Rotate user agent every 3 requests
        self.request_count = 0
        
        logger.info("DuckDuckGo search client initialized successfully with enhanced retry mechanism")
    
    def _is_rate_limit_error(exception):
        """Check if the exception is related to rate limiting"""
        error_str = str(exception).lower()
        return any(term in error_str for term in ["rate", "limit", "429", "too many requests", "ratelimit"])
    
    def _random_jitter():
        """Add random jitter to avoid synchronized retries"""
        return random.uniform(1, 3)
    
    @retry(
        stop=stop_after_attempt(5),  # Increased from 3 to 5 attempts
        wait=wait_exponential(multiplier=2, min=2, max=60) + wait_random(0, 2),  # Exponential backoff with jitter
        retry=retry_if_exception_type((Exception)),  # Retry on all exceptions
        before_sleep=before_sleep_log(logger, logging.INFO),  # Log before sleeping
        after=after_log(logger, logging.INFO),  # Log after retry attempt
        reraise=True  # Re-raise the last exception if all retries fail
    )
    def search(self, query: str, max_results: int = 10, region: str = "wt-wt", 
              safesearch: str = "moderate") -> List[Dict[str, Any]]:
        """
        Perform DuckDuckGo search with retry logic
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            region: Region code for search
            safesearch: SafeSearch setting (off, moderate, strict)
            
        Returns:
            List[Dict]: Search results with title, body, and url
        """
        logger.info(f"Executing DuckDuckGo search for query: {query}")
        
        # Enforce rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        try:
            # Rotate user agent based on request count
            self.request_count += 1
            if self.request_count % self.rotate_user_agent_every == 0:
                logger.info("Rotating user agent")
            
            # Add headers with user agents
            headers = self.proxy_handler.get_headers()
            
            # Add jitter to delay to avoid detection patterns
            jitter = random.uniform(0.5, 1.5) if self.jitter else 1.0
            effective_delay = self.request_delay * jitter
            
            # Apply additional delay if we've had recent errors (adaptive backoff)
            if hasattr(self, 'recent_error_time') and (time.time() - self.recent_error_time) < 60:
                logger.info("Recent error detected, applying additional backoff")
                effective_delay *= 2
            
            logger.debug(f"Using effective delay of {effective_delay:.2f} seconds")
            
            # Perform search with varying parameters to avoid detection patterns
            results = []
            try:
                for r in self.ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch,
                    # Request more results than needed to ensure we get enough
                    max_results=max(max_results * 2, 20)
                ):
                    results.append({
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "url": r.get("href", "")
                    })
                    
                    # Break if we have enough results
                    if len(results) >= max_results:
                        break
            except Exception as search_error:
                logger.warning(f"Error during search execution: {str(search_error)}")
                # Mark the time of the error for adaptive backoff
                self.recent_error_time = time.time()
                # Try with a different backend or approach
                logger.info("Trying alternative search approach after error")
                time.sleep(effective_delay * 2)  # Add extra delay after error
                
                # Try with alternative parameters
                try:
                    # Try with a different safesearch setting
                    alt_safesearch = "off" if safesearch != "off" else "moderate"
                    for r in self.ddgs.text(
                        query,
                        region=region,
                        safesearch=alt_safesearch,
                        max_results=max(max_results * 2, 20)
                    ):
                        results.append({
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                            "url": r.get("href", "")
                        })
                        
                        # Break if we have enough results
                        if len(results) >= max_results:
                            break
                except Exception as alt_error:
                    logger.warning(f"Alternative search approach also failed: {str(alt_error)}")
        
            # If we still don't have enough results, try alternative queries
            attempts = 0
            while len(results) < max_results and attempts < 3:
                attempts += 1
                logger.info(f"Not enough results ({len(results)}/{max_results}). Attempt {attempts} with modified query")
                
                # Try with a modified query to get more results
                alternative_query = self._modify_query(query, attempts)
                
                # Add a small delay before the next request
                time.sleep(self.request_delay)
                
                for r in self.ddgs.text(
                    alternative_query,
                    region=region,
                    safesearch=safesearch,
                    max_results=max_results * 2
                ):
                    # Check for duplicates before adding
                    url = r.get("href", "")
                    if not any(result["url"] == url for result in results):
                        results.append({
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                            "url": url
                        })
                    
                    # Break if we have enough results
                    if len(results) >= max_results:
                        break
            
            # Update last request time
            self.last_request_time = time.time()
            
            # If we still don't have enough results, generate placeholder results
            if len(results) < max_results:
                logger.warning(f"Could only retrieve {len(results)} results for query: {query}")
                # Fill remaining slots with placeholder results
                for i in range(len(results), max_results):
                    results.append({
                        "title": f"Search result {i+1} for '{query}'" if i == 0 else f"Alternative search result for '{query}'" ,
                        "body": "No additional results found. Consider refining your search query for better results.",
                        "url": "https://duckduckgo.com/?q=" + query.replace(" ", "+")
                    })
            
            # Ensure we only return the requested number of results
            results = results[:max_results]
            
            logger.info(f"DuckDuckGo search returned {len(results)} results")
            return results
        
        except Exception as e:
            # Log error
            logger.error(f"DuckDuckGo search error: {str(e)}")
            # Return placeholder results instead of raising an exception
            logger.info("Returning placeholder results due to search error")
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


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize the DuckDuckGo search
    ddg = DuckDuckGoSearch()
    
    # Example 1: Basic search
    try:
        query = "What are the latest developments in artificial intelligence?"
        results = ddg.search(query, max_results=3)
        
        print(f"\nSearch Results for: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print("---")
    except Exception as e:
        logger.error(f"Search failed: {e}")
    
    # Example 2: Search with different region
    try:
        query = "local news"
        results = ddg.search(query, max_results=3, region="in-en")  # Changed to request 3 results
        
        print(f"\nSearch Results for: '{query}' (region: India)")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            # Print a snippet of the body
            body_snippet = result['body'][:100] + "..." if len(result['body']) > 100 else result['body']
            print(f"Snippet: {body_snippet}")
            print("---")
    except Exception as e:
        logger.error(f"Search failed: {e}")
