"""
Unified Web Scraper for Agentic Researcher

This module implements a comprehensive web scraper with:
- Playwright-based browser automation
- Multiple fallback mechanisms
- Anti-detection and stealth mode
- User-agent and proxy rotation
- Respect for robots.txt and rate limits
- Advanced error handling and retries
- Content extraction and cleaning
"""
import os
import time
import random
import json
import re
import logging
import asyncio
import concurrent.futures
import tempfile
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional, Tuple, Union

# Playwright for browser automation
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
from playwright.async_api import async_playwright
import aiohttp
from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException

# Try to import reppy for robots.txt, but provide a fallback
try:
    from reppy.robots import Robots
    REPPY_AVAILABLE = True
except ImportError:
    REPPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("reppy module not available. Using simplified robots.txt handling.")

# Import project utilities
from ..utils.config import config

# Configure logging
logger = logging.getLogger(__name__)


class UnifiedScraper:
    """
    Unified web scraper with comprehensive capabilities
    
    This class provides:
    1. Browser-based and HTTP-based scraping
    2. Anti-detection and stealth features
    3. User-agent and proxy rotation
    4. Robots.txt handling and rate limiting
    5. Advanced content extraction
    6. Parallel scraping capabilities
    7. Database integration
    """
    
    # List of user agents to rotate through (updated for 2025)
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 OPR/108.0.0.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    ]
    
    def __init__(self, headless: bool = True, use_proxies: bool = False, 
                 rotate_user_agents: bool = True, min_delay: float = 1.0, 
                 max_delay: float = 3.0, honor_robots_txt: bool = False,
                 use_stealth_mode: bool = True):
        """
        Initialize the unified scraper
        
        Args:
            headless: Whether to run the browser in headless mode
            use_proxies: Whether to use proxy rotation
            rotate_user_agents: Whether to rotate user agents for each request
            min_delay: Minimum delay between requests to the same domain (seconds)
            max_delay: Maximum delay between requests to the same domain (seconds)
            honor_robots_txt: Whether to respect robots.txt rules
            use_stealth_mode: Whether to use stealth mode to avoid detection
        """
        self.logger = logging.getLogger("search.unified_scraper")
        
        # Use global config
        self.config = config
        
        # Browser settings
        self.headless = headless
        self.use_stealth_mode = use_stealth_mode
        
        # User-agent and proxy settings
        self.rotate_user_agents = rotate_user_agents
        self.use_proxies = use_proxies
        self.proxies = []
        if use_proxies:
            self._load_proxies()
        
        # Robots.txt and rate limiting
        self.honor_robots_txt = honor_robots_txt
        self.robots_cache = {}
        self.domain_delays = {}
        self.min_domain_delay = min_delay
        self.max_domain_delay = max_delay
        
        # Playwright objects
        self.playwright = None
        self.browser = None
        self.context = None
        self._title = None
        
        # Session management and rate limits
        self.sessions = {}
        self.rate_limits = {}
        
        # Track temporary files
        self.temp_files = []
        
        # Maximum retries for failed requests
        self.max_retries = 3
        
        # Tracker for requests to implement user agent rotation
        self.request_count = 0
        self.rotate_user_agent_every = 5
        
        logger.info(f"UnifiedScraper initialized (headless={headless}, proxies={'enabled' if use_proxies else 'disabled'}, robots_txt={honor_robots_txt})")
        if not honor_robots_txt:
            logger.warning("Robots.txt restrictions are being bypassed for research purposes")
    
    def _load_proxies(self) -> None:
        """Load proxies from configuration or file."""
        try:
            # Try to load from config
            proxy_list = getattr(self.config, "proxies", [])
            if proxy_list:
                self.proxies = proxy_list
                self.logger.info(f"Loaded {len(self.proxies)} proxies from config")
                return
            
            # Try to load from file
            proxy_file = "proxies.txt"
            if os.path.exists(proxy_file):
                with open(proxy_file, 'r') as f:
                    self.proxies = [line.strip() for line in f if line.strip()]
                self.logger.info(f"Loaded {len(self.proxies)} proxies from file")
        except Exception as e:
            self.logger.error(f"Error loading proxies: {str(e)}")
            self.proxies = []
    
    def _get_random_user_agent(self) -> str:
        """
        Get a random user agent from the list.
        
        Returns:
            str: Random user agent string
        """
        return random.choice(self.USER_AGENTS) if self.rotate_user_agents else self.USER_AGENTS[0]
    
    def _get_random_proxy(self) -> Optional[str]:
        """
        Get a random proxy from the list.
        
        Returns:
            Optional[str]: Random proxy or None if not using proxies
        """
        if not self.use_proxies or not self.proxies:
            return None
        return random.choice(self.proxies)
        
    def _check_robots_txt(self, url: str, user_agent: str) -> bool:
        """
        Check if the URL is allowed by robots.txt.
        
        Args:
            url: URL to check
            user_agent: User agent to check against
            
        Returns:
            bool: True if allowed, False if disallowed
        """
        if not self.honor_robots_txt:
            return True
            
        try:
            # Extract domain from URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            scheme = parsed_url.scheme
            
            # Check if we have a cached result
            if domain in self.robots_cache:
                robots_parser = self.robots_cache[domain]
                path = parsed_url.path
                if not path:
                    path = "/"
                return robots_parser.allowed(url, user_agent)
            
            # Fetch robots.txt
            robots_url = f"{scheme}://{domain}/robots.txt"
            
            if REPPY_AVAILABLE:
                # Use reppy for better parsing
                try:
                    robots_parser = Robots.fetch(robots_url)
                    self.robots_cache[domain] = robots_parser
                    path = parsed_url.path
                    if not path:
                        path = "/"
                    return robots_parser.allowed(url, user_agent)
                except Exception as e:
                    self.logger.warning(f"Error fetching robots.txt from {robots_url}: {str(e)}")
                    return True  # Allow by default if robots.txt can't be fetched
            else:
                # Simplified implementation without reppy
                try:
                    response = requests.get(robots_url, timeout=5)
                    if response.status_code != 200:
                        return True  # Allow if robots.txt not found
                    
                    # Basic parsing of robots.txt
                    lines = response.text.split('\n')
                    current_agent = None
                    disallowed_paths = []
                    
                    for line in lines:
                        line = line.strip().lower()
                        if not line or line.startswith('#'):
                            continue
                            
                        parts = line.split(':', 1)
                        if len(parts) != 2:
                            continue
                            
                        key, value = parts[0].strip(), parts[1].strip()
                        
                        if key == 'user-agent':
                            if value == '*' or value in user_agent.lower():
                                current_agent = value
                            else:
                                current_agent = None
                        elif key == 'disallow' and current_agent is not None:
                            if value:
                                disallowed_paths.append(value)
                    
                    # Check if path is disallowed
                    path = parsed_url.path
                    if not path:
                        path = "/"
                        
                    for disallowed in disallowed_paths:
                        if path.startswith(disallowed):
                            return False
                    
                    return True
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching robots.txt from {robots_url}: {str(e)}")
                    return True  # Allow by default if robots.txt can't be fetched
        except Exception as e:
            self.logger.error(f"Error checking robots.txt for {url}: {str(e)}")
            return True  # Allow by default on errors
    
    def _get_random_delay(self, domain: str) -> float:
        """
        Get a random delay with exponential backoff if needed.
        
        Args:
            domain: Domain being accessed
            
        Returns:
            float: Delay in seconds
        """
        # Check if domain has existing delay data
        now = time.time()
        if domain in self.domain_delays:
            last_access, backoff_factor = self.domain_delays[domain]
            
            # Calculate time since last access
            time_since_last = now - last_access
            
            # If accessed too recently, increase backoff
            if time_since_last < self.min_domain_delay:
                backoff_factor = min(backoff_factor * 1.5, 10.0)
                delay = self.min_domain_delay + random.uniform(0, self.max_domain_delay) * backoff_factor
            else:
                # Reset backoff if sufficient time has passed
                backoff_factor = 1.0
                delay = random.uniform(self.min_domain_delay, self.max_domain_delay)
        else:
            # First access to this domain
            backoff_factor = 1.0
            delay = random.uniform(self.min_domain_delay, self.max_domain_delay)
        
        # Update domain delay data
        self.domain_delays[domain] = (now, backoff_factor)
        
        return delay
    
    def _setup_browser(self) -> None:
        """Set up the browser if it's not already running"""
        if self.browser is not None:
            return
            
        try:
            # Create a new playwright instance
            self.playwright = sync_playwright().start()
            
            # Get browser type
            browser_type = self.playwright.chromium
            
            # Configure browser arguments
            if self.use_stealth_mode:
                # Stealth mode with anti-detection measures
                browser_args = [
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--window-size=1920,1080'
                ]
                logger.info("Using stealth mode to avoid detection")
            else:
                # Basic browser arguments
                browser_args = [
                    '--no-sandbox',
                    '--disable-dev-shm-usage'
                ]
            
            # Launch browser
            self.browser = browser_type.launch(
                headless=self.headless,
                args=browser_args
            )
            
            # Create a browser context with specific options
            context_options = {
                'viewport': {'width': 1920, 'height': 1080},
                'user_agent': self._get_random_user_agent(),
                'ignore_https_errors': True
            }
            
            # Add proxy if configured
            proxy = self._get_random_proxy()
            if proxy:
                context_options['proxy'] = {'server': proxy}
            
            # Create browser context
            self.context = self.browser.new_context(**context_options)
            
            # Add additional scripts for stealth mode
            if self.use_stealth_mode:
                # Add stealth script that modifies JS environment to avoid detection
                self.context.add_init_script("""
                () => {
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => false,
                    });
                    
                    // Add plugins for more realistic browser fingerprint
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5].map(() => ({
                            description: 'Plugin',
                            filename: 'plugin.dll',
                            name: 'Plugin'
                        })),
                    });
                    
                    // Modify the chrome property
                    window.chrome = {
                        app: {
                            isInstalled: false,
                        },
                        runtime: {}
                    };
                }
                """)
            
            logger.info(f"Browser setup complete (headless: {self.headless})")
        except Exception as e:
            logger.error(f"Error setting up browser: {str(e)}")
            raise
            
    def scrape(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Scrape content from a URL using the appropriate scraping method.
        
        Args:
            url: URL to scrape
            
        Returns:
            Tuple containing (content, metadata)
        """
        self.logger.info(f"Scraping URL: {url}")
        
        # Get domain for rate limiting
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Get user agent for this request
        self.request_count += 1
        if self.rotate_user_agents and self.request_count % self.rotate_user_agent_every == 0:
            user_agent = self._get_random_user_agent()
        else:
            user_agent = self.USER_AGENTS[0] if not self.rotate_user_agents else self._get_random_user_agent()
        
        # Check robots.txt
        if not self._check_robots_txt(url, user_agent):
            self.logger.warning(f"URL {url} is disallowed by robots.txt")
            return "", {"status": "disallowed_by_robots_txt", "url": url}
            
        # Apply rate limiting
        delay = self._get_random_delay(domain)
        if delay > 0:
            self.logger.debug(f"Waiting {delay:.2f}s before accessing {domain}")
            time.sleep(delay)
        
        # Determine scraping method based on URL or content type
        if url.lower().endswith(('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx')):
            return self._scrape_document(url)
        else:
            # Try browser-based scraping with fallback to HTTP
            try:
                # Start with browser-based scraping for JS-heavy sites
                content, metadata = self._scrape_with_browser(url, user_agent)
                
                # Check if content is empty or minimal
                if not content or len(content.strip()) < 100:
                    self.logger.info("Browser scraping returned minimal content, trying HTTP fallback")
                    http_content, http_metadata = self._scrape_with_http(url, user_agent)
                    
                    # Use HTTP result if it's better
                    if len(http_content.strip()) > len(content.strip()):
                        return http_content, http_metadata
                
                return content, metadata
            except Exception as e:
                self.logger.warning(f"Browser scraping failed: {str(e)}, trying HTTP fallback")
                try:
                    return self._scrape_with_http(url, user_agent)
                except Exception as http_e:
                    self.logger.error(f"All scraping methods failed for {url}: {str(http_e)}")
                    return "", {"status": "error", "url": url, "error": str(http_e)}
    
    def _scrape_with_browser(self, url: str, user_agent: str) -> Tuple[str, Dict[str, Any]]:
        """
        Scrape a URL using a browser.
        
        Args:
            url: URL to scrape
            user_agent: User agent to use
            
        Returns:
            Tuple containing (content, metadata)
        """
        # Set up browser if not already done
        if self.browser is None:
            self._setup_browser()
            
        # Create a new page
        page = self.context.new_page()
        
        # Initialize metadata
        metadata = {
            "url": url,
            "content_type": "text/html",
            "status": "success",
            "scraping_method": "browser"
        }
        
        try:
            # Set longer timeout for complex pages
            page.set_default_timeout(30000)  # 30 seconds
            
            # Navigate to the URL
            response = page.goto(url, wait_until="networkidle")
            
            # Update metadata
            metadata["status_code"] = response.status if response else 0
            
            # Wait for content to load
            page.wait_for_load_state("networkidle")
            
            # Get page title
            metadata["title"] = page.title()
            
            # Get content type
            response_headers = response.headers if response else {}
            metadata["content_type"] = response_headers.get("content-type", "text/html")
            
            # Handle non-HTML content types
            if metadata["content_type"] and not metadata["content_type"].startswith("text/html"):
                # For binary files, download and extract later
                if metadata["content_type"].startswith(("application/", "image/")):
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_path = temp_file.name
                        self.temp_files.append(temp_path)
                        
                        # Download the file
                        response.body().save_as(temp_path)
                        metadata["temp_file"] = temp_path
                        
                        return "", metadata
            
            # Extract main content
            content = ""
            
            # Try multiple content extraction strategies
            # 1. Get all text content
            content = page.content()
            
            # 2. Try to identify main content container
            main_content_selectors = [
                "article", "#content", ".content", "main", ".main", "#main", ".post", "#post",
                ".article", "#article", ".post-content", ".entry-content"
            ]
            
            for selector in main_content_selectors:
                if page.locator(selector).count() > 0:
                    main_content = page.locator(selector).first.inner_html()
                    if main_content and len(main_content) > 100:
                        content = main_content
                        break
            
            # Clean the content with BeautifulSoup
            if content:
                soup = BeautifulSoup(content, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                # Get text
                text_content = soup.get_text(separator=" ", strip=True)
                
                # Clean whitespace
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                
                content = text_content
            
            return content, metadata
            
        except Exception as e:
            self.logger.error(f"Error during browser scraping: {str(e)}")
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return "", metadata
        finally:
            # Close the page
            page.close()
            
    def _scrape_with_http(self, url: str, user_agent: str) -> Tuple[str, Dict[str, Any]]:
        """
        Scrape a URL using HTTP requests.
        
        Args:
            url: URL to scrape
            user_agent: User agent to use
            
        Returns:
            Tuple containing (content, metadata)
        """
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        # Initialize metadata
        metadata = {
            "url": url,
            "content_type": "text/html",
            "status": "success",
            "scraping_method": "http"
        }
        
        # Get proxy if needed
        proxy = self._get_random_proxy()
        proxies = {"http": proxy, "https": proxy} if proxy else None
        
        try:
            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(
                        url, 
                        headers=headers, 
                        proxies=proxies, 
                        timeout=15,
                        verify=False  # Allow self-signed certificates
                    )
                    
                    break  # Break if successful
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise  # Re-raise if last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Update metadata
            metadata["status_code"] = response.status_code
            metadata["content_type"] = response.headers.get("Content-Type", "text/html")
            
            # Check status code
            if response.status_code != 200:
                metadata["status"] = "error"
                metadata["error"] = f"HTTP error {response.status_code}"
                return "", metadata
                
            # Handle different content types
            content_type = metadata["content_type"].lower()
            
            # Handle HTML
            if "text/html" in content_type:
                # Extract title
                soup = BeautifulSoup(response.text, "html.parser")
                title_tag = soup.find("title")
                metadata["title"] = title_tag.get_text() if title_tag else ""
                
                # Extract main content
                main_content_tags = [
                    "article", "main", "#content", ".content", ".post", ".article",
                    ".post-content", ".entry-content"
                ]
                
                main_content = None
                for tag in main_content_tags:
                    content_element = soup.select_one(tag)
                    if content_element:
                        main_content = content_element
                        break
                
                if main_content:
                    # Remove script and style tags
                    for script in main_content(["script", "style"]):
                        script.decompose()
                    content = main_content.get_text(separator=" ", strip=True)
                else:
                    # Fallback to full page with cleaning
                    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        script.decompose()
                    content = soup.get_text(separator=" ", strip=True)
                
                # Clean whitespace
                content = re.sub(r'\s+', ' ', content).strip()
                
                return content, metadata
                
            # Handle binary files - save to temp file
            elif "application/" in content_type or "image/" in content_type:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                    self.temp_files.append(temp_path)
                    metadata["temp_file"] = temp_path
                    
                return "", metadata
                
            # Handle plain text
            elif "text/" in content_type:
                content = response.text
                return content, metadata
                
            # Unknown type
            else:
                metadata["status"] = "error"
                metadata["error"] = f"Unsupported content type: {content_type}"
                return "", metadata
                
        except Exception as e:
            self.logger.error(f"Error during HTTP scraping: {str(e)}")
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return "", metadata
            
    def _scrape_document(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Scrape content from a document URL (PDF, DOC, etc.)
        
        Args:
            url: URL to the document
            
        Returns:
            Tuple containing (content, metadata)
        """
        # Initialize metadata
        metadata = {
            "url": url,
            "status": "pending",
            "scraping_method": "document"
        }
        
        try:
            # Get file extension
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            # Determine content type based on extension
            if path.endswith(".pdf"):
                metadata["content_type"] = "application/pdf"
            elif path.endswith(".doc") or path.endswith(".docx"):
                metadata["content_type"] = "application/msword"
            elif path.endswith(".ppt") or path.endswith(".pptx"):
                metadata["content_type"] = "application/vnd.ms-powerpoint"
            elif path.endswith(".xls") or path.endswith(".xlsx"):
                metadata["content_type"] = "application/vnd.ms-excel"
            else:
                metadata["content_type"] = "application/octet-stream"
            
            # Download the file using HTTP
            user_agent = self._get_random_user_agent()
            headers = {"User-Agent": user_agent}
            
            # Get proxy if needed
            proxy = self._get_random_proxy()
            proxies = {"http": proxy, "https": proxy} if proxy else None
            
            # Download with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(
                        url, 
                        headers=headers, 
                        proxies=proxies, 
                        timeout=30,
                        verify=False,  # Allow self-signed certificates
                        stream=True     # Stream the file
                    )
                    
                    # Check status code
                    if response.status_code != 200:
                        metadata["status"] = "error"
                        metadata["error"] = f"HTTP error {response.status_code}"
                        return "", metadata
                    
                    # Get content type from response
                    metadata["content_type"] = response.headers.get("Content-Type", metadata["content_type"])
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                temp_file.write(chunk)
                        temp_path = temp_file.name
                    
                    # Track the temporary file
                    self.temp_files.append(temp_path)
                    metadata["temp_file"] = temp_path
                    metadata["status"] = "success"
                    
                    # For now, we don't extract content from the document here
                    # This would be handled by a specialized extractor like pdf_extractor.py
                    return "", metadata
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise  # Re-raise if last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            self.logger.error(f"Error during document scraping: {str(e)}")
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return "", metadata
    
    def scrape_multiple(self, urls: List[str], max_workers: int = 4) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Scrape multiple URLs in parallel
        
        Args:
            urls: List of URLs to scrape
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of tuples containing (content, metadata) for each URL
        """
        results = []
        
        # Use a smaller number of workers than requested to avoid rate limiting
        actual_workers = min(max_workers, len(urls))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all scraping tasks
            future_to_url = {executor.submit(self.scrape, url): url for url in urls}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content, metadata = future.result()
                    results.append((content, metadata))
                    self.logger.info(f"Completed scraping {url}")
                except Exception as e:
                    self.logger.error(f"Error scraping {url}: {str(e)}")
                    results.append(("", {"url": url, "status": "error", "error": str(e)}))
        
        return results
    
    async def async_scrape(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Asynchronously scrape a URL (HTTP only, no browser)
        
        Args:
            url: URL to scrape
            
        Returns:
            Tuple containing (content, metadata)
        """
        # Get user agent
        user_agent = self._get_random_user_agent()
        
        # Initialize metadata
        metadata = {
            "url": url,
            "content_type": "text/html",
            "status": "success",
            "scraping_method": "async_http"
        }
        
        # Check robots.txt
        if not self._check_robots_txt(url, user_agent):
            self.logger.warning(f"URL {url} is disallowed by robots.txt")
            return "", {"status": "disallowed_by_robots_txt", "url": url}
        
        # Apply rate limiting
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        delay = self._get_random_delay(domain)
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Set up headers
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        
        # Get proxy if needed
        proxy = self._get_random_proxy()
        
        try:
            # Make the request with retries
            for attempt in range(self.max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=headers, proxy=proxy, timeout=aiohttp.ClientTimeout(total=15)) as response:
                            # Update metadata
                            metadata["status_code"] = response.status
                            metadata["content_type"] = response.headers.get("Content-Type", "text/html")
                            
                            # Check status code
                            if response.status != 200:
                                metadata["status"] = "error"
                                metadata["error"] = f"HTTP error {response.status}"
                                return "", metadata
                            
                            # Get the content
                            content = await response.text()
                            
                            # Handle HTML content
                            if "text/html" in metadata["content_type"].lower():
                                # Parse with BeautifulSoup
                                soup = BeautifulSoup(content, "html.parser")
                                
                                # Extract title
                                title_tag = soup.find("title")
                                metadata["title"] = title_tag.get_text() if title_tag else ""
                                
                                # Clean the content
                                for script in soup(["script", "style", "nav", "footer", "header"]):
                                    script.decompose()
                                
                                # Get text
                                text_content = soup.get_text(separator=" ", strip=True)
                                text_content = re.sub(r'\s+', ' ', text_content).strip()
                                
                                return text_content, metadata
                            else:
                                return content, metadata
                            
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        except Exception as e:
            self.logger.error(f"Error during async scraping: {str(e)}")
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return "", metadata
    
    async def async_scrape_multiple(self, urls: List[str], max_concurrent: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Scrape multiple URLs asynchronously
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of tuples containing (content, metadata) for each URL
        """
        results = []
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_scrape(url):
            async with semaphore:
                return await self.async_scrape(url)
        
        # Create tasks for all URLs
        tasks = [bounded_scrape(url) for url in urls]
        
        # Wait for all tasks to complete
        for task in asyncio.as_completed(tasks):
            try:
                content, metadata = await task
                results.append((content, metadata))
            except Exception as e:
                # Add failed result
                results.append(("", {"status": "error", "error": str(e)}))
        
        return results
    
    def close(self) -> None:
        """
        Close all resources
        """
        # Close browser
        if self.browser:
            try:
                self.browser.close()
                self.browser = None
            except Exception as e:
                self.logger.error(f"Error closing browser: {str(e)}")
        
        # Close playwright
        if self.playwright:
            try:
                self.playwright.stop()
                self.playwright = None
            except Exception as e:
                self.logger.error(f"Error closing playwright: {str(e)}")
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                self.logger.error(f"Error removing temporary file {temp_file}: {str(e)}")
        
        self.temp_files = []
        self.logger.info("UnifiedScraper closed")
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage when run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create scraper with robots.txt bypassed for research purposes
    scraper = UnifiedScraper(headless=True, honor_robots_txt=False)
    
    try:
        # URLs to test
        test_urls = [
            "https://en.wikipedia.org/wiki/Volatility_Index",
            "https://www.cnbc.com/markets/",
            "https://github.com/features"
        ]
        
        # Single URL
        url = test_urls[0]
        print(f"\nScraping {url}...")
        content, metadata = scraper.scrape(url)
        
        print(f"Title: {metadata.get('title')}")
        print(f"Status: {metadata.get('status')}")
        print(f"Content Type: {metadata.get('content_type')}")
        print(f"Content Length: {len(content)} characters")
        print(f"Preview: {content[:200]}...")
        
        # Multiple URLs
        print("\nScraping multiple URLs...")
        results = scraper.scrape_multiple(test_urls)
        
        for i, (content, metadata) in enumerate(results):
            print(f"\nResult {i+1}: {metadata.get('url')}")
            print(f"Title: {metadata.get('title')}")
            print(f"Status: {metadata.get('status')}")
            print(f"Content Length: {len(content)} characters")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        scraper.close()
