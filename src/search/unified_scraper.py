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
import warnings
import certifi
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional, Tuple, Union

# Playwright for browser automation
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
from playwright.async_api import async_playwright
import aiohttp
from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException

# Suppress InsecureRequestWarning
from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Try to import reppy for robots.txt, but provide a fallback
try:
    from reppy.robots import Robots
    REPPY_AVAILABLE = True
except ImportError:
    REPPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("reppy module not available. Using simplified robots.txt handling.")

from src.utils.config import Config as config
import types

config = types.SimpleNamespace()
config.proxies = []
config.user_agents = []
# Configure logging
logger = logging.getLogger(__name__)

# Try to import PDF extraction functionality
try:
    from .pdf_extractor import PDFExtractor
    PDF_EXTRACTOR_AVAILABLE = True
except ImportError:
    try:
        from src.search.pdf_extractor import PDFExtractor
        PDF_EXTRACTOR_AVAILABLE = True
    except ImportError:
        PDF_EXTRACTOR_AVAILABLE = False
        logger.warning("PDF extraction functionality not available")


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
                 use_stealth_mode: bool = True, verify_ssl: bool = True):
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
            verify_ssl: Whether to verify SSL certificates
        """
        self.logger = logging.getLogger("search.unified_scraper")
        
        # Use global config
        self.config = config
        
        # Browser settings
        self.headless = headless
        self.verify_ssl = verify_ssl
        
        # Initialize PDF extractor if available
        self.pdf_extractor = None
        if PDF_EXTRACTOR_AVAILABLE:
            self.pdf_extractor = PDFExtractor()
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
        # Use a lock to prevent multiple threads from setting up browsers simultaneously
        try:
            # Check if browser is already running before trying to set it up
            if self.browser is not None:
                return
                
            # If another thread is setting up the browser, wait a bit
            if hasattr(self, '_browser_setup_in_progress') and self._browser_setup_in_progress:
                logger.info("Browser setup in progress in another thread, waiting...")
                for _ in range(10):  # Try for up to 5 seconds
                    time.sleep(0.5)
                    if self.browser is not None:
                        return
                        
            # Mark that we're setting up the browser
            self._browser_setup_in_progress = True
            
            try:
                # Create a new playwright instance
                self.playwright = sync_playwright().start()
                
                # Get browser type
                browser_type = self.playwright.chromium
                
                # Configure browser arguments for stability
                if self.use_stealth_mode:
                    # Stealth mode with anti-detection measures
                    browser_args = [
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-extensions',
                        '--disable-background-networking',
                        '--disable-background-timer-throttling',
                        '--disable-backgrounding-occluded-windows',
                        '--disable-breakpad',
                        '--disable-component-extensions-with-background-pages',
                        '--disable-features=TranslateUI,BlinkGenPropertyTrees',
                        '--disable-ipc-flooding-protection',
                        '--disable-renderer-backgrounding',
                        '--enable-features=NetworkService,NetworkServiceInProcess',
                        '--force-color-profile=srgb',
                        '--metrics-recording-only',
                        '--mute-audio',
                        '--window-size=1920,1080'
                    ]
                    logger.info("Using stealth mode to avoid detection")
                else:
                    # Basic browser arguments optimized for stability
                    browser_args = [
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-extensions',
                        '--disable-gpu',
                        '--mute-audio'
                    ]
                
                # Launch browser with thread-safe configuration
                self.browser = browser_type.launch(
                    headless=self.headless,
                    args=browser_args,
                    handle_sigint=False,  # Don't kill browser on Ctrl+C
                    handle_sigterm=False,  # Don't kill browser on termination signals
                    handle_sighup=False,   # Don't kill browser on terminal close
                    ignore_default_args=['--enable-automation']
                )
                
                # Create a browser context with specific options
                context_options = {
                    'viewport': {'width': 1920, 'height': 1080},
                    'user_agent': self._get_random_user_agent(),
                    'ignore_https_errors': not self.verify_ssl,
                    'java_script_enabled': True,
                    'bypass_csp': True  # Bypass Content-Security-Policy
                }
                
                # Add proxy if configured
                proxy = self._get_random_proxy()
                if proxy:
                    context_options['proxy'] = {'server': proxy}
                
                # Create browser context
                self.context = self.browser.new_context(**context_options)
                
                # Set timeout for stability
                self.context.set_default_timeout(30000)  # 30 seconds timeout
                
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
                        
                        // Modify navigator properties
                        const originalQuery = window.navigator.permissions.query;
                        window.navigator.permissions.query = (parameters) => (
                            parameters.name === 'notifications' ?
                                Promise.resolve({ state: Notification.permission }) :
                                originalQuery(parameters)
                        );
                    }
                    """)
                
                logger.info(f"Browser setup complete (headless: {self.headless})")
            except Exception as e:
                logger.error(f"Error setting up browser: {str(e)}")
                self._browser_setup_in_progress = False
                raise
                
            # Mark setup as complete
            self._browser_setup_in_progress = False
        except Exception as e:
            logger.error(f"Error in _setup_browser: {str(e)}")
            # Reset flag in case of error
            self._browser_setup_in_progress = False
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
        
        # Check if this is a PDF URL and we have a PDF extractor
        if self.pdf_extractor and self._is_likely_pdf_url(url):
            return self._scrape_pdf(url)
            
        # Get domain for rate limiting
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Get user agent for this request
        self.request_count += 1
        if self.rotate_user_agents and self.request_count % self.rotate_user_agent_every == 0:
            user_agent = self._get_random_user_agent()
        else:
            user_agent = self.USER_AGENTS[0] if not self.rotate_user_agents else self._get_random_user_agent()
            
        # Special handling for Wikipedia
        if 'wikipedia.org' in domain:
            return self._scrape_wikipedia(url, user_agent)
            
        # Check if this is a rate-limited domain
        if domain in self.rate_limits:
            last_access, count = self.rate_limits[domain]
            now = time.time()
            
            # If accessed recently and over limit, delay
            if now - last_access < 60 and count > 10:  # 10 requests per minute max
                delay = self._get_random_delay(domain)
                self.logger.info(f"Rate limiting {domain}, sleeping for {delay:.2f}s")
                time.sleep(delay)
            
            # Update rate limit counter
            self.rate_limits[domain] = (now, count + 1)
        else:
            # Initialize rate limit counter
            self.rate_limits[domain] = (time.time(), 1)
        
        # Check robots.txt
        if not self._check_robots_txt(url, user_agent):
            self.logger.warning(f"URL {url} disallowed by robots.txt")
            
        # Try browser-based scraping first, with HTTP as fallback
        try:
            content, metadata = self._scrape_with_browser(url, user_agent)
            
            # If we got empty content, try HTTP method as well
            if not content.strip() and "text/html" in metadata.get("content_type", ""):
                try:
                    http_content, http_metadata = self._scrape_with_http(url, user_agent)
                    
                    # Use HTTP result if it has more content
                    if len(http_content.strip()) > len(content.strip()):
                        return http_content, http_metadata
                except Exception:
                    pass  # Ignore HTTP errors if browser scraping worked
            
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
            
    def _is_likely_pdf_url(self, url: str) -> bool:
        """
        Check if a URL is likely to point to a PDF.
        
        Args:
            url: URL to check
            
        Returns:
            True if likely a PDF, False otherwise
        """
        # Direct check for PDF extension
        if url.lower().endswith('.pdf'):
            return True
            
        # Check URL path
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        query = parsed_url.query.lower()
        
        # Check for PDF in path or query
        if '.pdf' in path or 'pdf=' in query or 'format=pdf' in query:
            return True
            
        return False
        
    def _scrape_pdf(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from a PDF URL.
        
        Args:
            url: URL of the PDF
            
        Returns:
            Tuple containing (text, metadata)
        """
        self.logger.info(f"Extracting PDF content from {url}")
        
        # Initialize metadata
        metadata = {
            "url": url,
            "content_type": "application/pdf",
            "status": "pending",
            "scraping_method": "pdf_extractor"
        }
        
        # Make sure we have a PDF extractor
        if not self.pdf_extractor:
            metadata["status"] = "error"
            metadata["error"] = "PDF extraction not available"
            return "", metadata
        
        try:
            # Extract text from PDF
            result = self.pdf_extractor.extract_from_url(url)
            
            if result["success"]:
                metadata["status"] = "success"
                metadata["engine_used"] = result["engine_used"]
                metadata["processing_time"] = result["processing_time"]
                metadata["title"] = self._extract_pdf_title(result["text"])
                return result["text"], metadata
            else:
                metadata["status"] = "error"
                metadata["error"] = result["error"]
                return "", metadata
                
        except Exception as e:
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return "", metadata
    
    def _extract_pdf_title(self, text: str) -> str:
        """
        Attempt to extract a title from PDF text.
        
        Args:
            text: Extracted PDF text
            
        Returns:
            Extracted title or default title
        """
        if not text:
            return "Untitled PDF"
            
        # Try to get the first non-empty line as the title
        lines = text.strip().split('\n')
        for line in lines:
            clean_line = line.strip()
            if clean_line and len(clean_line) > 3 and len(clean_line) < 100:
                return clean_line.replace('\r', '')
                
        # Fallback to first chunk of text
        first_chunk = text[:100].replace('\n', ' ').strip()
        if first_chunk:
            return first_chunk + "..."
            
        return "Untitled PDF"
    
    def _scrape_wikipedia(self, url: str, user_agent: str) -> Tuple[str, Dict[str, Any]]:
        """
        Special method for Wikipedia scraping to handle their anti-bot measures.
        
        Args:
            url: Wikipedia URL to scrape
            user_agent: User agent to use
            
        Returns:
            Tuple containing (content, metadata)
        """
        self.logger.info(f"Using specialized Wikipedia scraper for {url}")
        
        # Initialize metadata
        metadata = {
            "url": url,
            "content_type": "text/html",
            "status": "success",
            "scraping_method": "wikipedia_specialized",
            "title": "Wikipedia Article"
        }
        
        try:
            # Use the standard URL (mobile version was causing 404 errors)
            target_url = url
            
            # Advanced headers specific for Wikipedia
            headers = {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
                'TE': 'Trailers',
                'DNT': '1'
            }
            
            # Use direct requests with appropriate headers
            response = requests.get(
                target_url, 
                headers=headers, 
                timeout=30,
                verify=self.verify_ssl
            )
            
            # Check if we got a valid response
            response.raise_for_status()
            
            # Process the content
            html_content = response.text
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get metadata
            title_tag = soup.find('title')
            if title_tag:
                metadata["title"] = title_tag.get_text()
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
                
            # Get article content - main content is typically in div with id="content" or "mw-content-text"
            content_div = soup.find('div', id='content') or soup.find('div', id='mw-content-text') or soup.find('div', class_='mw-parser-output')
            
            if content_div:
                content = content_div.get_text(separator=" ", strip=True)
            else:
                # Fallback to the whole body
                content = soup.get_text(separator=" ", strip=True)
                
            # Clean whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content, metadata
            
        except Exception as e:
            self.logger.error(f"Wikipedia scraping error: {str(e)}")
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return "", metadata
    
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
            "DNT": "1",  # Do Not Track
            "Cache-Control": "max-age=0",
            "Pragma": "no-cache"
        }
        
        # Configure SSL verification
        verify = certifi.where() if self.verify_ssl else False
        
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
                        verify=verify  # Allow self-signed certificates
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
                    # Fallback to the whole body
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
                
                # Track the temporary file
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
        
        # Group URLs by type for specialized handling
        wikipedia_urls = []
        pdf_urls = []
        regular_urls = []
        
        for url in urls:
            if 'wikipedia.org' in url:
                wikipedia_urls.append(url)
            elif self.pdf_extractor and self._is_likely_pdf_url(url):
                pdf_urls.append(url)
            else:
                regular_urls.append(url)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Start with specialized scrapers
            futures = []
            
            # Handle Wikipedia URLs with specialized method
            for url in wikipedia_urls:
                user_agent = self._get_random_user_agent()
                futures.append((executor.submit(self._scrape_wikipedia, url, user_agent), url))
            
            # Handle PDF URLs
            for url in pdf_urls:
                futures.append((executor.submit(self._scrape_pdf, url), url))
            
            # Handle regular URLs with HTTP method
            for url in regular_urls:
                user_agent = self._get_random_user_agent()
                futures.append((executor.submit(self._scrape_with_http, url, user_agent), url))
            
            # Process results as they complete
            for future, url in futures:
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
                    async with aiohttp.ClientSession(
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                        cookie_jar=aiohttp.CookieJar(unsafe=True)
                    ) as session:
                        async with session.get(url, proxy=proxy, ssl=False) as response:
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
            """Helper function to apply semaphore to async_scrape"""
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
        
        # Clean up PDF extractor if available
        if self.pdf_extractor:
            try:
                self.pdf_extractor.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up PDF extractor: {str(e)}")
                
        self.logger.info("UnifiedScraper closed")
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Comprehensive example usage for UnifiedScraper
if __name__ == "__main__":
    import sys
    import os
    import pandas as pd
    from datetime import datetime
    from pathlib import Path
    
    # Add project root to Python path if necessary
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Try to import other components from the Agentic Researcher project
    try:
        from src.utils.document_utils import extract_metadata, process_document
        from src.utils.file_utils import ensure_dir
        document_utils_available = True
    except ImportError:
        document_utils_available = False
        print("Warning: document_utils module not available for integration example")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory for example results
    output_dir = Path(os.path.join(project_root, "example_results", f"scraper_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    ensure_dir(output_dir)
    
    print("\n" + "=" * 80)
    print("UNIFIED SCRAPER - COMPREHENSIVE EXAMPLE")
    print("=" * 80)
    
    print("\nInitializing UnifiedScraper with various configurations...")
    print("\n1. Basic Configuration")
    scraper = UnifiedScraper(
        headless=True,  # Run browser in headless mode
        use_proxies=False,  # Don't use proxies for this example
        rotate_user_agents=True,  # Rotate user agents for anti-detection
        min_delay=1.0,  # Wait at least 1 second between requests
        max_delay=3.0,  # Maximum delay (important for rate-limiting)
        honor_robots_txt=False,  # Set to True in production for ethical scraping
        use_stealth_mode=True,  # Enable stealth mode to avoid detection
        verify_ssl=True  # Verify SSL certificates
    )
    
    try:
        print("\nDemonstrating different URL types and scraping methods:")
        
        # 1. DEMONSTRATION: Different types of URLs to showcase various capabilities
        test_urls = {
            "wikipedia": "https://en.wikipedia.org/wiki/Natural_language_processing",
            "http_friendly": "https://news.ycombinator.com/",
            "browser_required": "https://www.reuters.com/technology/",
            "pdf_document": "https://arxiv.org/pdf/2303.08774.pdf"  # ML research paper
        }
        
        # Example 1: Individual URL scraping with method selection
        print("\n" + "-" * 80)
        print("EXAMPLE 1: SPECIALIZED WIKIPEDIA SCRAPING")
        print("-" * 80)
        wiki_url = test_urls["wikipedia"]
        print(f"Scraping Wikipedia article: {wiki_url}")
        wiki_content, wiki_metadata = scraper.scrape(wiki_url)
        
        # Display Wikipedia-specific information
        print(f"Title: {wiki_metadata.get('title')}")
        print(f"Status: {wiki_metadata.get('status')}")
        print(f"Scraping Method: {wiki_metadata.get('scraping_method')}")
        print(f"Content Length: {len(wiki_content)} characters")
        print(f"Preview: {wiki_content[:200]}...")
        
        # Save Wikipedia content to file
        wiki_output_path = output_dir / "wikipedia_nlp.txt"
        with open(wiki_output_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {wiki_metadata.get('title')}\n\n")
            f.write(wiki_content)
        print(f"Saved Wikipedia content to: {wiki_output_path}")
        
        # Example 2: HTTP-based scraping (faster for simple sites)
        print("\n" + "-" * 80)
        print("EXAMPLE 2: HTTP-BASED SCRAPING")
        print("-" * 80)
        http_url = test_urls["http_friendly"]
        print(f"Scraping with HTTP requests: {http_url}")
        
        # Force HTTP-based scraping by using internal method directly
        user_agent = scraper._get_random_user_agent()
        http_content, http_metadata = scraper._scrape_with_http(http_url, user_agent)
        
        print(f"Title: {http_metadata.get('title')}")
        print(f"Status: {http_metadata.get('status')}")
        print(f"Content Type: {http_metadata.get('content_type')}")
        print(f"Content Length: {len(http_content)} characters")
        print(f"Preview: {http_content[:200]}...")
        
        # Example 3: Browser-based scraping (for JavaScript-heavy sites)
        print("\n" + "-" * 80)
        print("EXAMPLE 3: BROWSER-BASED SCRAPING")
        print("-" * 80)
        browser_url = test_urls["browser_required"]
        print(f"Scraping with Playwright browser: {browser_url}")
        
        # Force browser-based scraping by using internal method directly
        user_agent = scraper._get_random_user_agent()
        browser_content, browser_metadata = scraper._scrape_with_browser(browser_url, user_agent)
        
        print(f"Title: {browser_metadata.get('title')}")
        print(f"Status: {browser_metadata.get('status')}")
        print(f"JavaScript-rendered: Yes")
        print(f"Content Length: {len(browser_content)} characters")
        print(f"Preview: {browser_content[:200]}...")
        
        # Example 4: PDF extraction (if available)
        if PDF_EXTRACTOR_AVAILABLE:
            print("\n" + "-" * 80)
            print("EXAMPLE 4: PDF EXTRACTION")
            print("-" * 80)
            pdf_url = test_urls["pdf_document"]
            print(f"Extracting content from PDF: {pdf_url}")
            
            try:
                pdf_content, pdf_metadata = scraper._scrape_pdf(pdf_url)
                
                print(f"Title: {pdf_metadata.get('title')}")
                print(f"Status: {pdf_metadata.get('status')}")
                print(f"Content Type: {pdf_metadata.get('content_type')}")
                print(f"Content Length: {len(pdf_content)} characters")
                print(f"Preview: {pdf_content[:200]}...")
                
                # Save PDF text to file
                pdf_output_path = output_dir / "arxiv_paper.txt"
                with open(pdf_output_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {pdf_metadata.get('title')}\n\n")
                    f.write(pdf_content)
                print(f"Saved PDF content to: {pdf_output_path}")
                
            except Exception as e:
                print(f"Error extracting PDF (this is expected if PDF extraction libraries are not installed): {str(e)}")
        else:
            print("\nSkipping PDF extraction example as PDF_EXTRACTOR_AVAILABLE is False")
        
        # Example 5: Parallel scraping of multiple URLs
        print("\n" + "-" * 80)
        print("EXAMPLE 5: PARALLEL SCRAPING")
        print("-" * 80)
        print("Scraping multiple URLs in parallel (excluding PDF for this example)...")
        
        parallel_urls = [v for k, v in test_urls.items() if k != "pdf_document"]
        print(f"URLs to scrape: {len(parallel_urls)}")
        
        # Use parallel scraping
        start_time = time.time()
        results = scraper.scrape_multiple(parallel_urls)
        duration = time.time() - start_time
        
        print(f"Completed parallel scraping in {duration:.2f} seconds")
        
        # Create a DataFrame to display results
        results_data = []
        for i, (content, metadata) in enumerate(results):
            url = metadata.get('url', parallel_urls[i])
            results_data.append({
                "url": url,
                "title": metadata.get('title', 'Unknown'),
                "status": metadata.get('status', 'Unknown'),
                "method": metadata.get('scraping_method', 'Unknown'),
                "content_length": len(content)
            })
        
        results_df = pd.DataFrame(results_data)
        print("\nParallel scraping results:")
        print(results_df.to_string(index=False))
        
        # Example 6: Integration with document_utils (if available)
        if document_utils_available:
            print("\n" + "-" * 80)
            print("EXAMPLE 6: INTEGRATION WITH DOCUMENT UTILS")
            print("-" * 80)
            
            # Select a content and metadata pair to process
            integration_content, integration_metadata = results[0]
            url = integration_metadata.get('url')
            print(f"Processing content from {url} with document_utils")
            
            # Extract additional metadata
            enhanced_metadata = extract_metadata(integration_content, url)
            print("\nEnhanced metadata:")
            for key, value in enhanced_metadata.items():
                print(f"  {key}: {value}")
            
            # Process document (would normally chunk and vectorize)
            processed_doc = process_document(
                text=integration_content,
                metadata=enhanced_metadata,
                chunk_size=1000,  # Example chunk size
                chunk_overlap=200,  # Example overlap
                vectorize=False  # Don't vectorize in this example
            )
            
            print(f"\nProcessed document into {len(processed_doc['chunks'])} chunks")
            print(f"First chunk preview: {processed_doc['chunks'][0][:100]}...")
            
            # Save processed document to file
            doc_output_path = output_dir / "processed_document.json"
            with open(doc_output_path, "w", encoding="utf-8") as f:
                json.dump(processed_doc, f, indent=2)
            print(f"Saved processed document to: {doc_output_path}")
        
        # Example 7: Error handling and retry demonstration
        print("\n" + "-" * 80)
        print("EXAMPLE 7: ERROR HANDLING & RETRY LOGIC")
        print("-" * 80)
        
        # Intentionally invalid URL to demonstrate error handling
        invalid_url = "https://this-domain-does-not-exist-123456789.com"
        print(f"Attempting to scrape invalid URL: {invalid_url}")
        
        try:
            invalid_content, invalid_metadata = scraper.scrape(invalid_url)
            print(f"Result: {invalid_metadata.get('status')}")
            print(f"Error: {invalid_metadata.get('error')}")
        except Exception as e:
            print(f"Caught exception: {str(e)}")
            print("This demonstrates the scraper's error handling capability")
        
        print("\n" + "=" * 80)
        print("SUMMARY OF CAPABILITIES DEMONSTRATED:")
        print("=" * 80)
        print("1. Wikipedia-specific scraping (optimized for Wikipedia's structure)")
        print("2. HTTP-based scraping (fast for simple websites)")
        print("3. Browser-based scraping with Playwright (for JavaScript-heavy sites)")
        if PDF_EXTRACTOR_AVAILABLE:
            print("4. PDF extraction capabilities")
        print("5. Parallel scraping of multiple URLs")
        if document_utils_available:
            print("6. Integration with document processing pipeline")
        print("7. Error handling and retry mechanisms")
        print("8. User-agent rotation for avoiding detection")
        print("9. Stealth mode browser configuration")
        print("\nAll output files saved to: " + str(output_dir))
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up resources
        print("\nCleaning up resources...")
        scraper.close()
        print("UnifiedScraper closed successfully")
