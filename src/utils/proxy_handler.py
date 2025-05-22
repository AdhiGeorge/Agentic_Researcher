"""
Proxy Handler utility for Agentic Researcher
Manages rotating user-agents and headers for web scraping
"""
import random
import time
from typing import Dict, Optional
from fake_useragent import UserAgent

# Mock config for standalone example
class Config:
    def __init__(self):
        self.scraping_user_agent_rotation = True
        self.scraping_retry_attempts = 3
        self.scraping_retry_delay = 2

# Create config instance for standalone use
config = Config()

class ProxyHandler:
    """
    Proxy handler for web scraping
    Handles rotating user agents and request headers
    """
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProxyHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Use global config
        self.config = config
        
        # Initialize user agent generator
        self.user_agent_rotation = self.config.scraping_user_agent_rotation
        self.ua_generator = UserAgent()
        
        # Default headers
        self.default_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
        
        # Common desktop browsers for diversity
        self.browser_types = [
            "chrome", "firefox", "edge", "safari"
        ]
        
        # Last used timestamp to prevent too frequent rotation
        self.last_rotation_time = 0
        self.min_rotation_interval = 5  # seconds
        
        self._initialized = True
    
    def get_random_user_agent(self, browser_type: Optional[str] = None) -> str:
        """
        Get a random user agent string
        
        Args:
            browser_type: Optional browser type to generate agent for
            
        Returns:
            str: User agent string
        """
        if browser_type and browser_type in self.browser_types:
            return self.ua_generator[browser_type]
        else:
            # Random browser type
            browser = random.choice(self.browser_types)
            return self.ua_generator[browser]
    
    def get_headers(self, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get headers for HTTP requests with optional user agent rotation
        
        Args:
            custom_headers: Optional custom headers to include
            
        Returns:
            Dict[str, str]: Headers dict for requests
        """
        headers = self.default_headers.copy()
        
        # Rotate user agent if enabled
        if self.user_agent_rotation:
            current_time = time.time()
            if current_time - self.last_rotation_time >= self.min_rotation_interval:
                headers["User-Agent"] = self.get_random_user_agent()
                self.last_rotation_time = current_time
        
        # Add custom headers if provided
        if custom_headers:
            headers.update(custom_headers)
            
        return headers
    
    def get_retry_config(self) -> Dict[str, any]:
        """
        Get retry configuration for requests
        
        Returns:
            Dict: Retry configuration
        """
        return {
            "attempts": self.config.scraping_retry_attempts,
            "delay": self.config.scraping_retry_delay
        }
        
    def get_user_agents(self) -> list:
        """
        Get a list of diverse user agent strings
        
        Returns:
            list: List of user agent strings
        """
        user_agents = []
        for browser in self.browser_types:
            try:
                user_agents.append(self.ua_generator[browser])
            except:
                continue
        
        # Add some fallback user agents in case UserAgent fails
        fallback_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
        ]
        
        user_agents.extend(fallback_agents)
        return list(set(user_agents))  # Remove duplicates
        
    def get_proxy(self) -> Dict[str, str]:
        """
        Get proxy configuration (Currently returns None as proxies are not implemented)
        In a real implementation, this would return actual proxy servers
        
        Returns:
            Dict or None: Proxy configuration if available
        """
        # This is a placeholder for actual proxy implementation
        # In a real scenario, you would implement proxy rotation logic here
        return None


# Comprehensive example usage with real HTTP requests and project integration
if __name__ == "__main__":
    import os
    import sys
    import requests
    import json
    from pprint import pprint
    import time
    import tempfile
    from pathlib import Path
    import logging
    from urllib.parse import urlparse
    import random
    from concurrent.futures import ThreadPoolExecutor
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("proxy_handler_example")
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp(prefix="proxy_handler_example_")
    logger.info(f"Created temporary directory for outputs: {temp_dir}")
    
    print("\n" + "=" * 80)
    print("ProxyHandler: Advanced Example Usage & Integration")
    print("=" * 80)
    print("This example demonstrates how the ProxyHandler works with")
    print("web scraping workflows and integrates with other components.")
    
    # Initialize ProxyHandler
    proxy_handler = ProxyHandler()
    
    # Example 1: Basic usage - get headers and make a request
    print("\n" + "-" * 60)
    print("Example 1: Basic request with rotating user agents")
    print("-" * 60)
    
    try:
        # Get headers with rotating user agent
        headers = proxy_handler.get_headers()
        print("Generated headers:")
        pprint(headers)
        
        # Make a real request to a public API
        url = "https://httpbin.org/user-agent"
        print(f"\nMaking request to {url}")
        response = requests.get(url, headers=headers)
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Server detected user agent:")
            print(data.get('user-agent'))
    except Exception as e:
        print(f"Error in basic request: {e}")
    
    # Save results to a file in temp directory
    with open(os.path.join(temp_dir, "user_agent_test.json"), "w") as f:
        json.dump({"headers": headers, "response": data if response.status_code == 200 else None}, f, indent=2)
    
    # Example 2: Using multiple user agents
    print("\n" + "-" * 60)
    print("Example 2: Multiple requests with different user agents")
    print("-" * 60)
    try:
        # Get multiple user agents
        user_agents = proxy_handler.get_user_agents()
        print(f"Generated {len(user_agents)} different user agents")
        print("Sample user agents:")
        for i, ua in enumerate(user_agents[:3]):
            print(f"{i+1}. {ua[:80]}...")
            
        # Make multiple requests with different user agents
        print("\nMaking 3 requests with different user agents:")
        results = []
        for i in range(3):
            headers = proxy_handler.get_headers()
            print(f"\nRequest {i+1} with User-Agent: {headers.get('User-Agent', 'None')[:50]}...")
            
            response = requests.get("https://httpbin.org/headers", headers=headers)
            if response.status_code == 200:
                parsed = response.json()
                print(f"Server received headers:")
                received_ua = parsed.get('headers', {}).get('User-Agent', 'None')
                print(f"User-Agent: {received_ua[:50]}...")
                results.append({"user_agent": headers.get('User-Agent'), "response": parsed})
            
            # Add delay to simulate rate limiting avoidance
            if i < 2:  # No need to sleep after the last request
                time.sleep(2)
                
        # Save results to a file
        with open(os.path.join(temp_dir, "multiple_user_agents.json"), "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error in multiple requests: {e}")
    
    # Example 3: Testing retry configuration
    print("\n" + "-" * 60)
    print("Example 3: Retry configuration and error handling")
    print("-" * 60)
    try:
        retry_config = proxy_handler.get_retry_config()
        print("Retry configuration:")
        pprint(retry_config)
        
        # Define a function for making requests with retries
        def make_request_with_retry(url, max_attempts, delay, timeout=5):
            attempts = 0
            while attempts < max_attempts:
                try:
                    headers = proxy_handler.get_headers()
                    print(f"Attempt {attempts+1} with User-Agent: {headers.get('User-Agent', 'None')[:50]}...")
                    response = requests.get(url, headers=headers, timeout=timeout)
                    
                    if response.status_code < 400:  # Any successful status code
                        return response
                    else:
                        print(f"Received error status code: {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    print(f"Timeout error on attempt {attempts+1}")
                except requests.exceptions.ConnectionError:
                    print(f"Connection error on attempt {attempts+1}")
                except Exception as e:
                    print(f"Error on attempt {attempts+1}: {e}")
                    
                attempts += 1
                if attempts < max_attempts:
                    # Exponential backoff with jitter
                    jitter = random.uniform(0, 0.5)
                    sleep_time = delay * (2 ** (attempts - 1)) + jitter
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
            return None
        
        # First test with a URL that always succeeds
        print("\nTest 1: Request to a reliable endpoint:")
        result = make_request_with_retry(
            "https://httpbin.org/status/200",  # This URL always returns 200 OK
            retry_config['attempts'],
            retry_config['delay']
        )
        
        if result:
            print(f"Success! Status code: {result.status_code}")
        else:
            print("Failed after all retry attempts")
        
        # Test with a URL that returns errors to demonstrate retries
        print("\nTest 2: Request to an endpoint that randomly returns errors:")
        # This endpoint randomly returns 200, 429, or 503
        test_url = "https://httpbin.org/status/200,429,503"
        result = make_request_with_retry(
            test_url,
            retry_config['attempts'],
            retry_config['delay']
        )
        
        if result:
            print(f"Success after retries! Status code: {result.status_code}")
        else:
            print("Failed after all retry attempts")
            
    except Exception as e:
        print(f"Error in retry example: {e}")
    
    # Example 4: Integration with unified scraper
    print("\n" + "-" * 60)
    print("Example 4: Integration with Unified Scraper")
    print("-" * 60)
    
    # Simulate integration with a unified scraper
    class MockUnifiedScraper:
        def __init__(self, proxy_handler):
            self.proxy_handler = proxy_handler
            self.logger = logging.getLogger("unified_scraper")
            
        def scrape(self, url, method="requests"):
            """Simulate scraping a URL using different methods"""
            print(f"Scraping URL: {url} using method: {method}")
            
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Get appropriate headers for the domain
            custom_headers = {}
            if "wikipedia" in domain:
                custom_headers["Accept"] = "text/html,application/xhtml+xml"
            
            headers = self.proxy_handler.get_headers(custom_headers)
            
            # Different scraping methods
            if method == "requests":
                return self._scrape_with_requests(url, headers)
            elif method == "playwright":
                return self._simulate_playwright_scrape(url, headers)
            elif method == "pdf":
                return self._simulate_pdf_extraction(url, headers)
            else:
                raise ValueError(f"Unsupported scraping method: {method}")
        
        def _scrape_with_requests(self, url, headers):
            """Simulate scraping with requests"""
            try:
                # Simulate actual web request
                retry_config = self.proxy_handler.get_retry_config()
                response = None
                
                for attempt in range(retry_config['attempts']):
                    try:
                        print(f"HTTP request attempt {attempt+1} with headers:\n{json.dumps(headers, indent=2)[:200]}...")
                        response = requests.get(url, headers=headers, timeout=10)
                        if response.status_code < 400:
                            break
                    except Exception as e:
                        print(f"Error on attempt {attempt+1}: {e}")
                        if attempt < retry_config['attempts'] - 1:
                            time.sleep(retry_config['delay'])
                
                if response and response.status_code < 400:
                    content_type = response.headers.get('Content-Type', '')
                    content_length = len(response.content)
                    
                    print(f"Success! Received {content_length} bytes of {content_type}")
                    return {
                        "success": True,
                        "content_type": content_type,
                        "content_length": content_length,
                        "method": "requests"
                    }
                else:
                    status = response.status_code if response else "No response"
                    print(f"Failed with status: {status}")
                    return {"success": False, "error": f"HTTP error: {status}"}
            except Exception as e:
                print(f"Scraping error: {e}")
                return {"success": False, "error": str(e)}
        
        def _simulate_playwright_scrape(self, url, headers):
            """Simulate scraping with Playwright (browser automation)"""
            print("Simulating Playwright browser session...")
            print(f"Setting headers for browser session:\n{json.dumps(headers, indent=2)[:200]}...")
            time.sleep(1)  # Simulate browser startup time
            
            # Simulate successful scrape
            print("Browser navigated to URL")
            print("Waiting for page to fully load...")
            time.sleep(0.5)  # Simulate page load
            
            # Simulate content extraction
            print("Extracting page content...")
            time.sleep(0.5)  # Simulate extraction time
            
            return {
                "success": True,
                "content_type": "text/html",
                "content_length": random.randint(50000, 200000),  # Simulate HTML size
                "method": "playwright",
                "browser_data": {
                    "user_agent": headers.get("User-Agent"),
                    "cookies": {"session_id": "mock_session", "cf_clearance": "mock_cf_token"}
                }
            }
        
        def _simulate_pdf_extraction(self, url, headers):
            """Simulate PDF extraction"""
            print("Simulating PDF download and extraction...")
            
            # Simulate PDF download
            print(f"Downloading PDF from {url}")
            time.sleep(1)  # Simulate download time
            
            # Simulate content extraction from PDF
            print("Extracting text from PDF...")
            time.sleep(0.5)  # Simulate extraction time
            
            return {
                "success": True,
                "content_type": "application/pdf",
                "content_length": random.randint(100000, 5000000),  # Simulate PDF size
                "method": "pdf",
                "text_content_length": random.randint(5000, 50000),  # Simulated extracted text
                "pages": random.randint(1, 30)  # Simulated page count
            }
    
    # Instantiate the mock scraper with our proxy handler
    mock_scraper = MockUnifiedScraper(proxy_handler)
    
    # Test websites to scrape
    test_urls = [
        ("https://httpbin.org/get", "requests"),
        ("https://en.wikipedia.org/wiki/Web_scraping", "playwright"),
        ("https://www.example.com/sample.pdf", "pdf")
    ]
    
    # Scrape each URL with the appropriate method
    results = []
    for url, method in test_urls:
        print(f"\nScraping {url} with {method} method:")
        result = mock_scraper.scrape(url, method)
        results.append({"url": url, "method": method, "result": result})
        print(f"Result: {'Success' if result['success'] else 'Failure'}")
        print("Details:")
        pprint(result)
    
    # Save the results to a file
    with open(os.path.join(temp_dir, "scraping_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Example 5: Parallel scraping with proxy rotation
    print("\n" + "-" * 60)
    print("Example 5: Parallel scraping with proxy rotation")
    print("-" * 60)
    
    # Simulate parallel scraping with multiple threads
    def parallel_scrape(urls, max_workers=3):
        results = []
        
        def scrape_single_url(url):
            # Get fresh headers for each request
            headers = proxy_handler.get_headers()
            try:
                print(f"Scraping {url} with User-Agent: {headers.get('User-Agent', 'None')[:30]}...")
                response = requests.get(url, headers=headers, timeout=10)
                return {
                    "url": url,
                    "success": response.status_code < 400,
                    "status_code": response.status_code,
                    "content_type": response.headers.get('Content-Type'),
                    "user_agent": headers.get('User-Agent')
                }
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                return {"url": url, "success": False, "error": str(e)}
        
        print(f"Starting parallel scraping of {len(urls)} URLs with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(scrape_single_url, url): url for url in urls}
            for future in future_to_url:
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Finished {result['url']}: {'Success' if result.get('success') else 'Failed'}")
                except Exception as e:
                    url = future_to_url[future]
                    print(f"Worker exception processing {url}: {e}")
                    results.append({"url": url, "success": False, "error": str(e)})
        
        return results
    
    # Test parallel scraping with multiple URLs
    parallel_urls = [
        "https://httpbin.org/delay/1",  # 1 second delay
        "https://httpbin.org/status/200",
        "https://httpbin.org/headers",
        "https://httpbin.org/user-agent",
        "https://httpbin.org/ip"
    ]
    
    print("Starting parallel scraping test")
    parallel_results = parallel_scrape(parallel_urls, max_workers=3)
    
    print("\nParallel scraping results:")
    success_count = sum(1 for r in parallel_results if r.get('success'))
    print(f"Successfully scraped {success_count} out of {len(parallel_urls)} URLs")
    
    # Save the results to a file
    with open(os.path.join(temp_dir, "parallel_scraping_results.json"), "w") as f:
        json.dump(parallel_results, f, indent=2)
    
    # Example 6: Integration with DuckDuckGo search
    print("\n" + "-" * 60)
    print("Example 6: Integration with DuckDuckGo search client")
    print("-" * 60)
    
    # Simulate a DuckDuckGo search client using the proxy handler
    class MockDDGSearchClient:
        def __init__(self, proxy_handler):
            self.proxy_handler = proxy_handler
            self.base_url = "https://duckduckgo.com/"
            self.search_url = "https://duckduckgo.com/html/"
            self.logger = logging.getLogger("ddg_search")
        
        def search(self, query, max_results=5):
            """Simulate a search query to DuckDuckGo"""
            print(f"Searching DuckDuckGo for: '{query}'")
            
            # Prepare special headers for DuckDuckGo
            custom_headers = {
                "Referer": "https://duckduckgo.com/",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
            
            headers = self.proxy_handler.get_headers(custom_headers)
            print("Using headers:")
            pprint({k: v for k, v in headers.items() if k in ['User-Agent', 'Referer', 'Accept']})
            
            # Simulate search request
            print("Sending search request...")
            time.sleep(1)  # Simulate network delay
            
            # Simulate search results
            results = []
            for i in range(max_results):
                results.append({
                    "title": f"Result {i+1} for {query}",
                    "url": f"https://example.com/result{i+1}",
                    "description": f"This is a simulated search result {i+1} for '{query}'",
                    "position": i+1
                })
            
            print(f"Retrieved {len(results)} search results")
            return results
    
    # Instantiate the mock DDG search client
    ddg_client = MockDDGSearchClient(proxy_handler)
    
    # Perform some sample searches
    search_queries = [
        "web scraping techniques",
        "proxy rotation for web scraping",
        "avoiding detection when scraping"
    ]
    
    search_results = {}
    for query in search_queries:
        print(f"\nPerforming search for: '{query}'")
        results = ddg_client.search(query, max_results=3)
        search_results[query] = results
        
        print(f"Top results for '{query}':")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} - {result['url']}")
    
    # Save search results to a file
    with open(os.path.join(temp_dir, "search_results.json"), "w") as f:
        json.dump(search_results, f, indent=2)
    
    # Summary of all examples
    print("\n" + "=" * 80)
    print("ProxyHandler Example Summary")
    print("=" * 80)
    print(f"All example outputs saved to: {temp_dir}")
    print("\nExamples demonstrated:")
    print("1. Basic request with rotating user agents")
    print("2. Multiple requests with different user agents")
    print("3. Retry configuration and error handling")
    print("4. Integration with Unified Scraper")
    print("5. Parallel scraping with proxy rotation")
    print("6. Integration with DuckDuckGo search client")
    print("\nThese examples show how the ProxyHandler can be integrated")
    print("with various components of the Agentic Researcher project")
    print("to enable robust web scraping capabilities.")
    print("=" * 80)
