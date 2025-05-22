"""
Google Search Module for Agentic Researcher
Implements Google Custom Search Engine as a final fallback
"""
import sys
import os
import time
import requests
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.config import Config as config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class GoogleSearch:
    """
    Google search implementation
    Uses the Google Custom Search JSON API with retry logic
    """
    
    def __init__(self):
        # Use global config
        self.config = config
        
        # Get API key and CSE ID
        self.api_key = self.config.google_api_key
        self.cse_id = self.config.google_cse_id
        
        # API endpoint
        self.search_endpoint = "https://www.googleapis.com/customsearch/v1"
        
        # Rate limiting settings
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
        # Search settings
        self.max_retries = 3
        self.timeout = 10  # seconds
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform Google search with retry logic
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Search results with title, snippet, and url
        """
        # Check if API credentials are configured
        if not self.api_key or not self.cse_id:
            raise ValueError("Google API key or CSE ID not configured. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in config.toml.")
        
        # Enforce rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        try:
            # Prepare request parameters
            params = {
                "q": query,
                "key": self.api_key,
                "cx": self.cse_id,
                "num": min(10, max_results)  # Google API allows max 10 results per request
            }
            
            # Initialize results
            all_results = []
            
            # Make multiple requests if necessary (Google API returns max 10 per request)
            for start_index in range(1, max_results + 1, 10):
                if start_index > 1:
                    params["start"] = start_index
                
                # Send request
                response = requests.get(
                    self.search_endpoint,
                    params=params,
                    timeout=self.timeout
                )
                
                # Update last request time
                self.last_request_time = time.time()
                
                # Check response status
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                # Extract results
                items = data.get("items", [])
                
                for item in items:
                    all_results.append({
                        "title": item.get("title", ""),
                        "body": item.get("snippet", ""),
                        "url": item.get("link", "")
                    })
                
                # If we didn't get 10 results, or we have enough results, break
                if len(items) < 10 or len(all_results) >= max_results:
                    break
            
            # Trim to max_results
            return all_results[:max_results]
        
        except requests.exceptions.RequestException as e:
            # Log error
            print(f"Google search error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response text: {e.response.text}")
            raise
        
        except Exception as e:
            # Log other errors
            print(f"Google search error: {str(e)}")
            raise
    
    def image_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform Google image search
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Image search results
        """
        # Check if API credentials are configured
        if not self.api_key or not self.cse_id:
            raise ValueError("Google API key or CSE ID not configured. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in config.toml.")
        
        # Enforce rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        try:
            # Prepare request parameters
            params = {
                "q": query,
                "key": self.api_key,
                "cx": self.cse_id,
                "num": min(10, max_results),
                "searchType": "image"
            }
            
            # Initialize results
            all_results = []
            
            # Make multiple requests if necessary (Google API returns max 10 per request)
            for start_index in range(1, max_results + 1, 10):
                if start_index > 1:
                    params["start"] = start_index
                
                # Send request
                response = requests.get(
                    self.search_endpoint,
                    params=params,
                    timeout=self.timeout
                )
                
                # Update last request time
                self.last_request_time = time.time()
                
                # Check response status
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                # Extract results
                items = data.get("items", [])
                
                for item in items:
                    all_results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "image": item.get("link", ""),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                        "context": item.get("image", {}).get("contextLink", ""),
                        "height": item.get("image", {}).get("height", 0),
                        "width": item.get("image", {}).get("width", 0)
                    })
                
                # If we didn't get 10 results, or we have enough results, break
                if len(items) < 10 or len(all_results) >= max_results:
                    break
            
            # Trim to max_results
            return all_results[:max_results]
        
        except Exception as e:
            # Log error
            print(f"Google image search error: {str(e)}")
            return []


# Example usage
if __name__ == "__main__":
    # Add import path handling for direct execution
    import sys
    import os
    # Add the parent directory to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    try:
        # Initialize the Google search engine
        google_search = GoogleSearch()
        
        # Example 1: Basic web search
        print("\n=== Basic Web Search Example ===")
        query = "artificial intelligence research papers"
        results = google_search.search(query, max_results=3)
        
        print(f"Search results for: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['body']}")
        
        # Example 2: Image search
        print("\n\n=== Image Search Example ===")
        image_query = "machine learning diagrams"
        image_results = google_search.image_search(image_query, max_results=2)
        
        print(f"Image search results for: '{image_query}'")
        for i, result in enumerate(image_results, 1):
            print(f"\nImage {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Thumbnail: {result['thumbnail']}")
            print(f"Dimensions: {result['width']}x{result['height']}")
    
    except ValueError as e:
        print(f"Configuration Error: {str(e)}")
        print("Please ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are set in your config.toml file.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
