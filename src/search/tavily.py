"""
Tavily Search Module for Agentic Researcher
Implements Tavily search functionality for retrieving web links using LangChain integration
"""
import time
import logging
import os
import sys
from typing import List, Dict, Any

# Add project root to Python path for direct script execution
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    sys.path.insert(0, project_root)

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from src.utils.config import config

# Set up logger
logger = logging.getLogger(__name__)

class TavilySearch:
    """
    Tavily search implementation using LangChain's integration
    Focused on retrieving high-quality web links for research
    """
    
    def __init__(self):
        self.config = config
        
        self.api_key = self.config.tavily_api_key
        
        self.request_delay = 1.0
        self.last_request_time = 0
        
        try:
            self.tavily_client = TavilySearchAPIWrapper(tavily_api_key=self.api_key)
            
            self.max_results = 10
            self.search_depth = "basic"
            
            logger.info("Tavily search client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily client: {e}")
            self.tavily_client = None
    
    def search(self, query: str, max_results: int = 10, search_depth: str = "basic") -> List[Dict[str, Any]]:
        """
        Perform Tavily search using LangChain's integration
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_depth: Search depth (basic or advanced)
            
        Returns:
            List[Dict]: Search results with title, content, and url
        """
        if not self.tavily_client:
            logger.error("Tavily client not initialized")
            return []
            
        if not self.api_key:
            logger.error("Tavily API key not configured")
            return []
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        try:
            self.max_results = max_results
            self.search_depth = search_depth
            
            logger.info(f"Executing Tavily search for query: {query}")
            raw_results = self.tavily_client.results(
                query, 
                max_results=self.max_results,
                search_depth=self.search_depth
            )
            
            self.last_request_time = time.time()
            
            results = []
            for result in raw_results:
                results.append({
                    "title": result.get("title", ""),
                    "body": result.get("content", ""),
                    "url": result.get("url", "")
                })
            
            logger.info(f"Tavily search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Tavily search error: {str(e)}")
            return []
    
    def search_with_content(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform advanced Tavily search with detailed content
        Uses the advanced search depth to get more comprehensive results
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dict: Search results with included detailed content
        """
        if not self.tavily_client:
            logger.error("Tavily client not initialized")
            return {"error": "Tavily client not initialized", "results": []}
            
        if not self.api_key:
            logger.error("Tavily API key not configured")
            return {"error": "Tavily API key not configured", "results": []}
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        try:
            logger.info(f"Executing Tavily advanced search for query: {query}")
            raw_results = self.tavily_client.results(
                query,
                max_results=max_results, 
                search_depth="advanced"
            )
            
            self.last_request_time = time.time()
            
            formatted_results = {
                "results": [
                    {
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                        "url": result.get("url", "")
                    } for result in raw_results
                ]
            }
            
            logger.info("Tavily advanced search completed successfully")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Tavily advanced search error: {str(e)}")
            return {"error": str(e), "results": []}


# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    sys.path.insert(0, project_root)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize search
    tavily = TavilySearch()
    
    # Test basic search
    query = "What are the latest developments in artificial intelligence?"
    results = tavily.search(query, max_results=3)
    print(f"\nBasic Search Results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print("---")
        
    # Test advanced search with content
    advanced_query = "quantum computing applications"
    detailed_results = tavily.search_with_content(advanced_query, max_results=2)
    print(f"\nAdvanced Search Results for: '{advanced_query}'")
    
    for i, result in enumerate(detailed_results.get("results", []), 1):
        print(f"Detailed Result {i}:")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        # Print a snippet of the content
        content_snippet = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
        print(f"Content snippet: {content_snippet}")
        print("---")