
import logging
from typing import Dict, List, Any, Optional
import time
import random
import requests
from duckduckgo_search import DDGS

from src.config.system_config import SystemConfig

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """
    Agent responsible for web searches using various search engines.
    Currently implements DuckDuckGo search.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def search(self, query: str, depth: int = 3, include_academic: bool = True) -> List[Dict[str, Any]]:
        """
        Perform a web search for the given query.
        
        Args:
            query: The search query
            depth: Search depth (1-5), affects number of results
            include_academic: Whether to include academic sources
            
        Returns:
            List of search results
        """
        logger.info(f"Searching for: {query}")
        
        # Calculate number of results based on depth
        num_results = 5 * depth
        
        # Combined results from different sources
        combined_results = []
        
        # Web search using DuckDuckGo
        web_results = self._search_duckduckgo(query, num_results)
        combined_results.extend(web_results)
        
        # Academic search if requested
        if include_academic:
            academic_results = self._search_academic(query, num_results // 2)
            combined_results.extend(academic_results)
        
        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(combined_results)
        
        logger.info(f"Found {len(unique_results)} unique results")
        return unique_results[:num_results]  # Ensure we don't return too many
    
    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Perform a search using DuckDuckGo"""
        try:
            results = []
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=num_results))
                
                for i, r in enumerate(ddg_results):
                    results.append({
                        "title": r.get("title", "No Title"),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "source": "duckduckgo",
                        "rank": i + 1,
                        "relevance_score": self._calculate_relevance(query, r.get("title", ""), r.get("body", ""))
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {str(e)}")
            return []
    
    def _search_academic(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Search for academic papers.
        In a production system, this would use APIs like arXiv, Semantic Scholar, etc.
        This is a simplified placeholder implementation.
        """
        # In a real implementation, this would use academic APIs
        # For now, just return a simple placeholder
        return [
            {
                "title": f"Academic paper about {query}",
                "url": f"https://example.org/papers/{i}",
                "snippet": f"This academic paper discusses various aspects of {query}...",
                "source": "academic",
                "rank": i + 1,
                "relevance_score": random.uniform(0.7, 0.95)
            }
            for i in range(min(3, num_results))  # Return fewer academic results as a placeholder
        ]
    
    def _calculate_relevance(self, query: str, title: str, snippet: str) -> float:
        """
        Calculate a simple relevance score.
        In a real application, this would use more sophisticated algorithms.
        """
        # Simple implementation - count query terms in title and snippet
        query_terms = set(query.lower().split())
        title_terms = set(title.lower().split())
        snippet_terms = set(snippet.lower().split())
        
        title_matches = len(query_terms.intersection(title_terms))
        snippet_matches = len(query_terms.intersection(snippet_terms))
        
        # Weight title matches more heavily
        score = (title_matches * 2 + snippet_matches) / (len(query_terms) * 3)
        
        # Add some randomness to simulate more complex ranking
        score = min(1.0, score + random.uniform(0, 0.3))
        
        return score
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return unique_results
