"""
Query Hash Utility for Agentic Researcher

This module provides utilities for hashing queries, detecting duplicates,
and managing query histories efficiently.
"""

import hashlib
import re
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

class QueryHashManager:
    """
    Query Hash Manager for efficient query deduplication and management.
    
    This class provides:
    1. Consistent query normalization
    2. Efficient hashing algorithms
    3. Query similarity detection
    4. History tracking
    
    Attributes:
        logger (logging.Logger): Logger for the query hash manager
        sqlite_manager: SQLite database manager for persistence
        stopwords (Set[str]): Set of common stopwords to ignore in normalization
    """
    
    def __init__(self, sqlite_manager=None):
        """Initialize the QueryHashManager.
        
        Args:
            sqlite_manager: SQLite database manager for persistence
        """
        self.logger = logging.getLogger("utils.query_hash")
        self.sqlite_manager = sqlite_manager
        
        # Common English stopwords
        self.stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "from", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "to", "of", "in", "on", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "can", "could", "will",
            "would", "shall", "should", "may", "might", "must", "this", "that",
            "these", "those", "what", "which", "who", "whom", "whose", "how"
        }
        
        self.logger.info("QueryHashManager initialized")
    
    def normalize_query(self, query: str, remove_stopwords: bool = True) -> str:
        """Normalize a query by removing punctuation, extra spaces, and optionally stopwords.
        
        Args:
            query (str): The query to normalize
            remove_stopwords (bool, optional): Whether to remove stopwords. Defaults to True.
            
        Returns:
            str: Normalized query
        """
        if not query:
            return ""
        
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters and replace with space
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Replace multiple spaces with single space
        query = re.sub(r'\s+', ' ', query)
        
        # Remove stopwords if requested
        if remove_stopwords:
            words = query.split()
            filtered_words = [word for word in words if word not in self.stopwords]
            query = " ".join(filtered_words)
        
        # Final trim
        query = query.strip()
        
        return query
    
    def compute_hash(self, query: str, normalize: bool = True) -> str:
        """Compute a hash for the query.
        
        Args:
            query (str): The query to hash
            normalize (bool, optional): Whether to normalize the query first. Defaults to True.
            
        Returns:
            str: Hash of the query
        """
        if not query:
            return ""
        
        # Normalize if requested
        if normalize:
            query = self.normalize_query(query)
            
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(query.encode("utf-8"))
        return hash_obj.hexdigest()
    
    def compute_semantic_hash(self, query: str) -> str:
        """Compute a semantic-oriented hash that ignores word order.
        
        This creates a hash based on the sorted, normalized words,
        allowing queries with the same words in different orders to
        be identified as similar.
        
        Args:
            query (str): The query to hash
            
        Returns:
            str: Semantic hash of the query
        """
        if not query:
            return ""
        
        # Normalize and get words
        normalized = self.normalize_query(query)
        words = normalized.split()
        
        # Sort words alphabetically
        words.sort()
        
        # Join and hash
        sorted_query = " ".join(words)
        hash_obj = hashlib.sha256(sorted_query.encode("utf-8"))
        
        return hash_obj.hexdigest()
    
    def compute_ngram_hash(self, query: str, n: int = 3) -> str:
        """Compute a hash based on character n-grams.
        
        This creates a hash based on the sorted n-grams in the query,
        which can help identify queries with similar content even if
        words are slightly different.
        
        Args:
            query (str): The query to hash
            n (int, optional): Size of n-grams. Defaults to 3.
            
        Returns:
            str: N-gram hash of the query
        """
        if not query:
            return ""
        
        # Normalize
        normalized = self.normalize_query(query, remove_stopwords=False)
        
        # Generate n-grams
        ngrams = []
        for i in range(len(normalized) - n + 1):
            ngrams.append(normalized[i:i+n])
        
        # Sort and join
        ngrams.sort()
        ngram_text = "".join(ngrams)
        
        # Hash
        hash_obj = hashlib.sha256(ngram_text.encode("utf-8"))
        
        return hash_obj.hexdigest()
    
    def store_query(self, query: str, project_id: int = None, metadata: Dict = None) -> Dict:
        """Store a query with its hashes in the database.
        
        Args:
            query (str): The query to store
            project_id (int, optional): Associated project ID. Defaults to None.
            metadata (Dict, optional): Additional metadata. Defaults to None.
            
        Returns:
            Dict: Stored query data
        """
        if not query or not self.sqlite_manager:
            self.logger.warning("Cannot store query: missing query or SQLite manager")
            return {}
        
        try:
            # Compute hashes
            exact_hash = self.compute_hash(query)
            semantic_hash = self.compute_semantic_hash(query)
            ngram_hash = self.compute_ngram_hash(query)
            
            # Prepare query data
            query_data = {
                "query": query,
                "project_id": project_id,
                "exact_hash": exact_hash,
                "semantic_hash": semantic_hash,
                "ngram_hash": ngram_hash,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Store in database
            query_id = self.sqlite_manager.store_query(query_data)
            
            self.logger.info(f"Stored query with ID {query_id}: {query[:30]}...")
            
            # Add ID to returned data
            query_data["id"] = query_id
            
            return query_data
            
        except Exception as e:
            self.logger.error(f"Error storing query: {str(e)}")
            return {}
    
    def find_similar_queries(self, query: str, limit: int = 5) -> List[Dict]:
        """Find similar queries based on various hash methods.
        
        Args:
            query (str): The query to find similar matches for
            limit (int, optional): Maximum number of results. Defaults to 5.
            
        Returns:
            List[Dict]: List of similar queries with metadata
        """
        if not query or not self.sqlite_manager:
            return []
        
        try:
            # Compute hashes
            exact_hash = self.compute_hash(query)
            semantic_hash = self.compute_semantic_hash(query)
            ngram_hash = self.compute_ngram_hash(query)
            
            # Find matching queries
            results = self.sqlite_manager.find_similar_queries(
                exact_hash=exact_hash,
                semantic_hash=semantic_hash,
                ngram_hash=ngram_hash,
                limit=limit
            )
            
            if results:
                self.logger.info(f"Found {len(results)} similar queries for: {query[:30]}...")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar queries: {str(e)}")
            return []
    
    def get_query_history(self, project_id: int = None, limit: int = 50) -> List[Dict]:
        """Get query history, optionally filtered by project.
        
        Args:
            project_id (int, optional): Project ID to filter by. Defaults to None.
            limit (int, optional): Maximum number of results. Defaults to 50.
            
        Returns:
            List[Dict]: List of historical queries
        """
        if not self.sqlite_manager:
            return []
        
        try:
            # Get queries from database
            history = self.sqlite_manager.get_query_history(project_id, limit)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting query history: {str(e)}")
            return []
    
    def get_query_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get statistics about queries over a time period.
        
        Args:
            days (int, optional): Number of days to analyze. Defaults to 7.
            
        Returns:
            Dict[str, Any]: Query statistics
        """
        if not self.sqlite_manager:
            return {}
        
        try:
            # Get basic stats from database
            stats = self.sqlite_manager.get_query_stats(days)
            
            # Add derived statistics
            if stats.get("total_queries", 0) > 0:
                stats["unique_ratio"] = stats.get("unique_queries", 0) / stats["total_queries"]
            else:
                stats["unique_ratio"] = 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting query stats: {str(e)}")
            return {}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create query hash manager
    manager = QueryHashManager()
    
    # Test queries
    q1 = "What is the formula for calculating VIX?"
    q2 = "How is the VIX index calculated?"
    q3 = "How do you calculate the Volatility Index?"
    q4 = "What is the formula for the Volatility Index calculation?"
    
    # Show hashes
    print(f"Query: {q1}")
    print(f"Exact hash: {manager.compute_hash(q1)}")
    print(f"Semantic hash: {manager.compute_semantic_hash(q1)}")
    print(f"N-gram hash: {manager.compute_ngram_hash(q1)}")
    
    print("\nTesting hash similarity:")
    # Compare semantic hashes
    print("\nSemantic hash comparison:")
    hash1 = manager.compute_semantic_hash(q1)
    hash2 = manager.compute_semantic_hash(q2)
    hash3 = manager.compute_semantic_hash(q3)
    hash4 = manager.compute_semantic_hash(q4)
    
    print(f"Q1 vs Q2: {hash1 == hash2}")
    print(f"Q1 vs Q3: {hash1 == hash3}")
    print(f"Q1 vs Q4: {hash1 == hash4}")
    
    # Compare n-gram hashes
    print("\nN-gram hash comparison:")
    hash1 = manager.compute_ngram_hash(q1)
    hash2 = manager.compute_ngram_hash(q2)
    hash3 = manager.compute_ngram_hash(q3)
    hash4 = manager.compute_ngram_hash(q4)
    
    print(f"Q1 vs Q2: {hash1 == hash2}")
    print(f"Q1 vs Q3: {hash1 == hash3}")
    print(f"Q1 vs Q4: {hash1 == hash4}")
