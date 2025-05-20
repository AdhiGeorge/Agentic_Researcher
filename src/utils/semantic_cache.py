"""
Semantic Cache for Agentic Researcher

This module implements a semantic cache that combines hash-based exact matching
with vector similarity for efficient retrieval of previously processed queries.
"""

import logging
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta

class SemanticCache:
    """
    Semantic Cache that combines hash-based and vector similarity approaches.
    
    This class provides caching functionality with:
    1. Hash-based exact matching for identical queries
    2. Semantic similarity matching for similar queries
    3. Adaptive similarity thresholds
    4. Temporal relevance for cache invalidation
    
    Attributes:
        logger (logging.Logger): Logger for the cache
        sqlite_manager: SQLite database manager for persistence
        embedding_function: Function to generate embeddings
        default_threshold (float): Default similarity threshold
        min_threshold (float): Minimum similarity threshold
        max_threshold (float): Maximum similarity threshold
        cache_ttl (int): Time-to-live for cache entries in seconds
        stats (Dict): Cache statistics
    """
    
    def __init__(
        self, 
        sqlite_manager=None,
        embedding_function=None,
        default_threshold: float = 0.90,
        min_threshold: float = 0.85,
        max_threshold: float = 0.95,
        cache_ttl: int = 7 * 24 * 3600  # 1 week default TTL
    ):
        """Initialize the SemanticCache.
        
        Args:
            sqlite_manager: SQLite database manager for persistence
            embedding_function: Function to generate embeddings
            default_threshold (float, optional): Default similarity threshold. Defaults to 0.90.
            min_threshold (float, optional): Minimum similarity threshold. Defaults to 0.85.
            max_threshold (float, optional): Maximum similarity threshold. Defaults to 0.95.
            cache_ttl (int, optional): Time-to-live for cache entries in seconds. Defaults to 1 week.
        """
        self.logger = logging.getLogger("utils.semantic_cache")
        self.sqlite_manager = sqlite_manager
        self.embedding_function = embedding_function
        
        # Threshold settings
        self.default_threshold = default_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # TTL setting
        self.cache_ttl = cache_ttl
        
        # Stats tracking
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "invalidations": 0
        }
        
        self.logger.info(f"SemanticCache initialized with threshold range [{min_threshold}-{max_threshold}]")
    
    def compute_query_hash(self, query: str) -> str:
        """Compute a hash for the query string.
        
        Args:
            query (str): The query string
            
        Returns:
            str: Hash of the query
        """
        # Normalize the query by lower-casing and removing extra whitespace
        normalized_query = " ".join(query.lower().split())
        
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(normalized_query.encode("utf-8"))
        return hash_obj.hexdigest()
    
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for the text using the provided embedding function.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        if not self.embedding_function:
            self.logger.warning("No embedding function provided, using dummy embedding")
            # Return a dummy embedding if no function provided (for testing)
            return [0.0] * 768
        
        try:
            # Normalize text
            normalized_text = " ".join(text.lower().split())
            
            # Generate embedding
            embedding = self.embedding_function(normalized_text)
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error computing embedding: {str(e)}")
            # Return dummy embedding on error
            return [0.0] * 768
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding
            embedding2 (List[float]): Second embedding
            
        Returns:
            float: Cosine similarity (0-1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _adaptive_threshold(self, query: str) -> float:
        """Calculate adaptive threshold based on query complexity.
        
        Args:
            query (str): The query string
            
        Returns:
            float: Adjusted threshold
        """
        # Simple heuristic: longer queries may need lower thresholds
        query_length = len(query.split())
        
        if query_length > 20:
            # More complex queries -> lower threshold
            threshold = self.default_threshold - 0.02
        elif query_length < 5:
            # Simple queries -> higher threshold
            threshold = self.default_threshold + 0.02
        else:
            # Default threshold
            threshold = self.default_threshold
        
        # Ensure threshold stays within bounds
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))
        
        return threshold
    
    def get(self, query: str, threshold: float = None) -> Optional[Dict[str, Any]]:
        """Get result from cache using hash-based or semantic similarity.
        
        Args:
            query (str): The query to look up
            threshold (float, optional): Similarity threshold. If None, uses adaptive threshold.
            
        Returns:
            Optional[Dict[str, Any]]: Cached result or None if not found
        """
        if not query:
            return None
        
        # Compute query hash
        query_hash = self.compute_query_hash(query)
        
        # Try exact match with hash first
        if self.sqlite_manager:
            exact_match = self.sqlite_manager.get_cached_query_by_hash(query_hash)
            if exact_match:
                # Check if cache entry is still valid
                created_at = exact_match.get("created_at")
                if created_at:
                    creation_time = datetime.fromisoformat(created_at)
                    if datetime.now() - creation_time > timedelta(seconds=self.cache_ttl):
                        # Cache entry expired
                        self.logger.debug(f"Cache entry expired for hash {query_hash}")
                        self.stats["invalidations"] += 1
                        return None
                
                self.logger.info(f"Exact cache hit for query: {query[:30]}...")
                self.stats["exact_hits"] += 1
                return exact_match
        
        # No exact match, try semantic similarity
        if self.embedding_function and self.sqlite_manager:
            # Use adaptive threshold if not provided
            if threshold is None:
                threshold = self._adaptive_threshold(query)
            
            # Compute query embedding
            query_embedding = self.compute_embedding(query)
            
            # Get recent queries from database
            recent_queries = self.sqlite_manager.get_recent_queries(limit=50)
            
            # Find semantically similar queries
            best_match = None
            best_similarity = 0.0
            
            for cached_query in recent_queries:
                # Skip if no embedding
                if not cached_query.get("embedding"):
                    continue
                
                # Compute similarity
                similarity = self.compute_similarity(
                    query_embedding,
                    cached_query["embedding"]
                )
                
                # Update best match if better
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_query
            
            # Return best match if found
            if best_match:
                self.logger.info(
                    f"Semantic cache hit for query: {query[:30]}... "
                    f"(similarity: {best_similarity:.4f})"
                )
                self.stats["semantic_hits"] += 1
                
                # Add similarity info to result
                result = best_match.copy()
                result["similarity"] = best_similarity
                result["original_query"] = best_match.get("query", "")
                
                return result
        
        # No match found
        self.logger.debug(f"Cache miss for query: {query[:30]}...")
        self.stats["misses"] += 1
        return None
    
    def put(self, query: str, result: Dict[str, Any]) -> None:
        """Store a query result in the cache.
        
        Args:
            query (str): The query string
            result (Dict[str, Any]): The result to cache
        """
        if not query or not result or not self.sqlite_manager:
            return
        
        try:
            # Compute query hash
            query_hash = self.compute_query_hash(query)
            
            # Compute embedding if function available
            embedding = None
            if self.embedding_function:
                embedding = self.compute_embedding(query)
            
            # Prepare cache entry
            cache_entry = {
                "query": query,
                "query_hash": query_hash,
                "embedding": embedding,
                "result": result,
                "created_at": datetime.now().isoformat()
            }
            
            # Store in database
            self.sqlite_manager.cache_query_result(cache_entry)
            
            self.logger.debug(f"Cached result for query: {query[:30]}...")
            
        except Exception as e:
            self.logger.error(f"Error caching query result: {str(e)}")
    
    def invalidate(self, query_hash: str = None, older_than: int = None) -> int:
        """Invalidate cache entries.
        
        Args:
            query_hash (str, optional): Specific query hash to invalidate. Defaults to None.
            older_than (int, optional): Invalidate entries older than this many seconds. Defaults to None.
            
        Returns:
            int: Number of invalidated entries
        """
        if not self.sqlite_manager:
            return 0
        
        try:
            # Convert older_than to timestamp if provided
            timestamp = None
            if older_than:
                timestamp = (datetime.now() - timedelta(seconds=older_than)).isoformat()
            
            # Invalidate entries
            count = self.sqlite_manager.invalidate_cached_queries(query_hash, timestamp)
            
            self.logger.info(f"Invalidated {count} cache entries")
            self.stats["invalidations"] += count
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error invalidating cache entries: {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        # Calculate hit ratio
        total_requests = self.stats["exact_hits"] + self.stats["semantic_hits"] + self.stats["misses"]
        hit_ratio = 0.0
        if total_requests > 0:
            hit_ratio = (self.stats["exact_hits"] + self.stats["semantic_hits"]) / total_requests
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_ratio": hit_ratio,
            "threshold_range": [self.min_threshold, self.max_threshold],
            "default_threshold": self.default_threshold
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simple embedding function for testing
    def dummy_embedding(text):
        # This is just for testing - returns a random vector
        import random
        return [random.random() for _ in range(768)]
    
    # Create cache
    cache = SemanticCache(embedding_function=dummy_embedding)
    
    # Test queries
    q1 = "What is the VIX index and how is it calculated?"
    q2 = "How do you calculate the Volatility Index (VIX)?"
    q3 = "What's the formula for computing VIX?"
    
    # Test exact match
    print(f"Query: {q1}")
    result = cache.get(q1)
    print(f"Cache get result: {result}")
    
    # Store result
    cache.put(q1, {"answer": "VIX is calculated using option prices..."})
    
    # Try exact match again
    result = cache.get(q1)
    print(f"Cache hit after storing: {result is not None}")
    
    # Try semantic match
    result = cache.get(q2)
    if result:
        print(f"Semantic hit with similarity: {result.get('similarity', 0)}")
        print(f"Original query: {result.get('original_query')}")
    else:
        print("Semantic miss")
    
    # Print stats
    print("\nCache statistics:")
    for key, value in cache.get_stats().items():
        print(f"  {key}: {value}")
