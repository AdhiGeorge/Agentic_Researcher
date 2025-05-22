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


# Example usage with real-world scenarios and API calls
if __name__ == "__main__":
    import os
    import sys
    import time
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("===== Semantic Cache Example Usage =====")
    print("This example demonstrates the functionality of the semantic cache")
    print("with realistic queries and optional integration with real embeddings.")
    
    # Try to import from parent directory if running the file directly
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = current_dir.parent.parent
    if parent_dir not in sys.path:
        sys.path.append(str(parent_dir))
    
    # Try to use real embeddings if available
    use_real_embeddings = False
    embedding_function = None
    
    try:
        from src.utils.embedder import TextEmbedder
        from src.utils.config import Config
        
        # Initialize config and embedder
        config = Config()
        
        if config.get_azure_openai_config()['api_key']:
            print("\nUsing real Azure OpenAI embeddings!")
            embedder = TextEmbedder()
            embedding_function = embedder.embed_text
            use_real_embeddings = True
        else:
            print("\nAzure OpenAI API key not found. Using dummy embeddings.")
    except (ImportError, Exception) as e:
        print(f"\nCould not initialize real embeddings: {str(e)}")
        print("Using dummy embeddings for demonstration purposes.")
    
    # If real embeddings aren't available, use dummy function
    if not use_real_embeddings:
        def embedding_function(text):
            # Generate deterministic pseudo-random embeddings based on text content
            # to ensure similar texts get similar embeddings
            import hashlib
            import numpy as np
            
            # Get hash of text and use it as seed
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            seed = int(text_hash[:8], 16)  # Use first 8 chars of hash as seed
            np.random.seed(seed)
            
            # Generate embedding vector (dimension 1536 like text-embedding-3-small)
            embedding = np.random.normal(0, 1, 1536).tolist()
            
            # Normalize to unit length (cosine similarity preparation)
            magnitude = np.sqrt(sum(x*x for x in embedding))
            normalized = [x/magnitude for x in embedding]
            
            return normalized
    
    # Create database directory if needed (for optional real persistence)
    db_dir = Path(current_dir) / "example_data"
    db_dir.mkdir(exist_ok=True)
    
    # Optional: Setup SQLite persistence (commented out)
    # try:
    #     from src.utils.sqlite_manager import SQLiteManager
    #     sqlite_manager = SQLiteManager(db_path=str(db_dir / "semantic_cache.db"))
    # except ImportError:
    #     sqlite_manager = None
    #     print("SQLite persistence not available. Using in-memory cache.")
    
    # Create cache with the embedding function
    cache = SemanticCache(
        # sqlite_manager=sqlite_manager,  # Uncomment to use persistence
        embedding_function=embedding_function,
        default_threshold=0.90,
        min_threshold=0.85,
        max_threshold=0.95,
        cache_ttl=3600 * 24  # 1 day for this example
    )
    
    print("\nExample 1: Basic hash-based exact matching")
    print("-" * 50)
    
    # Test queries
    q1 = "What is the VIX index and how is it calculated?"
    
    # Check if entry exists (it shouldn't yet)
    result = cache.get(q1)
    print(f"Query: {q1}")
    print(f"Initial cache check: {'Hit' if result else 'Miss'}")
    
    # Store result
    example_result = {
        "answer": "The VIX (Volatility Index) is calculated using near- and next-term put and call options with more than 23 days and less than 37 days to expiration. The formula uses a weighted average of these options' prices to derive the expected volatility.",
        "sources": ["Chicago Board Options Exchange documentation"],
        "timestamp": time.time()
    }
    
    print("\nStoring result in cache...")
    cache.put(q1, example_result)
    
    # Try exact match again
    result = cache.get(q1)
    print(f"\nCache check after storing: {'Hit' if result else 'Miss'}")
    if result:
        print(f"Retrieved answer: {result.get('result', {}).get('answer', '')[:50]}...")
        print(f"Match type: {'Exact' if result.get('match_type') == 'exact' else 'Semantic'}")
    
    print("\nExample 2: Semantic similarity matching")
    print("-" * 50)
    
    # Similar queries with different wording
    similar_queries = [
        "How do you calculate the Volatility Index (VIX)?",
        "What's the formula for computing VIX?",
        "Explain the VIX calculation methodology.",
        "How is the CBOE Volatility Index determined?"
    ]
    
    print("Testing semantically similar queries against our cached entry:\n")
    
    for i, query in enumerate(similar_queries, 1):
        print(f"Query {i}: {query}")
        result = cache.get(query)
        
        if result:
            print(f"  Result: Hit")
            print(f"  Match type: {result.get('match_type', 'unknown')}")
            print(f"  Similarity: {result.get('similarity', 0):.4f}")
            print(f"  Original query: {result.get('original_query', '')}")
        else:
            print(f"  Result: Miss")
        print()
    
    print("Example 3: Completely different queries")
    print("-" * 50)
    
    unrelated_queries = [
        "What is the capital of France?",
        "How do neural networks work?",
        "Explain quantum computing principles."
    ]
    
    print("These should be cache misses:\n")
    
    for i, query in enumerate(unrelated_queries, 1):
        print(f"Query {i}: {query}")
        result = cache.get(query)
        
        if result:
            print(f"  Unexpected cache hit with similarity: {result.get('similarity', 0):.4f}")
        else:
            print(f"  Expected cache miss")
        print()
    
    # Store some of these for the next test
    cache.put(unrelated_queries[0], {"answer": "The capital of France is Paris.", "timestamp": time.time()})
    cache.put(unrelated_queries[1], {"answer": "Neural networks are computational systems inspired by the biological neural networks in animal brains...", "timestamp": time.time()})
    
    print("Example 4: Cache invalidation")
    print("-" * 50)
    
    print("Testing cache invalidation for specific query hash:")
    # Get the hash for the first unrelated query
    query_hash = cache.compute_query_hash(unrelated_queries[0])
    print(f"Invalidating entry for '{unrelated_queries[0]}'")
    
    # Check if it exists
    before = cache.get(unrelated_queries[0]) is not None
    print(f"Entry exists before invalidation: {before}")
    
    # Invalidate it
    invalidated = cache.invalidate(query_hash=query_hash)
    print(f"Invalidated {invalidated} entries")
    
    # Check if it exists after
    after = cache.get(unrelated_queries[0]) is not None
    print(f"Entry exists after invalidation: {after}")
    
    print("\nTesting time-based invalidation:")
    print("This would normally invalidate entries older than the specified time,")
    print("but for demonstration, we'll just show the API.")
    print("cache.invalidate(older_than=3600)  # Invalidate entries older than 1 hour")
    
    print("\nExample 5: Cache statistics")
    print("-" * 50)
    
    # Print cache statistics
    stats = cache.get_stats()
    print("Cache statistics:")
    for key, value in stats.items():
        # Format the value based on its type
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif isinstance(value, list):
            formatted_value = f"[{', '.join(str(x) for x in value)}]"
        else:
            formatted_value = str(value)
        
        print(f"  {key}: {formatted_value}")
    
    # Performance test (optional)
    if use_real_embeddings:
        print("\nExample 6: Performance test with real embeddings")
        print("-" * 50)
        
        test_queries = similar_queries + unrelated_queries
        print(f"Testing performance with {len(test_queries)} queries...")
        
        start_time = time.time()
        for query in test_queries:
            result = cache.get(query)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(test_queries) * 1000  # ms
        print(f"Average query time: {avg_time:.2f} ms per query")
    
    print("\n" + "=" * 80)
    print("Semantic Cache example completed!")
    print("This utility helps improve response time and reduce API costs")
    print("by caching results and finding semantically similar previous queries.")
    print("=" * 80)
