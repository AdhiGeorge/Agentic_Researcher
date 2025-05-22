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


# Example usage with comprehensive testing
if __name__ == "__main__":
    import os
    import sys
    import time
    import random
    from pathlib import Path
    import json
    from datetime import datetime, timedelta
    from pprint import pprint
    import sqlite3
    from collections import Counter
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("===== Query Hash Manager Example Usage =====")
    print("This example demonstrates query hash management,")
    print("deduplication, and similarity detection capabilities.")
    
    # Create temporary directory for outputs
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="query_hash_example_")
    print(f"\nCreated temporary directory for outputs: {temp_dir}")
    
    # Create query hash manager
    manager = QueryHashManager()
    
    print("\nExample 1: Basic Hash Generation and Normalization")
    print("-" * 60)
    
    # Test queries with different wording but similar meaning
    test_queries = [
        "What is the formula for calculating VIX?",
        "How is the VIX index calculated?",
        "How do you calculate the Volatility Index?",
        "What is the formula for the Volatility Index calculation?",
        "Can you explain the VIX formula?",
    ]
    
    # Demonstrate normalization first
    print("Normalization examples:")
    for i, query in enumerate(test_queries, 1):
        normalized = manager.normalize_query(query)
        print(f"\nOriginal query {i}: '{query}'")
        print(f"Normalized:      '{normalized}'")
    
    # Show all hash types for the first query
    example_query = test_queries[0]
    print("\nComparing different hash types for a single query:")
    print(f"Query: '{example_query}'")
    print(f"Exact hash:    {manager.compute_hash(example_query)}")
    print(f"Semantic hash: {manager.compute_semantic_hash(example_query)}")
    print(f"N-gram hash:   {manager.compute_ngram_hash(example_query)}")
    
    print("\nExample 2: Query Similarity Detection")
    print("-" * 60)
    
    print("A. Semantic Hash Comparison (word-order invariant):\n")
    # Create a matrix of comparisons
    semantic_matches = []
    print("Query Similarity Matrix:")
    print("-" * 40)
    
    for i, query1 in enumerate(test_queries):
        results = []
        for j, query2 in enumerate(test_queries):
            hash1 = manager.compute_semantic_hash(query1)
            hash2 = manager.compute_semantic_hash(query2)
            match = hash1 == hash2
            results.append(match)
            semantic_matches.append((i, j, match))
        
        # Display abbreviated query and match pattern
        abbrev = query1[:20] + "..." if len(query1) > 20 else query1
        match_pattern = "".join("Y" if r else "N" for r in results)
        print(f"{i+1}. {abbrev:<23} | {match_pattern}")
    
    print("\nB. N-gram Hash Comparison (character patterns):\n")
    print("Query Similarity Matrix:")
    print("-" * 40)
    
    ngram_matches = []
    for i, query1 in enumerate(test_queries):
        results = []
        for j, query2 in enumerate(test_queries):
            hash1 = manager.compute_ngram_hash(query1)
            hash2 = manager.compute_ngram_hash(query2)
            match = hash1 == hash2
            results.append(match)
            ngram_matches.append((i, j, match))
        
        # Display abbreviated query and match pattern
        abbrev = query1[:20] + "..." if len(query1) > 20 else query1
        match_pattern = "".join("Y" if r else "N" for r in results)
        print(f"{i+1}. {abbrev:<23} | {match_pattern}")
    
    print("\nExample 3: Mock Database Integration")
    print("-" * 60)
    
    # Create a simple in-memory SQLite database for demonstration
    class MockSQLiteManager:
        def __init__(self):
            self.conn = sqlite3.connect(":memory:")
            self.cursor = self.conn.cursor()
            
            # Create tables
            self.cursor.execute("""
            CREATE TABLE queries (
                id INTEGER PRIMARY KEY,
                query TEXT,
                exact_hash TEXT,
                semantic_hash TEXT,
                ngram_hash TEXT,
                project_id INTEGER,
                created_at TEXT,
                metadata TEXT
            )
            """)
            self.conn.commit()
        
        def store_query(self, query_data):
            # Store query in database
            self.cursor.execute("""
            INSERT INTO queries 
            (query, exact_hash, semantic_hash, ngram_hash, project_id, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                query_data["query"],
                query_data["exact_hash"],
                query_data["semantic_hash"],
                query_data["ngram_hash"],
                query_data.get("project_id"),
                query_data.get("created_at", datetime.now().isoformat()),
                json.dumps(query_data.get("metadata", {}))
            ))
            self.conn.commit()
            return self.cursor.lastrowid
        
        def find_similar_queries(self, exact_hash, semantic_hash, ngram_hash, limit=5):
            # Find similar queries
            self.cursor.execute("""
            SELECT query, exact_hash, semantic_hash, ngram_hash, created_at, metadata
            FROM queries
            WHERE exact_hash = ? OR semantic_hash = ? OR ngram_hash = ?
            ORDER BY 
                CASE WHEN exact_hash = ? THEN 1
                     WHEN semantic_hash = ? THEN 2
                     ELSE 3
                END,
                id DESC
            LIMIT ?
            """, (exact_hash, semantic_hash, ngram_hash, exact_hash, semantic_hash, limit))
            
            results = []
            for row in self.cursor.fetchall():
                query, e_hash, s_hash, n_hash, created_at, metadata_str = row
                
                # Determine match type
                if e_hash == exact_hash:
                    match_type = "exact"
                elif s_hash == semantic_hash:
                    match_type = "semantic"
                else:
                    match_type = "ngram"
                
                # Parse metadata
                try:
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}
                
                results.append({
                    "query": query,
                    "match_type": match_type,
                    "created_at": created_at,
                    "metadata": metadata
                })
            
            return results
        
        def get_query_history(self, project_id=None, limit=50):
            # Get query history
            if project_id is not None:
                self.cursor.execute("""
                SELECT query, created_at, metadata
                FROM queries
                WHERE project_id = ?
                ORDER BY id DESC
                LIMIT ?
                """, (project_id, limit))
            else:
                self.cursor.execute("""
                SELECT query, created_at, metadata
                FROM queries
                ORDER BY id DESC
                LIMIT ?
                """, (limit,))
            
            results = []
            for row in self.cursor.fetchall():
                query, created_at, metadata_str = row
                
                # Parse metadata
                try:
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}
                
                results.append({
                    "query": query,
                    "created_at": created_at,
                    "metadata": metadata
                })
            
            return results
        
        def get_query_stats(self, days=7):
            # Get query stats
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get total queries
            self.cursor.execute("""
            SELECT COUNT(*) FROM queries
            WHERE created_at >= ?
            """, (cutoff_date,))
            total_queries = self.cursor.fetchone()[0]
            
            # Get unique queries (by exact hash)
            self.cursor.execute("""
            SELECT COUNT(DISTINCT exact_hash) FROM queries
            WHERE created_at >= ?
            """, (cutoff_date,))
            unique_queries = self.cursor.fetchone()[0]
            
            # Get query frequency (most frequent queries)
            self.cursor.execute("""
            SELECT query, COUNT(*) as count
            FROM queries
            WHERE created_at >= ?
            GROUP BY exact_hash
            ORDER BY count DESC
            LIMIT 5
            """, (cutoff_date,))
            
            frequent_queries = []
            for row in self.cursor.fetchall():
                query, count = row
                frequent_queries.append({"query": query, "count": count})
            
            return {
                "total_queries": total_queries,
                "unique_queries": unique_queries,
                "frequent_queries": frequent_queries,
                "period_days": days
            }
    
    # Create a mock database manager
    mock_db = MockSQLiteManager()
    
    # Create a manager with the mock database
    db_manager = QueryHashManager(sqlite_manager=mock_db)
    
    # Demo dataset: queries across projects and time periods
    print("Populating mock database with example queries...")
    
    # Project IDs for demo
    projects = [1, 2, 3]  # Project 1: Finance, Project 2: Technology, Project 3: Health
    
    # Generate some example queries spanning multiple domains
    research_queries = [
        # Finance queries
        {"query": "What is the formula for calculating VIX?", "project_id": 1, "domain": "finance"},
        {"query": "How do interest rates affect stock market performance?", "project_id": 1, "domain": "finance"},
        {"query": "How is the VIX index calculated?", "project_id": 1, "domain": "finance"},
        {"query": "What are the main factors affecting cryptocurrency prices?", "project_id": 1, "domain": "finance"},
        {"query": "How is market volatility measured?", "project_id": 1, "domain": "finance"},
        
        # Technology queries
        {"query": "How does quantum computing differ from classical computing?", "project_id": 2, "domain": "technology"},
        {"query": "What is the difference between AI and machine learning?", "project_id": 2, "domain": "technology"},
        {"query": "How do large language models work?", "project_id": 2, "domain": "technology"},
        {"query": "What are the principles behind quantum computing?", "project_id": 2, "domain": "technology"},
        {"query": "How do neural networks learn from data?", "project_id": 2, "domain": "technology"},
        
        # Health queries
        {"query": "What are the most effective treatments for type 2 diabetes?", "project_id": 3, "domain": "health"},
        {"query": "How does exercise affect cardiovascular health?", "project_id": 3, "domain": "health"},
        {"query": "What is the relationship between diet and inflammation?", "project_id": 3, "domain": "health"},
        {"query": "What are the latest treatments for type 2 diabetes?", "project_id": 3, "domain": "health"},
        {"query": "How does intermittent fasting affect metabolism?", "project_id": 3, "domain": "health"},
    ]
    
    # Add some duplicates and variations to test deduplication
    additional_queries = [
        {"query": "How is VIX calculated?", "project_id": 1, "domain": "finance"},  # Variation of first query
        {"query": "What is the formula for calculating VIX?", "project_id": 1, "domain": "finance"},  # Exact duplicate of first query
        {"query": "How do quantum computers work?", "project_id": 2, "domain": "technology"},  # Variation of quantum computing query
        {"query": "What treatments work best for diabetes type 2?", "project_id": 3, "domain": "health"},  # Variation of diabetes query
    ]
    
    research_queries.extend(additional_queries)
    
    # Set random timestamps over the last 30 days
    now = datetime.now()
    for query_data in research_queries:
        # Random timestamp in the last 30 days
        days_ago = random.randint(0, 30)
        timestamp = (now - timedelta(days=days_ago)).isoformat()
        
        query_data["created_at"] = timestamp
        query_data["metadata"] = {"domain": query_data["domain"], "timestamp": timestamp}
        
        # Store in the database
        db_manager.store_query(
            query_data["query"],
            project_id=query_data["project_id"],
            metadata=query_data["metadata"]
        )
    
    print(f"Stored {len(research_queries)} queries in the database")
    
    # Now demonstrate the functionality
    print("\nA. Finding Similar Queries:")
    
    test_search_queries = [
        "How is the VIX calculated?",  # Should match finance queries
        "How do quantum computers process information?",  # Should match technology queries
        "What are effective diabetes treatments?",  # Should match health queries
    ]
    
    for query in test_search_queries:
        print(f"\nSearching for similar queries to: '{query}'")
        similar = db_manager.find_similar_queries(query, limit=3)
        
        if similar:
            print(f"Found {len(similar)} similar queries:")
            for i, match in enumerate(similar, 1):
                match_query = match.get("query", "")
                match_type = match.get("match_type", "unknown")
                print(f"  {i}. '{match_query}' (Match type: {match_type})")
        else:
            print("No similar queries found")
    
    print("\nB. Query History by Project:")
    
    for project_id in projects:
        print(f"\nProject {project_id} query history:")
        history = db_manager.get_query_history(project_id=project_id, limit=3)
        
        if history:
            for i, entry in enumerate(history, 1):
                query = entry.get("query", "")
                created_at = entry.get("created_at", "")
                # Format datetime for display
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at)
                        created_at = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                print(f"  {i}. '{query}' (Time: {created_at})")
        else:
            print("  No history found")
    
    print("\nC. Query Statistics:")
    
    # Get statistics for the last 30 days
    stats = db_manager.get_query_stats(days=30)
    
    print(f"Statistics for the last {stats.get('period_days', 30)} days:")
    print(f"  Total queries: {stats.get('total_queries', 0)}")
    print(f"  Unique queries: {stats.get('unique_queries', 0)}")
    
    if stats.get('unique_ratio') is not None:
        print(f"  Uniqueness ratio: {stats.get('unique_ratio', 0):.2f}")
    
    if stats.get('frequent_queries'):
        print("\n  Most frequent queries:")
        for i, query_data in enumerate(stats['frequent_queries'], 1):
            query = query_data.get("query", "")
            count = query_data.get("count", 0)
            print(f"    {i}. '{query}' (Count: {count})")
    
    print("\nExample 4: Research Pipeline Integration")
    print("-" * 60)
    
    # Simulate a research pipeline that uses query hashing
    def simulate_research_pipeline():
        print("Simulating a complete research pipeline with query deduplication:\n")
        
        # Step 1: User submits a new query
        user_query = "What is the relationship between market volatility and option pricing?"
        print(f"User submits query: '{user_query}'")
        
        # Step 2: Check for similar queries in history
        similar = db_manager.find_similar_queries(user_query, limit=2)
        
        if similar:
            print("Found similar previous queries:")
            for i, match in enumerate(similar, 1):
                print(f"  {i}. '{match.get('query', '')}' (Match: {match.get('match_type', '')})")
                
            # In a real system, we might offer to reuse previous results here
            print("Offering to reuse previous research results...")
            print("User chooses to continue with new query.")
        
        # Step 3: Store the new query
        print("\nStoring new query in database...")
        query_data = db_manager.store_query(
            user_query,
            project_id=1,  # Finance project
            metadata={"domain": "finance", "priority": "high"}
        )
        
        # Step 4: Simulate performing research
        print("Performing research steps:")
        print("  1. Searching external sources")
        print("  2. Extracting and analyzing content")
        print("  3. Generating research summary")
        
        # Step 5: Update query metadata with results
        print("\nResearch complete! Updating query data with results")
        
        # In a real system, we would update the database here
        
        # Step 6: When user makes a follow-up query, detect similarity
        follow_up = "How does volatility affect option prices?"
        print(f"\nUser makes follow-up query: '{follow_up}'")
        
        # Check similarity
        similar = db_manager.find_similar_queries(follow_up, limit=3)
        if similar:
            print("Detected as related to previous query!")
            print("System can now provide continuity in the research process.")
        
        return "Research pipeline simulation completed"
    
    # Run the simulation
    result = simulate_research_pipeline()
    print(f"\n{result}")
    
    print("\n" + "=" * 80)
    print("Query Hash Manager examples completed!")
    print("This utility enables efficient query deduplication and")
    print("similarity detection for research continuity.")
    print("=" * 80)
