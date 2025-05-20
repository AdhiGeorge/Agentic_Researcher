"""
SQLite Database Manager for Agentic Researcher
Manages state, conversations, and query caching with SQLite
"""
import os
import sys
import json
import sqlite3
import time
import hashlib
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from src.utils.config import config as ConfigLoader

# Configure logging
logger = logging.getLogger(__name__)

class SQLiteManager:
    """
    SQLite Manager for state management, conversations, and query caching
    
    This class provides:
    1. Agent state management (storing current state and history)
    2. Conversation tracking (storing user queries and agent responses)
    3. Query caching (storing previous queries and results)
    4. Project management (tracking research projects)
    5. Raw scraped data storage for later reference
    """
    
    def __init__(self):
        """Initialize the SQLiteManager"""
        # Use direct access to config properties instead of get_value method
        # Set default database path since it's not in the Config class
        self.db_path = "agentic_researcher.db"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        # Initialize embedding functionality
        try:
            # Use Azure OpenAI for embeddings
            # Import here to avoid circular imports
            from src.utils.openai_client import _get_client, AzureOpenAIClient
            # Get the full client instance, not just the underlying API client
            self.openai_client = _get_client()
            self.embedding_model = self.openai_client  # Use the full client with embedding methods
            logger.info("Using Azure OpenAI for embeddings in SQLite similarity search")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            self.openai_client = None
            self.embedding_model = None
            
        # Initialize the database
        self._init_database()
        
        logger.info(f"SQLiteManager initialized with database at {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check for schema version
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Get current schema version
            cursor.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            current_version = result[0] if result else 0
            
            # Run migrations if needed
            if current_version < 1:
                self._migrate_to_v1(conn)
                # Update schema version
                cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (1,))
                conn.commit()
                
            # Run migration to v2 if needed (for processed flag in scraped_data)
            if current_version < 2:
                self._migrate_to_v2(conn)
                # Update schema version
                cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (2,))
                conn.commit()
                
            # Run migration to v3 if needed (for semantic_hash column in query_cache)
            if current_version < 3:
                self._migrate_to_v3(conn)
                # Update schema version
                cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (3,))
                conn.commit()
            
            # Create projects table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create agent_states table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_states (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                agent_type TEXT NOT NULL,
                state_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
            ''')
            
            # Create conversations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
            ''')
            
            # Create queries table with hash and embedding columns
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                embedding BLOB,
                state_data TEXT,
                cached BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
            ''')
            
            # Create scraped_data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraped_data (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                content TEXT,
                metadata TEXT,
                processed BOOLEAN DEFAULT 0,
                vector_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
            ''')
            
            # Create query_cache table for fast exact and semantic matching
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_cache (
                id INTEGER PRIMARY KEY,
                query_hash TEXT NOT NULL,
                semantic_hash TEXT,
                ngram_hash TEXT,
                query TEXT NOT NULL,
                result TEXT NOT NULL,
                similarity_score REAL,
                embedding BLOB,
                project_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
            ''')
            
            # Create indices for fast lookups
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_query_hash ON query_cache(query_hash)
            ''')
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_semantic_hash ON query_cache(semantic_hash)
            ''')
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ngram_hash ON query_cache(ngram_hash)
            ''')
            
            conn.commit()
    
    def _migrate_to_v1(self, conn):
        """Migrate database schema to version 1"""
        cursor = conn.cursor()
        
        try:
            # Check if project_id column exists in query_cache table
            cursor.execute("PRAGMA table_info(query_cache)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'project_id' not in column_names:
                logger.info("Migrating query_cache table to include project_id column")
                
                # Create a new table with the updated schema
                cursor.execute('''
                CREATE TABLE query_cache_new (
                    id INTEGER PRIMARY KEY,
                    query_hash TEXT NOT NULL UNIQUE,
                    query TEXT NOT NULL,
                    result TEXT NOT NULL,
                    similarity_score REAL,
                    embedding BLOB,
                    project_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
                ''')
                
                # Copy data from the old table to the new one
                cursor.execute('''
                INSERT INTO query_cache_new (id, query_hash, query, result, similarity_score, embedding, created_at, accessed_at)
                SELECT id, query_hash, query, result, similarity_score, embedding, created_at, accessed_at FROM query_cache
                ''')
                
                # Drop the old table
                cursor.execute("DROP TABLE query_cache")
                
                # Rename the new table to the original name
                cursor.execute("ALTER TABLE query_cache_new RENAME TO query_cache")
                
                # Recreate the index
                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_query_hash ON query_cache(query_hash)
                ''')
                
                logger.info("Successfully migrated query_cache table")
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error during migration to v1: {str(e)}")
            conn.rollback()
        
    def _migrate_to_v2(self, conn):
        """Migrate database schema to version 2 (add processed flag to scraped_data)"""
        cursor = conn.cursor()
        
        try:
            # Check if scraped_data table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scraped_data'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check if processed column already exists
                cursor.execute("PRAGMA table_info(scraped_data)")
                columns = cursor.fetchall()
                processed_exists = any(column[1] == 'processed' for column in columns)
                
                if not processed_exists:
                    logger.info("Adding processed column to scraped_data table")
                    # Add processed column
                    cursor.execute("ALTER TABLE scraped_data ADD COLUMN processed INTEGER DEFAULT 0")
                    conn.commit()
                
                # Check if vector_id column already exists
                vector_id_exists = any(column[1] == 'vector_id' for column in columns)
                
                if not vector_id_exists:
                    logger.info("Adding vector_id column to scraped_data table")
                    # Add vector_id column
                    cursor.execute("ALTER TABLE scraped_data ADD COLUMN vector_id TEXT")
                    conn.commit()
            
            logger.info("Migration to schema version 2 completed")
            conn.commit()
        except Exception as e:
            logger.error(f"Error during migration to v2: {str(e)}")
            conn.rollback()
    
    def _migrate_to_v3(self, conn):
        """Migrate database schema to version 3 (ensure semantic_hash column exists in query_cache)"""
        cursor = conn.cursor()
        
        try:
            # Check if query_cache table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_cache'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check if semantic_hash column already exists
                cursor.execute("PRAGMA table_info(query_cache)")
                columns = cursor.fetchall()
                semantic_hash_exists = any(column[1] == 'semantic_hash' for column in columns)
                
                if not semantic_hash_exists:
                    logger.info("Adding semantic_hash column to query_cache table")
                    # Add semantic_hash column
                    cursor.execute("ALTER TABLE query_cache ADD COLUMN semantic_hash TEXT")
                    conn.commit()
                    
                    # Create index on the new column
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_hash ON query_cache(semantic_hash)")
                    conn.commit()
                
                # Check if ngram_hash column already exists
                ngram_hash_exists = any(column[1] == 'ngram_hash' for column in columns)
                
                if not ngram_hash_exists:
                    logger.info("Adding ngram_hash column to query_cache table")
                    # Add ngram_hash column
                    cursor.execute("ALTER TABLE query_cache ADD COLUMN ngram_hash TEXT")
                    conn.commit()
                    
                    # Create index on the new column
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ngram_hash ON query_cache(ngram_hash)")
                    conn.commit()
            
            logger.info("Migration to schema version 3 completed")
            conn.commit()
        except Exception as e:
            logger.error(f"Error during migration to v3: {str(e)}")
            conn.rollback()
    
    def create_project(self, name: str, description: str = "") -> int:
        """
        Create a new project
        
        Args:
            name: Project name
            description: Project description
            
        Returns:
            int: Project ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if project already exists
            cursor.execute(
                "SELECT id FROM projects WHERE name = ?",
                (name,)
            )
            existing = cursor.fetchone()
            
            if existing:
                logger.info(f"Project '{name}' already exists with ID {existing[0]}")
                return existing[0]
            
            # Create new project
            cursor.execute(
                "INSERT INTO projects (name, description, status) VALUES (?, ?, ?)",
                (name, description, "created")
            )
            conn.commit()
            
            project_id = cursor.lastrowid
            logger.info(f"Created new project '{name}' with ID {project_id}")
            
            return project_id
    
    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Get project by ID
        
        Args:
            project_id: Project ID
            
        Returns:
            Optional[Dict]: Project data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM projects WHERE id = ?",
                (project_id,)
            )
            
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            
            return None
    
    def update_project_status(self, project_id: int, status: str) -> bool:
        """
        Update project status
        
        Args:
            project_id: Project ID
            status: New status
            
        Returns:
            bool: True if successful, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE projects SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, project_id)
            )
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    def save_agent_state(self, project_id: int, agent_type: str, state_data: Dict[str, Any]) -> int:
        """
        Save agent state
        
        Args:
            project_id: Project ID
            agent_type: Type of agent (planner, researcher, etc.)
            state_data: State data (will be JSON serialized)
            
        Returns:
            int: State ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Serialize state data to JSON
            state_json = json.dumps(state_data)
            
            cursor.execute(
                "INSERT INTO agent_states (project_id, agent_type, state_data) VALUES (?, ?, ?)",
                (project_id, agent_type, state_json)
            )
            
            conn.commit()
            
            return cursor.lastrowid
    
    def get_agent_states(self, project_id: int, agent_type: Optional[str] = None, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get agent states
        
        Args:
            project_id: Project ID
            agent_type: Optional agent type to filter
            limit: Maximum number of states to return
            
        Returns:
            List[Dict]: List of agent states
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM agent_states WHERE project_id = ?"
            params = [project_id]
            
            if agent_type:
                query += " AND agent_type = ?"
                params.append(agent_type)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                data = dict(row)
                # Parse state_data from JSON
                data["state_data"] = json.loads(data["state_data"])
                result.append(data)
            
            return result
    
    def add_conversation_message(self, project_id: int, sender: str, message: str) -> int:
        """
        Add a message to the conversation
        
        Args:
            project_id: Project ID
            sender: Message sender (user, agent, etc.)
            message: Message content
            
        Returns:
            int: Message ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO conversations (project_id, sender, message) VALUES (?, ?, ?)",
                (project_id, sender, message)
            )
            
            conn.commit()
            
            return cursor.lastrowid
    
    def get_conversation(self, project_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            project_id: Project ID
            limit: Maximum number of messages to return
            
        Returns:
            List[Dict]: List of conversation messages
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM conversations WHERE project_id = ? ORDER BY timestamp ASC LIMIT ?",
                (project_id, limit)
            )
            
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def store_query(self, project_id: int, query: str, state_data: Optional[Dict[str, Any]] = None,
                  cached: bool = False) -> int:
        """
        Store a query with hash and embedding
        
        Args:
            project_id: Project ID
            query: The query string
            state_data: Optional state data associated with the query
            cached: Whether the query has cached results
            
        Returns:
            int: Query ID
        """
        # Generate query hash
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Generate embedding if model is available
        embedding_blob = None
        if self.embedding_model:
            try:
                # Use get_single_embedding method from Azure OpenAI client
                embedding = self.embedding_model.get_single_embedding(query)
                # Convert the list of floats to numpy array and then to bytes
                import numpy as np
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert state_data to JSON if provided
            state_json = json.dumps(state_data) if state_data is not None else None
            
            cursor.execute(
                "INSERT INTO queries (project_id, query, query_hash, embedding, state_data, cached) VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, query, query_hash, embedding_blob, state_json, cached)
            )
            
            conn.commit()
            
            return cursor.lastrowid
    
    def find_exact_query_match(self, query: str, project_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Find exact match for a query using hash
        
        Args:
            query: Query string
            project_id: Optional project ID to filter
            
        Returns:
            Optional[Dict]: Matching query or None if not found
        """
        # Generate query hash
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if project_id:
                cursor.execute(
                    "SELECT * FROM queries WHERE query_hash = ? AND project_id = ? AND cached = 1",
                    (query_hash, project_id)
                )
            else:
                cursor.execute(
                    "SELECT * FROM queries WHERE query_hash = ? AND cached = 1",
                    (query_hash,)
                )
            
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                
                # Parse state_data from JSON
                if result["state_data"]:
                    result["state_data"] = json.loads(result["state_data"])
                
                # Convert embedding blob to numpy array if present
                if result["embedding"]:
                    result["embedding"] = np.frombuffer(result["embedding"], dtype=np.float32)
                
                return result
            
            return None
    
    def find_similar_query(self, query: str, project_id: Optional[int] = None, 
                         threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """
        Find semantically similar query using embeddings
        
        Args:
            query: Query string
            project_id: Optional project ID to filter
            threshold: Similarity threshold (0-1)
            
        Returns:
            Optional[Dict]: Most similar query or None if no match above threshold
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available for similarity search")
            return None
        
        try:
            # Generate query embedding using the correct method
            query_embedding = self.embedding_model.get_single_embedding(query)
            # Convert to numpy array for vector operations
            import numpy as np
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get all cached queries
                if project_id:
                    cursor.execute(
                        "SELECT * FROM queries WHERE project_id = ? AND embedding IS NOT NULL AND cached = 1",
                        (project_id,)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM queries WHERE embedding IS NOT NULL AND cached = 1"
                    )
                
                rows = cursor.fetchall()
                
                # Calculate similarities
                most_similar = None
                highest_similarity = 0
                
                for row in rows:
                    row_dict = dict(row)
                    
                    # Get embedding
                    embedding_blob = row_dict["embedding"]
                    if not embedding_blob:
                        continue
                    
                    # Convert blob to numpy array
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar = row_dict
                
                # Return most similar if above threshold
                if most_similar and highest_similarity >= threshold:
                    result = most_similar
                    
                    # Parse state_data from JSON
                    if result["state_data"]:
                        result["state_data"] = json.loads(result["state_data"])
                    
                    # Add similarity score
                    result["similarity"] = highest_similarity
                    
                    return result
                
                return None
        
        except Exception as e:
            logger.error(f"Error finding similar query: {str(e)}")
            return None
    
    def store_scraped_data(self, project_id: int, url: str, title: str, 
                         content: str, metadata: Dict[str, Any], 
                         processed: bool = False, vector_id: Optional[str] = None) -> int:
        """
        Store raw scraped data
        
        Args:
            project_id: Project ID
            url: URL that was scraped
            title: Page title
            content: Raw scraped content
            metadata: Metadata about the scrape
            processed: Whether this data has been processed and stored in Qdrant
            vector_id: ID of the vector in Qdrant if processed
            
        Returns:
            int: Scraped data ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert metadata to JSON
            metadata_json = json.dumps(metadata)
            
            cursor.execute(
                "INSERT INTO scraped_data (project_id, url, title, content, metadata, processed, vector_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (project_id, url, title, content, metadata_json, 1 if processed else 0, vector_id)
            )
            
            conn.commit()
            
            return cursor.lastrowid
    
    def get_scraped_data(self, project_id: int, url: Optional[str] = None, limit: int = 100, processed: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get scraped data
        
        Args:
            project_id: Project ID
            url: Optional URL to filter
            limit: Maximum number of records to return
            processed: Filter by processed status (True/False/None for all)
            
        Returns:
            List[Dict]: List of scraped data records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM scraped_data WHERE project_id = ?"
            params = [project_id]
            
            if url:
                query += " AND url = ?"
                params.append(url)
                
            if processed is not None:
                query += " AND processed = ?"
                params.append(1 if processed else 0)
            
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                item = dict(row)
                
                # Parse JSON metadata
                if "metadata" in item and item["metadata"]:
                    try:
                        item["metadata"] = json.loads(item["metadata"])
                    except json.JSONDecodeError:
                        item["metadata"] = {}
                
                results.append(item)
            
            return results
            
    def mark_scraped_data_processed(self, scraped_id: int, vector_id: str) -> bool:
        """
        Mark a scraped data record as processed with its vector ID
        
        Args:
            scraped_id: ID of the scraped data record
            vector_id: ID of the vector in Qdrant
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "UPDATE scraped_data SET processed = 1, vector_id = ? WHERE id = ?",
                    (vector_id, scraped_id)
                )
                
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error marking scraped data as processed: {str(e)}")
            return False
            
    def get_unprocessed_scraped_data(self, project_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get scraped data that hasn't been processed yet
        
        Args:
            project_id: Project ID
            limit: Maximum number of records to return
            
        Returns:
            List[Dict]: List of unprocessed scraped data records
        """
        return self.get_scraped_data(project_id=project_id, processed=False, limit=limit)
    
    def cache_query_result(self, query: str, result: Dict[str, Any], project_id: Optional[int] = None, query_hash: Optional[str] = None, similarity_score: float = 0.0) -> int:
        """
        Cache a query result for future use
        
        Args:
            query: The query string
            result: The query result to cache
            project_id: Optional project ID to associate with the cached query
            query_hash: MD5 hash of the query for exact matching (generated if not provided)
            similarity_score: Similarity score for the result
            
        Returns:
            int: Query ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert result to JSON string
                result_json = json.dumps(result)
                
                # Generate query hash if not provided
                if query_hash is None:
                    query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
                
                # Generate embedding for the query if model is available
                embedding_blob = None
                if self.embedding_model:
                    try:
                        embedding = self.embedding_service.encode(query)
                        embedding_blob = embedding.tobytes()
                    except Exception as e:
                        logger.warning(f"Error generating embedding for query cache: {str(e)}")
                
                # Prepare SQL query based on whether project_id is provided
                if project_id is not None:
                    # Insert or replace the cached query with project_id
                    cursor.execute(
                        """INSERT OR REPLACE INTO query_cache 
                           (query_hash, query, result, similarity_score, embedding, project_id, created_at) 
                           VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
                        (query_hash, query, result_json, similarity_score, embedding_blob, project_id)
                    )
                else:
                    # Insert or replace the cached query without project_id
                    cursor.execute(
                        """INSERT OR REPLACE INTO query_cache 
                           (query_hash, query, result, similarity_score, embedding, created_at) 
                           VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                        (query_hash, query, result_json, similarity_score, embedding_blob)
                    )
                
                conn.commit()
                query_id = cursor.lastrowid
            
            logger.info(f"Cached query result for hash: {query_hash}, query: {query[:50]}..., project_id: {project_id}")
            return query_id
        except Exception as e:
            logger.error(f"Error caching query result: {str(e)}")
            return -1
            
    def close(self):
        """Close any open resources"""
        try:
            # Release embedding model resources if needed
            # (No explicit close needed for SentenceTransformer)
            self.embedding_model = None
                
            logger.info("SQLiteManager resources released")
        except Exception as e:
            logger.error(f"Error closing SQLiteManager: {str(e)}")
            
    def get_connection(self):
        """Get a SQLite connection
        
        Returns:
            sqlite3.Connection: Database connection
        """
        return sqlite3.connect(self.db_path)
            
    def check_query_cache(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Check if a query hash exists in the cache
        
        Args:
            query_hash: MD5 hash of the query
            
        Returns:
            Optional[Dict]: Cached result or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Look for exact match by hash
                cursor.execute(
                    "SELECT * FROM query_cache WHERE query_hash = ?",
                    (query_hash,)
                )
                
                row = cursor.fetchone()
                
                if row:
                    data = dict(row)
                    
                    # Parse result from JSON
                    if data["result"]:
                        data["result"] = json.loads(data["result"])
                        return data["result"]
            
            return None
        except Exception as e:
            logger.error(f"Error checking query cache: {str(e)}")
            return None


# Example usage when run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create SQLiteManager
    db = SQLiteManager()
    
    # Test project creation
    project_id = db.create_project("VIX Research", "Research on Volatility Index calculation")
    print(f"Created project with ID: {project_id}")
    
    # Test agent state
    state_id = db.save_agent_state(
        project_id=project_id,
        agent_type="planner",
        state_data={"status": "planning", "step": 1}
    )
    print(f"Saved agent state with ID: {state_id}")
    
    # Test conversation
    message_id = db.add_conversation_message(
        project_id=project_id,
        sender="user",
        message="What is volatility index and what is the mathematical formula to calculate the VIX score?"
    )
    print(f"Added conversation message with ID: {message_id}")
    
    # Test query storage with embedding
    query = "How to calculate VIX scores in Python"
    query_id = db.store_query(
        project_id=project_id,
        query=query,
        state_data={"status": "completed", "result": "Some result"},
        cached=True
    )
    print(f"Stored query with ID: {query_id}")
    
    # Test query retrieval
    similar_query = db.find_similar_query(
        query="Python code for computing volatility index",
        project_id=project_id
    )
    
    if similar_query:
        print(f"Found similar query with similarity score: {similar_query.get('similarity', 0)}")
        print(f"Original query: {similar_query.get('query')}")
    else:
        print("No similar query found")
