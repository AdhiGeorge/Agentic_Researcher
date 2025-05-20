"""
Qdrant Vector Database Manager for Agentic Researcher
Handles vector embeddings and similarity search for research data
"""
import os
import sys
import json
import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union

# Add project root to Python path to allow direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.utils.config import config
from src.utils.text_processors import TextSplitter

# Configure logging
logger = logging.getLogger(__name__)

class QdrantManager:
    """
    Qdrant Vector Database Manager for storing and retrieving research data
    
    This class provides:
    1. Vector embeddings for text using SentenceTransformers
    2. Storage of research data (web content, PDF content, etc.)
    3. Semantic search for finding similar content
    4. Storage and retrieval of processed research results
    """
    
    def __init__(self, collection_name: str = "research_data"):
        """
        Initialize the QdrantManager
        
        Args:
            collection_name: Name of the Qdrant collection
        """
        # We're using the global config object imported at the top of the file
        # No need to create a local instance
        
        # Initialize text splitting utilities
        from src.utils.text_processors import SemanticChunker, TextSplitter
        self.chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
        self.text_splitter = TextSplitter(default_chunk_size=800, default_overlap=50)
        
        # Get Qdrant settings
        self.qdrant_host = config.qdrant_url  # Using the correct property name from the config
        self.qdrant_port = config.qdrant_port
        
        # Collection settings
        self.collection_name = collection_name
        self.vector_size = 768  # Default for SentenceTransformer models
        
        # Initialize Azure OpenAI embedding client
        self._init_embedding_client()
        
        # Connect to Qdrant
        self._init_qdrant_client()
        
        logger.info(f"QdrantManager initialized with collection '{collection_name}'")
    
    def _init_embedding_client(self):
        """Initialize the Azure OpenAI embedding client"""
        try:
            # Import Azure client
            from src.utils.openai_client import get_embedding_client
            self.embedding_client = get_embedding_client()
            
            # Set vector size for Azure OpenAI text-embedding-3-small
            self.embedding_dimension = 1536  # This is the dimension for text-embedding-3-small
            self.vector_size = self.embedding_dimension  # Default to embedding model's dimension
            
            logger.info(f"Initialized Azure OpenAI embedding client with {self.embedding_dimension} dimensions")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI embedding client: {str(e)}")
            raise
    
    def _init_qdrant_client(self):
        """Initialize the Qdrant client and create collection if it doesn't exist"""
        try:
            # Connect to Qdrant
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            logger.info(f"Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.vector_size,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection '{self.collection_name}' in Qdrant")
            else:
                # Collection exists, check if vector dimensions match
                try:
                    collection_info = self.client.get_collection(collection_name=self.collection_name)
                    existing_vector_size = collection_info.config.params.vectors.size
                    
                    if existing_vector_size != self.embedding_dimension:
                        # Use a more informative message
                        logger.info(f"Collection '{self.collection_name}' uses {existing_vector_size} dimensions, while embedding model produces {self.embedding_dimension} dimensions")
                        # Store the collection's vector size to use for encoding
                        self.vector_size = existing_vector_size
                        logger.info(f"Will automatically adjust vectors to {self.vector_size} dimensions for compatibility")
                except Exception as e:
                    logger.warning(f"Could not check vector dimensions: {str(e)}")
                
                logger.info(f"Collection '{self.collection_name}' already exists in Qdrant")
        
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {str(e)}")
            raise
    
    def _encode_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using Azure OpenAI
        
        Args:
            text: Text to encode
            
        Returns:
            List[float]: Text embedding
        """
        try:
            # Generate embedding using Azure OpenAI client
            # Use the generate_single_embedding method from our OpenAI client wrapper
            if hasattr(self.embedding_client, 'get_single_embedding'):
                # Use the wrapper method if available
                vector = self.embedding_client.get_single_embedding(text)
            else:
                # Direct API call as fallback
                from src.utils.openai_client import generate_embedding
                vector = generate_embedding(text)
            
            # Convert to list if it's a numpy array
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
                
            return vector
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zeros as fallback
            return [0.0] * self.vector_size
    
    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 50, metadata: Dict = None, entities: List[Dict] = None) -> List[str]:
        """
        Split text into semantic chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            metadata: Metadata to attach to chunks
            entities: Entities to preserve across chunks
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            # Update text splitter settings if provided
            self.text_splitter.default_chunk_size = chunk_size
            self.text_splitter.default_overlap = overlap
            
            metadata = metadata or {}
            
            # Try semantic chunking first
            try:
                if entities and hasattr(self.chunker, 'chunk_with_entities'):
                    # Use entity-aware chunking
                    chunk_objects = self.chunker.chunk_with_entities(text, entities, metadata)
                else:
                    # Use standard semantic chunking
                    chunk_objects = self.chunker.create_chunks(text, metadata)
                    
                # Extract just the text from each chunk object
                return [chunk.get('text', '') if isinstance(chunk, dict) else chunk for chunk in chunk_objects]
            except Exception as e:
                logger.warning(f"Semantic chunking failed, falling back to basic chunking: {str(e)}")
                # Fallback to standard text splitter
                return self.text_splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            # Last resort: basic chunking
            return [text]
    
    def store_document(self, document: Dict[str, Any], project_id: int, 
                      chunk_size: int = 800, overlap: int = 50) -> List[int]:
        """
        Store a document in Qdrant
        
        Args:
            document: Document with text, metadata, etc.
            project_id: Project ID
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            List[int]: List of point IDs in Qdrant
        """
        try:
            # Extract text and metadata
            text = document.get("text", "")
            url = document.get("url", "")
            title = document.get("title", "")
            
            if not text:
                logger.warning(f"Empty text for document {url or title}")
                return []
            
            # Split text into chunks
            chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            
            if not chunks:
                logger.warning(f"No chunks generated for document {url or title}")
                return []
            
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Generate point IDs
            timestamp = int(time.time() * 1000)
            point_ids = []
            
            # Store each chunk
            for i, chunk in enumerate(chunks):
                # Generate a unique ID for this chunk
                point_id = timestamp + i
                point_ids.append(point_id)
                
                # Generate embedding
                embedding = self._encode_text(chunk)
                
                # Prepare metadata
                metadata = {
                    "project_id": project_id,
                    "url": url,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": timestamp,
                    "source": document.get("source", "web")
                }
                
                # Add custom metadata if available
                if "metadata" in document and isinstance(document["metadata"], dict):
                    # Filter out any non-serializable values
                    filtered_metadata = {}
                    for k, v in document["metadata"].items():
                        try:
                            # Test JSON serialization
                            json.dumps({k: v})
                            filtered_metadata[k] = v
                        except:
                            # Skip values that can't be serialized
                            pass
                    
                    metadata.update(filtered_metadata)
                
                # Prepare point for storing in Qdrant
                # Ensure the embedding is the right format (list of floats)
                if isinstance(embedding, list):
                    vector = embedding
                elif hasattr(embedding, 'tolist'):
                    # Convert numpy array to list
                    vector = embedding.tolist()
                else:
                    # Ensure it's a list of floats
                    vector = list(map(float, embedding))
                
                # Adjust vector dimensions if needed (without repetitive warnings)
                if len(vector) != self.vector_size:
                    if len(vector) > self.vector_size:
                        # Truncate the vector
                        vector = vector[:self.vector_size]
                    elif len(vector) < self.vector_size:
                        # Pad with zeros
                        padding = [0.0] * (self.vector_size - len(vector))
                        vector = vector + padding
                
                point = {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "text": chunk,
                        "metadata": metadata
                    }
                }
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
            
            logger.info(f"Stored {len(point_ids)} chunks in Qdrant for project {project_id}")
            return point_ids
        
        except Exception as e:
            logger.error(f"Error storing document in Qdrant: {str(e)}")
            return []
    
    def store_multiple_documents(self, documents: List[Dict[str, Any]], project_id: int,
                               chunk_size: int = 800, overlap: int = 50) -> Dict[str, List[int]]:
        """
        Store multiple documents in Qdrant
        
        Args:
            documents: List of documents to store
            project_id: Project ID
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            Dict[str, List[int]]: Mapping of document URLs to point IDs
        """
        results = {}
        
        for document in documents:
            url = document.get("url", "")
            point_ids = self.store_document(document, project_id, chunk_size, overlap)
            
            if url and point_ids:
                results[url] = point_ids
        
        return results
    
    def search_similar(self, query: str, project_id: Optional[int] = None, 
                      limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for content similar to the query
        
        Args:
            query: Search query
            project_id: Optional project ID to filter results
            limit: Maximum number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List[Dict]: List of matching results with text and metadata
        """
        try:
            # Generate query embedding
            query_vector = self._encode_text(query)
            
            # Ensure the vector has the correct dimension for the collection (without excessive warnings)
            if len(query_vector) != self.vector_size:
                # Handle the dimension mismatch silently
                if len(query_vector) > self.vector_size:
                    # Truncate the vector if it's too large
                    query_vector = query_vector[:self.vector_size]
                elif len(query_vector) < self.vector_size:
                    # Pad with zeros if it's too small
                    padding = [0.0] * (self.vector_size - len(query_vector))
                    query_vector = query_vector + padding
                
                # Verify dimensions match now
                if len(query_vector) != self.vector_size:
                    logger.error("Failed to adjust vector dimensions")
                    return []
            
            # Prepare filter
            search_filter = None
            if project_id is not None:
                search_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.project_id",
                            match=qdrant_models.MatchValue(value=project_id)
                        )
                    ]
                )
            
            # Execute search
            # Ensure the query vector is properly formatted
            if hasattr(query_vector, 'tolist'):
                # Convert numpy array to list
                query_vector = query_vector.tolist()
            
            # Search for similar content using the modern API
            try:
                # Use the search method with deprecation warning suppressed
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    search_results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        query_filter=search_filter,
                        limit=limit,
                        score_threshold=threshold
                    )
            except Exception as e:
                logger.error(f"Error searching in Qdrant: {str(e)}")
                return []
            
            # Format results
            results = []
            for result in search_results:
                # Extract text and metadata
                payload = result.payload
                text = payload.get("text", "")
                metadata = payload.get("metadata", {})
                
                # Ensure content field exists for backward compatibility with researcher.py
                content = text
                
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": text,
                    "content": content,  # Add content field for backward compatibility
                    "metadata": metadata
                })
            
            logger.info(f"Found {len(results)} similar results for query '{query}'")
            return results
        
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            return []
    
    def delete_project_data(self, project_id: int) -> bool:
        """
        Delete all data for a project
        
        Args:
            project_id: Project ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create filter for the project
            delete_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.project_id",
                        match=qdrant_models.MatchValue(value=project_id)
                    )
                ]
            )
            
            # Delete points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.FilterSelector(filter=delete_filter)
            )
            
            logger.info(f"Deleted all data for project {project_id} from Qdrant")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting project data from Qdrant: {str(e)}")
            return False
    
    def close(self):
        """Close the Qdrant client connection"""
        if hasattr(self, 'client'):
            # No explicit close method in qdrant_client, but we could set it to None
            self.client = None
            logger.info("Qdrant client connection closed")


# Example usage when run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*80)
    print("QDRANT MANAGER EXAMPLE USAGE")
    print("="*80)
    print("This example demonstrates how to use QdrantManager for vector storage and retrieval")
    print()
    
    # Create a unique collection for this test run
    import time
    collection_name = f"test_collection_{int(time.time())}"  
    print(f"Creating new collection: {collection_name}")
    
    # Initialize the QdrantManager
    qdrant = QdrantManager(collection_name=collection_name)
    
    # Define a project ID for our test data
    project_id = 12345
    
    print("\n1. STORING INDIVIDUAL DOCUMENTS\n" + "-"*30)
    
    # Store multiple individual test documents
    docs = [
        {
            "url": "https://example.com/quantum-computing",
            "title": "Introduction to Quantum Computing",
            "text": "Quantum computing is a type of computing that uses quantum mechanics phenomena such as superposition and entanglement to perform operations on data. Quantum computers use qubits instead of classical bits, allowing them to solve certain problems much faster than classical computers.",
            "source": "research_paper"
        },
        {
            "url": "https://example.com/machine-learning",
            "title": "Machine Learning Fundamentals",
            "text": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data. These systems automatically improve with experience without being explicitly programmed. Common algorithms include neural networks, decision trees, and support vector machines.",
            "source": "textbook"
        },
        {
            "url": "https://example.com/blockchain",
            "title": "Understanding Blockchain Technology",
            "text": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records called blocks. Each block contains a timestamp and a link to the previous block, making the data tamper-resistant. It forms the foundation for cryptocurrencies like Bitcoin and Ethereum.",
            "source": "blog"
        }
    ]
    
    # Store each document and collect the point IDs
    all_point_ids = []
    for i, doc in enumerate(docs):
        print(f"Storing document {i+1}: {doc['title']}")
        point_ids = qdrant.store_document(doc, project_id=project_id)
        all_point_ids.extend(point_ids)
        print(f"  -> Stored with point IDs: {point_ids}")
    
    print(f"\nTotal points stored: {len(all_point_ids)}")
    
    print("\n2. SEMANTIC SEARCH EXAMPLES\n" + "-"*30)
    
    # Example search queries to demonstrate semantic search
    search_queries = [
        "quantum physics and computing",
        "artificial intelligence algorithms",
        "distributed ledger technologies"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        # Use a lower threshold (0.5) to get more semantic matches
        results = qdrant.search_similar(query, project_id=project_id, limit=3, threshold=0.5)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\nResult {i+1} [Score: {result['score']:.4f}]")
            print(f"Title: {result['metadata'].get('title', 'No title')}")
            print(f"Text: {result['text'][:100]}...")
            print(f"URL: {result['metadata'].get('url', 'N/A')}")
    
    print("\n3. BATCH DOCUMENT STORAGE\n" + "-"*30)
    
    # Store multiple documents at once
    print("Storing multiple documents in batch mode")
    
    batch_docs = [
        {
            "url": "https://example.com/nlp",
            "title": "Natural Language Processing",
            "text": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables computers to understand, interpret, and generate human language in a valuable way.",
            "source": "research"
        },
        {
            "url": "https://example.com/cloud-computing",
            "title": "Cloud Computing Architectures",
            "text": "Cloud computing delivers computer services over the internet, including storage, databases, networking, software, and analytics. It offers faster innovation, flexible resources, and economies of scale.",
            "source": "whitepaper"
        }
    ]
    
    batch_results = qdrant.store_multiple_documents(batch_docs, project_id=project_id)
    print(f"Batch storage results: {batch_results}")
    
    # Cross-referencing search - find cloud computing content
    print("\nSearching across all stored documents for 'cloud services'")
    cloud_results = qdrant.search_similar("cloud services and infrastructure", project_id=project_id, threshold=0.5)
    
    for i, result in enumerate(cloud_results):
        print(f"\nResult {i+1} [Score: {result['score']:.4f}]")
        print(f"Title: {result['metadata'].get('title', 'No title')}")
        print(f"Text: {result['text'][:100]}...")
    
    print("\n4. CLEAN UP\n" + "-"*30)
    print(f"Deleting all data for project {project_id}")
    deleted = qdrant.delete_project_data(project_id=project_id)
    print(f"Deletion successful: {deleted}")
    
    # Close the connection
    qdrant.close()
    print("\nQdrant Manager example completed successfully!")
    print("="*80)