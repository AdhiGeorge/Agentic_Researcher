
import os
import logging
import json
from typing import Dict, List, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from src.config.system_config import VectorStoreConfig

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store for embeddings using Qdrant.
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        
        # Initialize Qdrant client
        if config.url:
            # Use remote Qdrant instance
            self.client = QdrantClient(
                url=config.url,
                api_key=config.api_key
            )
        else:
            # Use local Qdrant instance
            self.client = QdrantClient(location=config.location)
        
        # Check if collection exists, create if not
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if config.collection_name not in collection_names:
            # Create new collection
            self.client.create_collection(
                collection_name=config.collection_name,
                vectors_config=models.VectorParams(
                    size=config.dimension,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new Qdrant collection: {config.collection_name}")
        
        logger.info(f"Initialized vector store with embedding model: {config.embedding_model}")
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of text strings to embed and store
            metadatas: Optional list of metadata dictionaries, one per text
            
        Returns:
            List of IDs for the added texts
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Generate embeddings for all texts
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Generate IDs
        ids = [str(hash(text + str(metadata))) for text, metadata in zip(texts, metadatas)]
        
        # Prepare points for Qdrant
        points = [
            models.PointStruct(
                id=id,
                vector=embedding.tolist(),
                payload={"text": text, **metadata}
            )
            for id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas)
        ]
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
        
        logger.info(f"Added {len(texts)} texts to vector store")
        return ids
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search for the query.
        
        Args:
            query: The query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries with text, score, and metadata
        """
        if not self.client.collection_exists(self.config.collection_name):
            logger.warning(f"Collection {self.config.collection_name} does not exist")
            return []
        
        # Generate embedding for query
        query_embedding = self.model.encode(query)
        
        # Perform search
        search_results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k
        )
        
        # Format results
        results = []
        for res in search_results:
            result = {
                "text": res.payload.get("text", ""),
                "score": res.score,
                "metadata": {k: v for k, v in res.payload.items() if k != "text"}
            }
            results.append(result)
        
        logger.info(f"Returning {len(results)} results for query: {query[:30]}...")
        return results
    
    def clear(self):
        """Clear the vector store"""
        if self.client.collection_exists(self.config.collection_name):
            self.client.delete_collection(self.config.collection_name)
            # Recreate the empty collection
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.dimension,
                    distance=models.Distance.COSINE
                )
            )
            logger.info("Vector store cleared")
