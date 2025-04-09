
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import os
import json
import time

from src.config.system_config import SystemConfig
from src.memory.vector_store import VectorStore
from src.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class RAGAgent:
    """
    Retrieval-Augmented Generation agent that processes content using a vector store
    and generates responses based on retrieved context.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.vector_store = VectorStore(config.vector_store)
        self.llm = LLMManager(config.llm)
    
    def process(self, query: str, content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process the scraped content using RAG techniques
        
        Args:
            query: The original research query
            content: List of dictionaries with scraped content
            
        Returns:
            Dictionary with RAG results
        """
        logger.info(f"Processing {len(content)} content items with RAG")
        
        # Extract and chunk content
        chunks = self._chunk_content(content)
        logger.info(f"Created {len(chunks)} chunks from content")
        
        # Add chunks to vector store
        chunk_ids = self.vector_store.add_texts([c["text"] for c in chunks], [c["metadata"] for c in chunks])
        
        # Perform similarity search
        top_chunks = self.vector_store.similarity_search(query, k=10)
        
        # Generate a comprehensive answer using the top chunks as context
        answer = self._generate_answer(query, top_chunks)
        
        # Generate insights and key points
        insights = self._generate_insights(query, top_chunks)
        
        # Prepare the final result
        result = {
            "answer": answer,
            "insights": insights,
            "top_chunks": top_chunks,
            "num_chunks": len(chunks),
            "timestamp": time.time()
        }
        
        return result
    
    def _chunk_content(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split content into smaller chunks suitable for embedding"""
        chunks = []
        chunk_size = 500  # Approximate words per chunk
        overlap = 50  # Words of overlap between chunks
        
        for item in content:
            text = item.get("content", "")
            if not text:
                continue
                
            # Split into words for easier chunking
            words = text.split()
            
            # Calculate number of chunks
            n_chunks = max(1, len(words) // (chunk_size - overlap))
            
            for i in range(n_chunks):
                start = max(0, i * (chunk_size - overlap))
                end = min(len(words), start + chunk_size)
                
                chunk_text = " ".join(words[start:end])
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "chunk_id": f"{len(chunks)}",
                        "relevance_score": item.get("relevance_score", 0.0)
                    }
                })
        
        return chunks
    
    def _generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive answer using the LLM and context chunks"""
        # Combine context chunks into a single context string
        context = "\n\n".join([
            f"Source: {chunk['metadata']['title']} ({chunk['metadata']['url']})\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Prompt for the LLM
        prompt = f"""
        Research Query: {query}
        
        Context Information:
        {context}
        
        Based on the provided context information, provide a comprehensive answer to the research query.
        Focus on being accurate, informative, and thorough. Cite sources where appropriate.
        """
        
        # Get response from LLM
        response = self.llm.generate(prompt)
        
        return response
    
    def _generate_insights(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate key insights from the research results"""
        # Combine context chunks
        context = "\n\n".join([
            f"Source: {chunk['metadata']['title']}\n{chunk['text'][:200]}..."
            for chunk in context_chunks[:5]  # Use fewer chunks for insights
        ])
        
        # Prompt for insights
        prompt = f"""
        Research Query: {query}
        
        Context Information:
        {context}
        
        Based on the provided context, extract 5 key insights or findings related to the research query.
        Each insight should be concise (1-2 sentences) and represent an important takeaway from the research.
        Format your response as a list with each insight on a new line.
        """
        
        # Get response from LLM
        response = self.llm.generate(prompt)
        
        # Parse response into a list of insights
        insights = [line.strip() for line in response.split("\n") if line.strip()]
        
        # Filter out non-insight lines
        insights = [line for line in insights if len(line) > 20][:5]
        
        return insights
