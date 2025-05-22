"""
Azure OpenAI Client for Agentic Researcher

This module provides a unified client for interacting with Azure OpenAI services,
including chat completions and embeddings using various models like GPT-4o and text-embedding-3-small.

The module offers both object-oriented and functional interfaces for flexibility.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Handle both direct execution and module import scenarios
try:
    # When imported as a module
    from src.utils.config import config
except ModuleNotFoundError:
    # When run directly as a script
    from config import config

# Configure logging
logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    """Client for Azure OpenAI API with retries and configuration management
    
    This class handles connections to Azure OpenAI API services, including:
    - Chat completions using the GPT-4o model
    - Embeddings using the text-embedding-3-small model
    - Automatic retries with exponential backoff
    - Configuration management
    """
    
    def __init__(self):
        """Initialize the Azure OpenAI client with configuration"""
        # Get Azure OpenAI credentials from config
        self.api_key = config.azure_openai_api_key
        self.endpoint = config.azure_openai_endpoint
        self.api_version_chat = config.azure_api_version_chat
        self.api_version_embeddings = config.azure_api_version_embeddings
        self.chat_deployment = config.azure_openai_deployment
        self.embedding_deployment = config.azure_openai_embedding_deployment
        
        # Create clients for chat and embeddings (with different API versions)
        self.chat_client = self._create_client(self.api_version_chat)
        self.embedding_client = self._create_client(self.api_version_embeddings)
        
        logger.info(f"Initialized Azure OpenAI client with chat model {self.chat_deployment} and embedding model {self.embedding_deployment}")
    
    def _create_client(self, api_version: str) -> AzureOpenAI:
        """Create an Azure OpenAI client with the specified API version
        
        Args:
            api_version: API version to use
            
        Returns:
            AzureOpenAI: Configured client
        """
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=api_version,
            azure_endpoint=self.endpoint
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_completion(self, messages: List[Dict[str, str]], 
                           temperature: float = 0.3,
                           max_tokens: int = 2000,
                           model: str = None,
                           top_p: float = 0.95,
                           frequency_penalty: float = 0.0,
                           presence_penalty: float = 0.0) -> str:
        """Generate a completion using the chat API with retries
        
        Args:
            messages: List of message dictionaries with role and content
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            model: Optional model name override
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            
        Returns:
            str: Generated completion text
        """
        try:
            start_time = time.time()
            
            # Use the specified model or the default model from config
            model_name = model or self.chat_deployment
            
            # Make API call with retry logic
            response = self.chat_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            
            # Log the API call duration
            duration = time.time() - start_time
            logger.debug(f"Azure OpenAI completion generated in {duration:.2f}s (tokens: {response.usage.total_tokens})")
            
            # Extract and return the completion text
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_embeddings(self, texts: Union[str, List[str]], model: str = None) -> List[List[float]]:
        """Generate embeddings for one or more texts using the embeddings API with retries
        
        Args:
            texts: Text or list of texts to embed
            model: Optional model name override
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            start_time = time.time()
            
            # Ensure input is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Use the specified model or the default model from config
            model_name = model or self.embedding_deployment
            
            # Make API call with retry logic
            response = self.embedding_client.embeddings.create(
                model=model_name,
                input=texts
            )
            
            # Log the API call duration
            duration = time.time() - start_time
            logger.debug(f"Azure OpenAI embeddings generated in {duration:.2f}s (count: {len(texts)})")
            
            # Extract and return the embeddings
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_single_embedding(self, text: str, model: str = None) -> List[float]:
        """Get an embedding for a single text string
        
        Args:
            text: Text to embed
            model: Optional model name override
            
        Returns:
            List[float]: Embedding vector
        """
        embeddings = self.generate_embeddings(text, model)
        return embeddings[0] if embeddings else []

#============================================================================
# Functional Interface - Backwards Compatibility Layer
#============================================================================

# Global client instance for singleton access pattern
_openai_client = None

def _get_client() -> AzureOpenAIClient:
    """Get or create the singleton Azure OpenAI client.
    
    Returns:
        AzureOpenAIClient: Azure OpenAI client instance
    """
    global _openai_client
    
    if _openai_client is None:
        try:
            _openai_client = AzureOpenAIClient()
            logger.info("Azure OpenAI client initialized")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {str(e)}")
            raise
            
    return _openai_client

def get_chat_client():
    """Get the Azure OpenAI client for chat completions.
    
    Returns:
        An object with the chat completions API
    """
    # Return the client's raw client object for chat functions
    return _get_client().chat_client

def get_embedding_client():
    """Get the Azure OpenAI client for embeddings.
    
    Returns:
        An object with the embeddings API
    """
    # Return the client's raw client object for embedding functions
    return _get_client().embedding_client

def generate_chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """Generate a chat completion using Azure OpenAI.
    
    Args:
        system_prompt (str): System prompt
        user_prompt (str): User prompt
        model (str, optional): Model to use. Defaults to config default.
        temperature (float, optional): Temperature. Defaults to 0.7.
        max_tokens (int, optional): Maximum tokens. Defaults to 1000.
        
    Returns:
        str: Generated text
    """
    try:
        client = _get_client()
        
        # Create messages in the format expected by the API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use the robust implementation that includes retries
        return client.generate_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model  # Will use default if None
        )
    except Exception as e:
        logger.error(f"Error generating chat completion: {str(e)}")
        return f"Error generating response: {str(e)}"

def generate_embedding(text: str, model: str = None) -> List[float]:
    """Generate an embedding for the given text using Azure OpenAI.
    
    Args:
        text (str): Text to embed
        model (str, optional): Model to use. Defaults to config default.
        
    Returns:
        List[float]: Embedding vector
    """
    try:
        client = _get_client()
        
        # Use the robust implementation with retries
        embeddings = client.generate_embeddings(text, model)
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        else:
            raise ValueError("No embeddings returned")
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Return empty embedding on error with the correct dimension
        return [0.0] * 1536  # Default size for text-embedding-3-small

def information(query: str, model: str = "gpt-4o"):
    """Generate a response to the given query using GPT-4o.
    
    Args:
        query (str): The query
        model (str, optional): Model name. Defaults to "gpt-4o".
        
    Returns:
        str: Generated response
    """
    client = _get_client()
    
    # Create messages in the format expected by the API
    messages = [
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": query}
    ]
    
    # Use the robust implementation with retries
    return client.generate_completion(
        messages=messages,
        model=model
    )

def embedding(data: str, model: str = "text-embedding-3-small"):
    """Generate an embedding for the given data.
    
    Args:
        data (str): Text to embed
        model (str, optional): Model name. Defaults to "text-embedding-3-small".
        
    Returns:
        List[float]: Embedding vector
    """
    # Simply use the generate_embedding function as they do the same thing
    # The model parameter is handled by the AzureOpenAIClient configuration
    return generate_embedding(data, model)

# Example usage with actual Azure OpenAI API calls
if __name__ == "__main__":
    import os
    import json
    import time
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Configure logging for standalone usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n===== Azure OpenAI Client - Real API Examples =====\n")
    
    # Ensure API credentials are available
    if not config.azure_openai_api_key or not config.azure_openai_endpoint:
        print("ERROR: Azure OpenAI credentials not found. Ensure your environment variables are set:")
        print("  AZURE_OPENAI_API_KEY: Your Azure OpenAI API key")
        print("  AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint")
        print("  AZURE_OPENAI_DEPLOYMENT: Your Azure OpenAI deployment name")
        print("  AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Your Azure OpenAI embedding model deployment")
        print("\nSkipping examples due to missing credentials.")
        import sys
        sys.exit(1)
    
    try:
        print("EXAMPLE 1: OBJECT-ORIENTED API USAGE")
        print("-" * 50)
        client = AzureOpenAIClient()
        
        # Display configuration
        print(f"Using Azure OpenAI endpoint: {client.endpoint}")
        print(f"Chat model deployment: {client.chat_deployment}")
        print(f"Embedding model deployment: {client.embedding_deployment}")
        
        # Example 1: Generate a chat completion for Quantum Computing research
        print("\n1. Generating Chat Completion - Research Summary")
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": "You are a scientific research assistant specializing in quantum computing."},
            {"role": "user", "content": "Summarize the latest breakthroughs in quantum error correction in 3-4 sentences."}
        ]
        
        completion = client.generate_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=150
        )
        
        duration = time.time() - start_time
        print(f"\nResponse (generated in {duration:.2f}s):")
        print(f"\"{completion}\"")
        
        # Example 2: Generate embeddings for semantic search
        print("\n2. Generating Embeddings for Semantic Search")
        search_queries = [
            "How do quantum computers perform factorization?",
            "What is quantum supremacy and why is it important?",
            "Latest developments in quantum machine learning"
        ]
        
        start_time = time.time()
        embeddings = client.generate_embeddings(search_queries)
        duration = time.time() - start_time
        
        print(f"Generated {len(embeddings)} embeddings in {duration:.2f}s")
        print(f"Embedding dimensions: {len(embeddings[0])}")
        print(f"First query embedding (first 5 values): {embeddings[0][:5]}")
        
        # Example 3: Calculating embedding similarity for semantic search
        print("\n3. Calculating Embedding Similarity for Semantic Search")
        
        # Create document embeddings
        documents = [
            "Quantum computers use qubits instead of classical bits to perform computations.",
            "Quantum supremacy refers to when a quantum computer solves a problem faster than classical computers.",
            "Quantum machine learning combines quantum computing with AI algorithms for potential speedups."
        ]
        
        # Get embeddings for documents
        doc_embeddings = client.generate_embeddings(documents)
        
        # Get embedding for a search query
        query = "Tell me about quantum supremacy"
        query_embedding = client.get_single_embedding(query)
        
        # Simple cosine similarity function
        def cosine_similarity(vec1, vec2):
            import numpy as np
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append({"document": documents[i], "similarity": similarity})
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Display results
        print(f"Query: \"{query}\"")
        print("\nRanked Results by Similarity:")
        for i, result in enumerate(similarities, 1):
            print(f"{i}. Similarity: {result['similarity']:.4f}")
            print(f"   Document: \"{result['document']}\"")
        
        print("\nEXAMPLE 4: FUNCTIONAL API USAGE")
        print("-" * 50)
        
        # Generate a chat completion using the functional API
        print("\n1. Generating Chat Completion with Functional API")
        system_prompt = "You are an AI research assistant specializing in quantum algorithms."
        user_query = "Explain Shor's algorithm for quantum factorization in simple terms."
        
        start_time = time.time()
        completion = generate_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_query,
            temperature=0.5,
            max_tokens=200
        )
        duration = time.time() - start_time
        
        print(f"\nResponse (generated in {duration:.2f}s):")
        print(f"\"{completion}\"")
        
        # Generate embeddings using functional API
        print("\n2. Generating Embeddings with Functional API")
        embedding_text = "What is quantum entanglement?"
        
        start_time = time.time()
        embedding = generate_embedding(embedding_text)
        duration = time.time() - start_time
        
        print(f"Embedding generated in {duration:.2f}s")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # EXAMPLE 5: Real-world application - Simple question answering system
        print("\n\nEXAMPLE 5: REAL-WORLD APPLICATION - QUESTION ANSWERING")
        print("-" * 50)
        
        print("Building a simple question answering system with context")
        
        # Sample knowledge base
        knowledge_base = [
            "Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data.",
            "Quantum entanglement is a physical phenomenon that occurs when pairs or groups of particles are generated, interact, or share spatial proximity in ways such that the quantum state of each particle cannot be described independently of the others.",
            "Quantum supremacy refers to the experimental demonstration of a quantum computer's solving a problem that classical computers practically cannot solve.",
            "Shor's algorithm is a quantum algorithm for integer factorization, formulated in 1994 by the American mathematician Peter Shor. It can break RSA encryption by finding the prime factors of a number efficiently on a quantum computer.",
            "Quantum error correction (QEC) is used to protect quantum information from errors due to decoherence and other quantum noise. It is essential for fault-tolerant quantum computing."
        ]
        
        # Embed the knowledge base
        kb_embeddings = client.generate_embeddings(knowledge_base)
        
        # Process a user question
        user_question = "How does quantum computing relate to encryption?"
        question_embedding = client.get_single_embedding(user_question)
        
        # Find the most relevant knowledge base entry
        similarities = [cosine_similarity(question_embedding, kb_embedding) for kb_embedding in kb_embeddings]
        max_sim_idx = similarities.index(max(similarities))
        context = knowledge_base[max_sim_idx]
        
        print(f"\nUser question: \"{user_question}\"")
        print(f"\nRetrieved context: \"{context}\"")
        print(f"Similarity score: {similarities[max_sim_idx]:.4f}")
        
        # Generate answer with context
        system_message = "You are a helpful assistant specializing in quantum computing. Use the provided context to answer the question accurately and concisely."
        prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:"
        
        answer = client.generate_completion([
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ])
        
        print("\nGenerated answer:")
        print(f"\"{answer}\"")
        
    except Exception as e:
        print(f"\nError in example: {str(e)}")
        import traceback
        traceback.print_exc()
