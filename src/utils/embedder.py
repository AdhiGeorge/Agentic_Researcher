"""
Embedder utility for Agentic Researcher
Uses Azure OpenAI to generate embeddings for text
"""
import time
import openai
import os
import sys
from typing import List

# Handle imports to work both when imported as a module and when run directly
if __name__ == "__main__":
    # Add project root to path when running as script
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Direct relative import when running as script
    from config import Config as ConfigLoader
else:
    # Regular import when imported as a module
    from src.utils.config import Config as ConfigLoader

class TextEmbedder:
    """
    Text embedder utility using Azure OpenAI API
    Generates embeddings for text chunks
    """
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Load configuration
        self.config = ConfigLoader()
        
        # Flag to track if we'll use mock embeddings
        self.use_mock_embeddings = False
        self.mock_embedding_dim = 1536  # Standard OpenAI embedding size
        
        try:
            # Get Azure OpenAI credentials from config class
            azure_embeddings_config = self.config.get_azure_embeddings_config()
            api_key = azure_embeddings_config.get("api_key")
            api_version = azure_embeddings_config.get("api_version")
            azure_endpoint = azure_embeddings_config.get("azure_endpoint")
            
            # Print the credentials for debugging (without showing the full API key)
            api_key_masked = api_key[:5] + '...' + api_key[-5:] if len(api_key) > 10 else '***'
            print(f"Azure OpenAI API Key: {api_key_masked}")
            print(f"Azure OpenAI API Version: {api_version}")
            print(f"Azure OpenAI Endpoint: {azure_endpoint}")
            
            # Get the embedding deployment name
            self.embedding_deployment = azure_embeddings_config.get("deployment")
            print(f"Azure OpenAI Embedding Deployment: {self.embedding_deployment}")
            
            # Initialize Azure OpenAI client
            try:
                print("Initializing Azure OpenAI client...")
                self.client = openai.AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint
                )
                print("Azure OpenAI client initialized successfully.")
            except Exception as client_init_error:
                print(f"Error initializing Azure OpenAI client: {str(client_init_error)}")
                raise client_init_error
            
            # For Azure OpenAI, we need to create a deployment with a specific name
            # The deployment name is what we use in the API calls, not the model name
            
            # Test the embedding deployment with a small request
            print(f"Testing Azure OpenAI embedding deployment: {self.embedding_deployment}")
            try:
                # Try to make a network request to the Azure OpenAI endpoint first to check connectivity
                import requests
                print(f"Checking connectivity to Azure OpenAI endpoint: {azure_endpoint}")
                response = requests.get(azure_endpoint, timeout=10)
                print(f"Connectivity check response status: {response.status_code}")
                
                # Now try the actual embedding API call
                print(f"Making embedding API call with deployment: {self.embedding_deployment}")
                response = self.client.embeddings.create(
                    input="test",
                    model=self.embedding_deployment
                )
            except requests.exceptions.RequestException as req_error:
                print(f"Network error when connecting to Azure OpenAI endpoint: {str(req_error)}")
                raise Exception(f"Network connectivity issue: {str(req_error)}")
            except Exception as api_error:
                print(f"Error during embedding API call: {str(api_error)}")
                raise api_error
            
            # If we get here, the embedding deployment exists and works
            print(f"Azure OpenAI API configured correctly.")
            print(f"Embedding model deployment '{self.embedding_deployment}' is active.")
            print(f"Generated embedding dimension: {len(response.data[0].embedding)}")
            
        except Exception as e:
            error_msg = str(e)
            if "unknown_model" in error_msg:
                print(f"Azure OpenAI doesn't recognize model '{self.embedding_deployment}'.")
                print("In Azure OpenAI, you need to create a deployment of an embedding model and use that deployment name.")
            elif "DeploymentNotFound" in error_msg:
                print(f"Embedding model deployment '{self.embedding_deployment}' not found.")
                print("To use real embeddings, create an embedding model deployment in your Azure OpenAI resource.")
            elif "AuthenticationError" in error_msg:
                print("Azure OpenAI authentication failed. Check your API key and endpoint.")
            elif "Connection error" in error_msg:
                print("Connection error when trying to reach Azure OpenAI. Check your network connection and endpoint URL.")
            else:
                print(f"Embedding test failed: {error_msg}")
                
            # We want to fail if there's an issue with Azure OpenAI
            raise e
            
        # Batch settings
        self.max_batch_size = 16  # Maximum texts to embed in a single API call
        self.retry_count = 3
        self.retry_delay = 2  # seconds
        
        self._initialized = True
        
    def _get_mock_embedding(self, text_count=1):
        """Generate mock embeddings when Azure OpenAI embedding model is not available"""
        if hasattr(self, 'np') and self.np:
            # Use numpy for better random distributions if available
            vectors = [
                self.np.random.normal(0, 0.1, self.mock_embedding_dim).tolist()
                for _ in range(text_count)
            ]
        else:
            # Fallback to simple random if numpy is not available
            vectors = [
                [random.uniform(-0.1, 0.1) for _ in range(self.mock_embedding_dim)]
                for _ in range(text_count)
            ]
            
        # Normalize the vectors (important for similarity calculations)
        normalized_vectors = []
        for vec in vectors:
            # Calculate magnitude
            magnitude = sum(x*x for x in vec) ** 0.5
            # Normalize
            if magnitude > 0:
                normalized_vectors.append([x/magnitude for x in vec])
            else:
                normalized_vectors.append(vec)  # Avoid division by zero
                
        return normalized_vectors if text_count > 1 else normalized_vectors[0]

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        # Use Azure OpenAI embedding model
        for attempt in range(self.retry_count):
            try:
                # Using the OpenAI API v1.0.0+ syntax
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_deployment
                )
                # The structure of the response in v1.0.0+
                return response.data[0].embedding
            except Exception as e:
                if attempt < self.retry_count - 1:
                    print(f"Retry {attempt+1}/{self.retry_count} for embedding generation: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    # If all retries fail, raise the exception
                    print(f"Error using Azure embedding after {self.retry_count} attempts: {str(e)}")
                    raise e
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        Uses batching to optimize API calls
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            
            # Use Azure OpenAI API
            for attempt in range(self.retry_count):
                try:
                    # Using the OpenAI API v1.0.0+ syntax
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.embedding_deployment
                    )
                    
                    # Sort the results by index to maintain original order
                    batch_embeddings = sorted(response.data, key=lambda x: x.index)
                    batch_results = [item.embedding for item in batch_embeddings]
                    results.extend(batch_results)
                    
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < self.retry_count - 1:
                        print(f"Retry {attempt+1}/{self.retry_count} for batch embedding generation: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        # If all retries fail, raise the exception
                        print(f"Error using Azure embedding batch after {self.retry_count} attempts: {str(e)}")
                        raise e
        
        return results


# Example usage
if __name__ == "__main__":
    import numpy as np
    from pprint import pprint
    
    print("TextEmbedder Example Usage")
    print("-" * 50)
    
    try:
        # Initialize the embedder (singleton pattern)
        embedder = TextEmbedder()
        
        # Check if API keys and deployment are properly configured
        has_valid_config = True
        mock_data_reason = ""
        
        # First test: Do a quick check to see if we have basic credentials
        try:
            api_key = embedder.config.get_azure_api_key()
            api_endpoint = embedder.config.get_azure_endpoint()
            deployment_name = embedder.embedding_deployment
            
            if not api_key or api_key == "YOUR_API_KEY_HERE" or not api_endpoint:
                has_valid_config = False
                mock_data_reason = "Azure OpenAI API credentials not configured"
            elif not deployment_name:
                has_valid_config = False
                mock_data_reason = "Azure OpenAI embedding deployment name not set"
        except Exception as e:
            has_valid_config = False
            mock_data_reason = f"Configuration error: {str(e)}"
            
        # If credentials look ok, try a small test call to confirm deployment exists
        if has_valid_config:
            try:
                # Test with the smallest possible text to minimize token usage
                test_result = embedder.embed_text("test")
                print("\nAzure OpenAI API configured correctly.")
                print(f"Embedding model deployment '{deployment_name}' is active.")
                print(f"Generated embedding dimension: {len(test_result)}")
            except Exception as e:
                has_valid_config = False
                error_msg = str(e)
                if "DeploymentNotFound" in error_msg:
                    mock_data_reason = f"Deployment '{deployment_name}' not found in your Azure OpenAI resource"
                else:
                    mock_data_reason = f"API error: {error_msg}"
        
        # If we're using mock data, explain why
        if not has_valid_config:
            print(f"\nWarning: {mock_data_reason}")
            print("Using mock data for demonstration purposes.")
            print("No API calls will be made, but the TextEmbedder interface will function properly.")
            print("\nTo configure Azure OpenAI:")
            print("1. Get API key and endpoint from your Azure OpenAI resource")
            print("2. Create an embedding model deployment (e.g. 'text-embedding-ada-002')")
            print("3. Update your configuration with these values")
        
        # Example 1: Generate embedding for a single text
        print("\nExample 1: Generating embedding for a single text")
        text = "This is a sample text for embedding generation."
        print(f"Input text: '{text}'")
        
        if has_valid_config:
            try:
                # Actual API call if configuration is valid
                embedding = embedder.embed_text(text)
                
                # Print partial results (first 5 dimensions)
                print(f"\nEmbedding vector (first 5 dimensions): {embedding[:5]}")
                print(f"Embedding dimension: {len(embedding)}")
                print(f"Embedding type: {type(embedding)}")
            except Exception as e:
                print(f"\nError generating embedding: {str(e)}")
                print("This could be due to API rate limits, network issues, or invalid credentials.")
        else:
            # Mock result for demonstration purposes
            mock_embedding = np.random.rand(1536).tolist()
            print("\nMock embedding vector (first 5 dimensions):", mock_embedding[:5])
            print(f"Mock embedding dimension: {len(mock_embedding)}")
            print("Note: This is simulated data since API credentials are not configured.")
        
        # Example 2: Generate embeddings for a batch of texts
        print("\nExample 2: Generating embeddings for multiple texts (batch)")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can process natural language.",
            "Embeddings capture semantic meaning of text."
        ]
        print("Input texts:")
        for i, t in enumerate(texts):
            print(f"{i+1}. '{t}'")
        
        if has_valid_config:
            try:
                # Actual batch API call if configuration is valid
                embeddings = embedder.embed_batch(texts)
                
                # Print partial results
                print("\nGenerated embeddings (showing first 3 dimensions):")
                for i, emb in enumerate(embeddings):
                    print(f"Text {i+1}: {emb[:3]}...")
                print(f"\nNumber of embeddings: {len(embeddings)}")
                print(f"Dimensions per embedding: {len(embeddings[0])}")
            except Exception as e:
                print(f"\nError generating batch embeddings: {str(e)}")
                print("This could be due to API rate limits, network issues, or invalid credentials.")
        else:
            # Mock results for demonstration
            mock_embeddings = [np.random.rand(1536).tolist() for _ in range(len(texts))]
            print("\nMock embeddings (showing first 3 dimensions):")
            for i, emb in enumerate(mock_embeddings):
                print(f"Text {i+1}: {emb[:3]}...")
            print(f"\nNumber of mock embeddings: {len(mock_embeddings)}")
            print(f"Dimensions per mock embedding: {len(mock_embeddings[0])}")
            print("Note: These are simulated embeddings since API credentials are not configured.")
        
        # Example 3: Compare similarity between embeddings (cosine similarity)
        print("\nExample 3: Comparing text similarity using embeddings")
        
        def cosine_similarity(vec1, vec2):
            # Calculate cosine similarity between two vectors
            dot_product = sum(a*b for a, b in zip(vec1, vec2))
            norm_a = sum(a*a for a in vec1) ** 0.5
            norm_b = sum(b*b for b in vec2) ** 0.5
            return dot_product / (norm_a * norm_b)
        
        if has_valid_config:
            try:
                similarity_texts = [
                    "Artificial intelligence is transforming industries.",
                    "AI is changing how companies operate.",
                    "Quantum physics explores subatomic particles."
                ]
                print("Computing similarity between:")
                for i, t in enumerate(similarity_texts):
                    print(f"{i+1}. '{t}'")
                
                similarity_embeddings = embedder.embed_batch(similarity_texts)
                
                # Compare first two texts (semantically similar)
                sim_1_2 = cosine_similarity(similarity_embeddings[0], similarity_embeddings[1])
                # Compare first and third texts (semantically different)
                sim_1_3 = cosine_similarity(similarity_embeddings[0], similarity_embeddings[2])
                
                print(f"\nSimilarity between texts 1 and 2: {sim_1_2:.4f}")
                print(f"Similarity between texts 1 and 3: {sim_1_3:.4f}")
                print("Note: Higher values (closer to 1.0) indicate greater similarity")
            except Exception as e:
                print(f"\nError in similarity comparison: {str(e)}")
        else:
            # Mock similarity results
            print("\nMock similarity results:")
            print("Similarity between semantically similar texts: 0.8734")
            print("Similarity between semantically different texts: 0.3421")
            print("Note: These are simulated similarity scores for demonstration purposes.")
        
        print("\nTextEmbedder example completed successfully.")
        
    except Exception as e:
        print(f"\nError in embedder example: {str(e)}")