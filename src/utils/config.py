"""
Configuration manager for Agentic Researcher
Loads environment variables from .env file and provides defaults
"""
import os
import dotenv
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """
    Configuration manager for Agentic Researcher
    Provides access to environment variables with defaults
    """
    
    def __init__(self):
        """Initialize the configuration manager"""
        # Load environment variables from .env file
        dotenv.load_dotenv()
        
        # Azure OpenAI API configuration
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "08b98992f8ab46c39ac14597735e1f82")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://bu01azureopenai.openai.azure.com")
        self.azure_api_version_chat = os.getenv("AZURE_API_VERSION_CHAT", "2025-01-01-preview")
        self.azure_api_version_embeddings = os.getenv("AZURE_API_VERSION_EMBEDDINGS", "2023-05-15")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        # Search API configuration
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyATbsFgbbci8T_mJ5wtfPAROfWFd-DhQ4c")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID", "e37191c870ec64043")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "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")
        
        # Database configuration
        self.qdrant_url = os.getenv("QDRANT_URL", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        # Directory paths
        self.project_root = Path(__file__).parent.parent.parent.absolute()
        self.logs_dir = os.getenv("LOGS_DIR", str(self.project_root / "logs"))
        self.scraped_data_dir = os.getenv("SCRAPED_DATA_DIR", str(self.project_root / "scraped_data"))
        self.processed_data_dir = os.getenv("PROCESSED_DATA_DIR", str(self.project_root / "processed_data"))
        
        # Create directories if they don't exist
        Path(self.logs_dir).mkdir(exist_ok=True)
        Path(self.scraped_data_dir).mkdir(exist_ok=True)
        Path(self.processed_data_dir).mkdir(exist_ok=True)
        
        # Scraping configuration
        self.scraping_headless = os.getenv("SCRAPING_HEADLESS", "True").lower() in ("true", "1", "t")
        self.scraping_timeout = int(os.getenv("SCRAPING_TIMEOUT", "30000"))
        self.scraping_user_agent_rotation = os.getenv("SCRAPING_USER_AGENT_ROTATION", "True").lower() in ("true", "1", "t")
        self.scraping_respect_robots = os.getenv("SCRAPING_RESPECT_ROBOTS", "True").lower() in ("true", "1", "t")
        self.scraping_retry_attempts = int(os.getenv("SCRAPING_RETRY_ATTEMPTS", "3"))
        self.scraping_retry_delay = int(os.getenv("SCRAPING_RETRY_DELAY", "2"))
        self.scraping_user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]
    
    def get_azure_openai_config(self) -> Dict[str, str]:
        """
        Get Azure OpenAI API configuration
        
        Returns:
            Dict[str, str]: Dictionary of configuration values
        """
        return {
            "api_key": self.azure_openai_api_key,
            "api_version": self.azure_api_version_chat,
            "azure_endpoint": self.azure_openai_endpoint,
            "deployment": self.azure_openai_deployment
        }
    
    def get_azure_embeddings_config(self) -> Dict[str, str]:
        """
        Get Azure OpenAI embeddings configuration
        
        Returns:
            Dict[str, str]: Dictionary of configuration values
        """
        return {
            "api_key": self.azure_openai_api_key,
            "api_version": self.azure_api_version_embeddings,
            "azure_endpoint": self.azure_openai_endpoint,
            "deployment": self.azure_openai_embedding_deployment
        }
        
    def get_search_config(self) -> Dict[str, str]:
        """
        Get search API configuration
        
        Returns:
            Dict[str, str]: Dictionary of configuration values
        """
        return {
            "google_api_key": self.google_api_key,
            "google_cse_id": self.google_cse_id,
            "tavily_api_key": self.tavily_api_key
        }
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """
        Get Qdrant configuration
        
        Returns:
            Dict[str, Any]: Dictionary of configuration values
        """
        return {
            "url": self.qdrant_url,
            "port": self.qdrant_port
        }


# Singleton instance
config = Config()


# Example usage
if __name__ == "__main__":
    print("Azure OpenAI configuration:")
    print(config.get_azure_openai_config())
    
    print("\nAzure Embeddings configuration:")
    print(config.get_azure_embeddings_config())
    
    print("\nSearch API configuration:")
    print(config.get_search_config())
    
    print("\nQdrant configuration:")
    print(config.get_qdrant_config())
    
    print("\nDirectory paths:")
    print(f"Project root: {config.project_root}")
    print(f"Logs directory: {config.logs_dir}")
    print(f"Scraped data directory: {config.scraped_data_dir}")
    print(f"Processed data directory: {config.processed_data_dir}")
