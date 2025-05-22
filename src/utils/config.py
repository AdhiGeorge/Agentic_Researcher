"""
Configuration manager for Agentic Researcher
Loads environment variables from .env file and settings from config.yaml
"""
import os
import yaml
import dotenv
from pathlib import Path
from typing import Dict, Any, Optional, List

class Config:
    """
    Configuration manager for Agentic Researcher
    Provides access to configuration settings with fallbacks to environment variables
    """
    
    def __init__(self):
        """Initialize the configuration manager"""
        # Load environment variables from .env file
        dotenv.load_dotenv()
        
        # Load settings from config.yaml
        self.project_root = Path(__file__).parent.parent.parent.absolute()
        self.config_file = self.project_root / 'config.yaml'
        self.yaml_config = self._load_yaml_config()
        
        # Azure OpenAI API configuration - load from .env for security
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        
        # Get API versions from config or fallback to defaults
        self.azure_api_version_chat = self._get_config_value(
            ['api_keys', 'AZURE_API_VERSION'], 
            os.getenv("AZURE_API_VERSION_CHAT", "2023-07-01-preview")
        )
        self.azure_api_version_embeddings = self._get_config_value(
            ['api_keys', 'AZURE_API_VERSION'], 
            os.getenv("AZURE_API_VERSION_EMBEDDINGS", "2023-05-15")
        )
        self.azure_openai_deployment = self._get_config_value(
            ['api_keys', 'AZURE_OPENAI_DEPLOYMENT'],
            os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        )
        self.azure_openai_embedding_deployment = self._get_config_value(
            ['api_keys', 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT'],
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        )
        
        # Search API configuration - load from .env for security
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID", "")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        
        # Database configuration
        self.qdrant_url = self._get_config_value(
            ['database', 'QDRANT_HOST'],
            os.getenv("QDRANT_URL", "localhost")
        )
        self.qdrant_port = int(self._get_config_value(
            ['database', 'QDRANT_PORT'],
            os.getenv("QDRANT_PORT", "6333")
        ))
        
        # Directory paths - load from config.yaml
        self.logs_dir = self._get_config_value(
            ['paths', 'LOGS_DIR'],
            os.getenv("LOGS_DIR", str(self.project_root / "logs"))
        )
        self.scraped_data_dir = self._get_config_value(
            ['paths', 'SCRAPED_DATA_DIR'],
            os.getenv("SCRAPED_DATA_DIR", str(self.project_root / "scraped_data"))
        )
        self.processed_data_dir = self._get_config_value(
            ['paths', 'PROCESSED_DATA_DIR'],
            os.getenv("PROCESSED_DATA_DIR", str(self.project_root / "processed_data"))
        )
        self.web_links_dir = self._get_config_value(
            ['paths', 'WEB_LINKS_DIR'],
            os.getenv("WEB_LINKS_DIR", str(self.project_root / "web_links"))
        )
        
        # Create directories if they don't exist
        Path(self.logs_dir).mkdir(exist_ok=True)
        Path(self.scraped_data_dir).mkdir(exist_ok=True)
        Path(self.processed_data_dir).mkdir(exist_ok=True)
        Path(self.web_links_dir).mkdir(exist_ok=True)
        
        # Scraping configuration
        self.scraping_headless = self._get_config_value(
            ['scraping', 'HEADLESS'],
            os.getenv("SCRAPING_HEADLESS", "True")
        )
        if isinstance(self.scraping_headless, str):
            self.scraping_headless = self.scraping_headless.lower() in ("true", "1", "t")
            
        self.scraping_timeout = int(self._get_config_value(
            ['scraping', 'TIMEOUT'],
            os.getenv("SCRAPING_TIMEOUT", "30000")
        ))
        
        # Get user agent rotation setting
        self.scraping_user_agent_rotation = self._get_config_value(
            ['scraping', 'USER_AGENT_ROTATION'],
            os.getenv("SCRAPING_USER_AGENT_ROTATION", "True")
        )
        if isinstance(self.scraping_user_agent_rotation, str):
            self.scraping_user_agent_rotation = self.scraping_user_agent_rotation.lower() in ("true", "1", "t")
            
        # Get robots.txt respect setting    
        self.scraping_respect_robots = self._get_config_value(
            ['scraping', 'RESPECT_ROBOTS_TXT'],
            os.getenv("SCRAPING_RESPECT_ROBOTS", "True")
        )
        if isinstance(self.scraping_respect_robots, str):
            self.scraping_respect_robots = self.scraping_respect_robots.lower() in ("true", "1", "t")
        
        # Get retry settings
        self.scraping_retry_attempts = int(self._get_config_value(
            ['scraping', 'RETRY_ATTEMPTS'],
            os.getenv("SCRAPING_RETRY_ATTEMPTS", "3")
        ))
        self.scraping_retry_delay = int(self._get_config_value(
            ['scraping', 'RETRY_DELAY'],
            os.getenv("SCRAPING_RETRY_DELAY", "2")
        ))
        
        # Get user agents from config or use defaults
        default_user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]
        
        self.scraping_user_agents = self._get_config_value(
            ['scraping', 'USER_AGENTS'],
            default_user_agents
        )
    
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
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                print(f"Warning: Config file not found at {self.config_file}")
                return {}
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}
    
    def _get_config_value(self, keys: List[str], default: Any = None) -> Any:
        """Get a value from the config.yaml file
        
        Args:
            keys: The keys to access the value, forming a path in the config
            default: The default value to return if the key is not found
            
        Returns:
            The value from config or the default
        """
        config = self.yaml_config
        
        for key in keys:
            if not config or key not in config:
                return default
            config = config[key]
            
        return config if config is not None else default
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using a simplified interface
        
        This method provides a dictionary-like interface to configuration values
        for compatibility with code expecting a dict-like object.
        
        Args:
            key: The key to access the value
            default: The default value to return if the key is not found
            
        Returns:
            The value from config or the default
        """
        # First check if it's a direct attribute of the config object
        if hasattr(self, key):
            return getattr(self, key)
            
        # Then check in the yaml_config
        if self.yaml_config and key in self.yaml_config:
            return self.yaml_config[key]
            
        # Return the default if not found
        return default
    
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
