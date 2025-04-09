
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class LLMConfig:
    provider: str = "azure_openai"
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1500

@dataclass
class ScraperConfig:
    user_agent_rotation: bool = True
    proxy_rotation: bool = False
    respect_robots_txt: bool = True
    delay_min: float = 0.5
    delay_max: float = 2.0
    max_retries: int = 3
    timeout: int = 30

@dataclass
class VectorStoreConfig:
    provider: str = "qdrant"
    collection_name: str = "research_data"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384

@dataclass
class SandboxConfig:
    enabled: bool = True
    provider: str = "firejail"
    timeout: int = 60
    max_memory_mb: int = 1024
    allow_network: bool = False

@dataclass
class SystemConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    
    memory_path: str = "./memory"
    output_path: str = "./outputs"
    tools_path: str = "./tools"
    
    def __post_init__(self):
        # Load from environment variables if available
        if os.getenv("OPENAI_API_KEY"):
            self.llm.api_key = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("OPENAI_ENDPOINT"):
            self.llm.endpoint = os.getenv("OPENAI_ENDPOINT")
        
        # Create necessary directories
        os.makedirs(self.memory_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.tools_path, exist_ok=True)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens
            },
            "scraper": {
                "user_agent_rotation": self.scraper.user_agent_rotation,
                "proxy_rotation": self.scraper.proxy_rotation,
                "respect_robots_txt": self.scraper.respect_robots_txt
            },
            "vector_store": {
                "provider": self.vector_store.provider,
                "embedding_model": self.vector_store.embedding_model
            },
            "sandbox": {
                "enabled": self.sandbox.enabled,
                "provider": self.sandbox.provider
            }
        }
