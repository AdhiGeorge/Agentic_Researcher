
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class LLMConfig:
    provider: str = "azure_openai"  # Explicitly set to Azure OpenAI
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: str = "2023-05-15"  # Azure OpenAI API version
    deployment_name: str = "gpt-4o"  # Azure OpenAI deployment name
    temperature: float = 0.7
    max_tokens: int = 1500
    fallback_providers: List[str] = field(default_factory=lambda: ["mistral", "local"])

@dataclass
class ScraperConfig:
    user_agent_rotation: bool = True
    proxy_rotation: bool = False
    respect_robots_txt: bool = True
    headless_evasion: bool = True
    delay_min: float = 0.5
    delay_max: float = 2.0
    max_retries: int = 3
    timeout: int = 30
    use_playwright: bool = True

@dataclass
class VectorStoreConfig:
    provider: str = "qdrant"  # Explicitly set to Qdrant
    collection_name: str = "research_data"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    url: Optional[str] = None
    api_key: Optional[str] = None
    location: str = ":memory:"  # Local in-memory DB by default

@dataclass
class SandboxConfig:
    enabled: bool = True
    provider: str = "firejail"
    timeout: int = 60
    max_memory_mb: int = 1024
    allow_network: bool = False

@dataclass
class SwarmConfig:
    enabled: bool = True
    protocol: str = "lightweight"  # Lightweight swarm protocol
    communication_method: str = "direct"
    max_parallel_agents: int = 3
    coordination_interval: float = 0.5  # seconds
    agent_timeout: int = 120  # seconds

@dataclass
class SystemConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    
    memory_path: str = "./memory"
    output_path: str = "./outputs"
    tools_path: str = "./tools"
    
    def __post_init__(self):
        # Load from environment variables if available
        if os.getenv("AZURE_OPENAI_API_KEY"):
            self.llm.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.llm.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            
        if os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
            self.llm.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            
        if os.getenv("QDRANT_URL"):
            self.vector_store.url = os.getenv("QDRANT_URL")
            
        if os.getenv("QDRANT_API_KEY"):
            self.vector_store.api_key = os.getenv("QDRANT_API_KEY")
        
        # Create necessary directories
        os.makedirs(self.memory_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.tools_path, "custom"), exist_ok=True)
        os.makedirs(os.path.join(self.tools_path, "tests"), exist_ok=True)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "fallback_providers": self.llm.fallback_providers
            },
            "scraper": {
                "user_agent_rotation": self.scraper.user_agent_rotation,
                "proxy_rotation": self.scraper.proxy_rotation,
                "respect_robots_txt": self.scraper.respect_robots_txt,
                "headless_evasion": self.scraper.headless_evasion,
                "use_playwright": self.scraper.use_playwright
            },
            "vector_store": {
                "provider": self.vector_store.provider,
                "embedding_model": self.vector_store.embedding_model
            },
            "sandbox": {
                "enabled": self.sandbox.enabled,
                "provider": self.sandbox.provider
            },
            "swarm": {
                "enabled": self.swarm.enabled,
                "protocol": self.swarm.protocol,
                "max_parallel_agents": self.swarm.max_parallel_agents
            }
        }
