"""
Proxy Handler utility for Agentic Researcher
Manages rotating user-agents and headers for web scraping
"""
import random
import time
from typing import Dict, Optional
from fake_useragent import UserAgent
from src.utils.config import config

class ProxyHandler:
    """
    Proxy handler for web scraping
    Handles rotating user agents and request headers
    """
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProxyHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Use global config
        self.config = config
        
        # Initialize user agent generator
        self.user_agent_rotation = self.config.scraping_user_agent_rotation
        self.ua_generator = UserAgent()
        
        # Default headers
        self.default_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
        
        # Common desktop browsers for diversity
        self.browser_types = [
            "chrome", "firefox", "edge", "safari"
        ]
        
        # Last used timestamp to prevent too frequent rotation
        self.last_rotation_time = 0
        self.min_rotation_interval = 5  # seconds
        
        self._initialized = True
    
    def get_random_user_agent(self, browser_type: Optional[str] = None) -> str:
        """
        Get a random user agent string
        
        Args:
            browser_type: Optional browser type to generate agent for
            
        Returns:
            str: User agent string
        """
        if browser_type and browser_type in self.browser_types:
            return self.ua_generator[browser_type]
        else:
            # Random browser type
            browser = random.choice(self.browser_types)
            return self.ua_generator[browser]
    
    def get_headers(self, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get headers for HTTP requests with optional user agent rotation
        
        Args:
            custom_headers: Optional custom headers to include
            
        Returns:
            Dict[str, str]: Headers dict for requests
        """
        headers = self.default_headers.copy()
        
        # Rotate user agent if enabled
        if self.user_agent_rotation:
            current_time = time.time()
            if current_time - self.last_rotation_time >= self.min_rotation_interval:
                headers["User-Agent"] = self.get_random_user_agent()
                self.last_rotation_time = current_time
        
        # Add custom headers if provided
        if custom_headers:
            headers.update(custom_headers)
            
        return headers
    
    def get_retry_config(self) -> Dict[str, any]:
        """
        Get retry configuration for requests
        
        Returns:
            Dict: Retry configuration
        """
        return {
            "attempts": self.config.scraping_retry_attempts,
            "delay": self.config.scraping_retry_delay
        }
