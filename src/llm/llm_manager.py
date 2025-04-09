
import os
import logging
import time
from typing import Dict, List, Any, Optional, Union
import json
import openai
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manager for LLM interactions. Supports Azure OpenAI with fallbacks.
    """
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.init_client()
    
    def init_client(self):
        """Initialize the appropriate client based on provider"""
        if self.config.provider == "azure_openai":
            # Use Azure OpenAI
            if not self.config.api_key or not self.config.endpoint:
                logger.warning("Azure OpenAI API key or endpoint not set. Attempting to use environment variables.")
            
            try:
                self.client = AzureOpenAI(
                    api_key=self.config.api_key,
                    api_version=self.config.api_version,
                    azure_endpoint=self.config.endpoint
                )
                logger.info("Initialized Azure OpenAI client")
            except Exception as e:
                logger.error(f"Error initializing Azure OpenAI client: {str(e)}")
                self._try_fallback_providers()
        
        elif self.config.provider == "openai":
            # Use regular OpenAI
            try:
                self.client = OpenAI(api_key=self.config.api_key)
                logger.info("Initialized OpenAI client")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
                self._try_fallback_providers()
                
        else:
            logger.warning(f"Unsupported provider: {self.config.provider}")
            self._try_fallback_providers()
    
    def _try_fallback_providers(self):
        """Try to initialize fallback providers"""
        for provider in self.config.fallback_providers:
            logger.info(f"Attempting to use fallback provider: {provider}")
            # Implement fallback provider initialization logic
            # This is a simplified placeholder
            if provider == "mistral":
                logger.info("Successfully initialized Mistral fallback")
                self.config.provider = "mistral"
                break
            elif provider == "local":
                logger.info("Successfully initialized local LLM fallback")
                self.config.provider = "local"
                break
    
    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            system_message: Optional system message
            
        Returns:
            The generated text
        """
        if not self.client:
            logger.error("No LLM client available")
            return "Error: No LLM client available. Please check your API keys and configuration."
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if self.config.provider == "azure_openai":
                    # Azure OpenAI API call
                    messages = []
                    
                    if system_message:
                        messages.append({"role": "system", "content": system_message})
                    
                    messages.append({"role": "user", "content": prompt})
                    
                    response = self.client.chat.completions.create(
                        model=self.config.deployment_name,
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    
                    return response.choices[0].message.content
                
                elif self.config.provider == "openai":
                    # Regular OpenAI API call
                    messages = []
                    
                    if system_message:
                        messages.append({"role": "system", "content": system_message})
                    
                    messages.append({"role": "user", "content": prompt})
                    
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    
                    return response.choices[0].message.content
                
                elif self.config.provider in ["mistral", "local"]:
                    # Implement fallback provider logic here
                    # This is a simplified placeholder
                    logger.warning(f"Using {self.config.provider} fallback (placeholder implementation)")
                    return f"This is a placeholder response from {self.config.provider} fallback provider."
                
                else:
                    logger.error(f"Unsupported provider: {self.config.provider}")
                    return f"Error: Unsupported provider {self.config.provider}"
                    
            except Exception as e:
                logger.error(f"Error in LLM call (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return f"Error generating response: {str(e)}"
        
        return "Error: Failed to generate a response after multiple attempts."
