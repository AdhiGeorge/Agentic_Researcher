"""
Prompt Loader for Agentic Researcher
Manages loading and templating of prompts from YAML file
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional
from jinja2 import Template

from src.utils.config import Config as ConfigLoader

# Configure logging
logger = logging.getLogger(__name__)

class PromptLoader:
    """
    Prompt loader for Agentic Researcher
    Loads prompts from YAML file and provides templating functionality
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Load configuration
        self.config = ConfigLoader()
        
        # Get prompt file path
        self.prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "prompts.yaml"
        )
        
        # Load prompts
        self.prompts = self._load_prompts()
        self._initialized = True
        
        logger.info(f"PromptLoader initialized with prompts from {self.prompt_path}")
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file"""
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                prompts = yaml.safe_load(f)
                logger.info(f"Loaded {len(prompts)} prompt categories")
                return prompts
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {self.prompt_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            return {}
    
    def get_system_prompt(self, agent_type: str) -> str:
        """
        Get system prompt for an agent
        
        Args:
            agent_type: Type of agent (planner, researcher, etc.)
            
        Returns:
            str: System prompt for the agent
        """
        try:
            return self.prompts.get(agent_type, {}).get("system", "")
        except Exception as e:
            logger.error(f"Error getting system prompt for {agent_type}: {str(e)}")
            return ""
    
    def get_task_prompt(self, agent_type: str, variables: Dict[str, Any] = None) -> str:
        """
        Get task prompt with variables substituted
        
        Args:
            agent_type: Type of agent (planner, researcher, etc.)
            variables: Variables to substitute in the prompt
            
        Returns:
            str: Task prompt with variables substituted
        """
        if variables is None:
            variables = {}
            
        try:
            prompt_template = self.prompts.get(agent_type, {}).get("task_template", "")
            template = Template(prompt_template)
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Error rendering task prompt for {agent_type}: {str(e)}")
            return prompt_template  # Return unrendered template on error
    
    def reload_prompts(self) -> bool:
        """
        Reload prompts from YAML file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.prompts = self._load_prompts()
            return True
        except Exception as e:
            logger.error(f"Error reloading prompts: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create prompt loader
    prompt_loader = PromptLoader()
    
    # Test planner prompts
    print("\nPlanner System Prompt:")
    print(prompt_loader.get_system_prompt("planner"))
    
    print("\nPlanner Task Prompt:")
    planner_variables = {
        "query": "What is the volatility index and how is it calculated?"
    }
    print(prompt_loader.get_task_prompt("planner", planner_variables))
    
    # Test researcher prompts
    print("\nResearcher System Prompt:")
    print(prompt_loader.get_system_prompt("researcher"))
