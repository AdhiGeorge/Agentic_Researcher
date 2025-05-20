"""Base agent class for Agentic Researcher

This module provides a base agent class that all specialized agents will inherit from.
It handles common functionality such as LLM interactions, context management, and state tracking.
"""

from abc import ABC, abstractmethod
import os
import sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
import json
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BaseAgent(ABC):
    """Base agent class that all specialized agents will inherit from.
    
    Attributes:
        name (str): Name of the agent
        description (str): Description of the agent's purpose
        logger (logging.Logger): Logger instance for this agent
        state (Dict): Current state of the agent
    """
    
    def __init__(self, name: str, description: str):
        """Initialize the base agent with a name and description.
        
        Args:
            name (str): Name of the agent
            description (str): Description of the agent's purpose
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"agent.{name}")
        self.state = {}
        self.logger.info(f"Initialized {name} agent")
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data and return results.
        
        This method must be implemented by all derived agent classes.
        
        Args:
            input_data: The input data to process
            
        Returns:
            Dict[str, Any]: The results of processing
        """
        pass
    
    def update_state(self, state_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update the agent's internal state.
        
        Args:
            state_updates (Dict[str, Any]): Updates to apply to the state
            
        Returns:
            Dict[str, Any]: The updated state
        """
        self.state.update(state_updates)
        self.logger.debug(f"Updated state: {state_updates}")
        return self.state
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent.
        
        Returns:
            Dict[str, Any]: The current state
        """
        return self.state
    
    def log_activity(self, activity: str, level: str = "info") -> None:
        """Log an agent activity.
        
        Args:
            activity (str): Description of the activity
            level (str): Logging level (debug, info, warning, error, critical)
        """
        log_method = getattr(self.logger, level.lower())
        log_method(f"{self.name}: {activity}")
    
    def __str__(self) -> str:
        """String representation of the agent.
        
        Returns:
            str: String representation
        """
        return f"{self.name} - {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the agent
        """
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state
        }


if __name__ == "__main__":
    # Example usage of BaseAgent
    class ExampleAgent(BaseAgent):
        def process(self, input_data):
            self.log_activity(f"Processing input: {input_data}")
            result = {"processed": f"Processed {input_data} with {self.name}"}
            self.update_state({"last_processed": input_data})
            return result
    
    # Create an example agent
    example = ExampleAgent("ExampleAgent", "An example agent implementation")
    
    # Process some data
    result = example.process("test data")
    print(f"Result: {result}")
    print(f"Agent state: {example.get_state()}")
