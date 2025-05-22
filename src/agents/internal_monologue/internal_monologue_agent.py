"""Internal Monologue Agent for Agentic Researcher

This module implements the Internal Monologue agent that provides human-like
reasoning and conversational flow for the agent system.
"""
import json
import logging
from typing import Dict, Any, Optional

import os
import sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.utils.config import Config as config
from src.db.sqlite_manager import SQLiteManager
from src.utils.openai_client import AzureOpenAIClient
from src.agents.base_agent import BaseAgent

# Configure logging
logger = logging.getLogger(__name__)

class InternalMonologueAgent(BaseAgent):
    """Internal Monologue Agent for providing human-like reasoning and thought processes"""
    
    def __init__(self, config_obj=None, **kwargs):
        """Initialize the Internal Monologue agent
        
        Args:
            config_obj: Optional configuration object
            **kwargs: Additional arguments
        """
        super().__init__(
            name="internal_monologue", 
            description="Agent that generates human-like internal thought processes and reasoning",
            **kwargs
        )
        logger.info("Initializing Internal Monologue Agent")
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAIClient()
        self.db = kwargs.get("sqlite_db") or SQLiteManager()
        self.config = config_obj or config
        
        # Import prompt loader with dual import pattern
        try:
            # When imported as a module
            from src.utils.prompt_loader import PromptLoader
        except ModuleNotFoundError:
            # When run directly as a script
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from utils.prompt_loader import PromptLoader
            
        # Load prompt from centralized prompts.yaml
        prompt_loader = PromptLoader()
        self.system_prompt = prompt_loader.get_prompt('internal_monologue.system')
        
        # Fall back to default if prompt not found
        if not self.system_prompt:
            self.logger.warning("Failed to load internal monologue system prompt from prompts.yaml, using default")
            self.system_prompt = (
                "You are an internal monologue system for an AI researcher. "
                "Your job is to express the AI's internal thoughts and reasoning process "
                "while it works on research tasks. Make the reasoning process feel natural, "
                "showing how the AI is thinking about the problem, considering different angles, "
                "and making connections between pieces of information. Keep responses concise "
                "but insightful."
            )
    
    def generate_monologue(self, context: str, state: Dict[str, Any] = None) -> str:
        """Generate human-like internal monologue based on context
        
        Args:
            context: The context to generate monologue for
            state: Optional state information
            
        Returns:
            str: Internal monologue text
        """
        try:
            # Prepare the prompt
            user_prompt = f"Generate an internal monologue about this context: {context}"
            
            # Call OpenAI
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.openai_client.generate_completion(
                messages=messages,
                model=self.config.azure_openai_deployment,
                temperature=0.7,
                max_tokens=300
            )
            
            # Extract and return monologue
            monologue = response.strip()
            logger.info(f"Generated internal monologue: {monologue[:50]}...")
            
            # Store in state if provided
            if state and isinstance(state, dict):
                state["monologue"] = monologue
                
            return monologue
            
        except Exception as e:
            logger.error(f"Error generating internal monologue: {str(e)}")
            return "Thinking about how to approach this research task..."
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and generate internal monologue
        
        Args:
            input_data: Input data containing context
            
        Returns:
            Dict: Result with monologue
        """
        context = input_data.get("context", "")
        state = input_data.get("state", {})
        
        monologue = self.generate_monologue(context, state)
        
        return {
            "monologue": monologue,
            "status": "success"
        }


# Example usage
if __name__ == "__main__":
    print("\n===== InternalMonologueAgent Example =====\n")
    print("Initializing InternalMonologueAgent...")
    
    try:
        # Initialize the agent with direct API access
        monologue_agent = InternalMonologueAgent()
        
        # Example research contexts to generate monologues for
        research_contexts = [
            {
                "title": "Summarizing a research paper",
                "context": "I'm reviewing a paper on transformer-based architectures for natural language processing. "
                        "The paper introduces a new attention mechanism that claims to reduce computational complexity "
                        "while maintaining accuracy. I need to verify these claims and understand how this might apply "
                        "to our current research projects."
            },
            {
                "title": "Planning a data analysis approach",
                "context": "We have a large dataset of user interactions with our recommendation system. "
                        "We need to analyze patterns of engagement and identify factors that lead to higher "
                        "retention rates. There seem to be several outliers in the data that might skew results."
            },
            {
                "title": "Connecting interdisciplinary research",
                "context": "Our project is combining findings from neuroscience, machine learning, and linguistics "
                        "to create better natural language understanding systems. I need to identify the key "
                        "connections between these fields that could lead to innovative approaches."
            }
        ]
        
        print(f"\nLoaded {len(research_contexts)} example research contexts to process")
        print("Using Azure OpenAI deployment: ", config.azure_openai_deployment)
        
        # Process each context and generate monologues
        for i, example in enumerate(research_contexts, 1):
            print(f"\n----- Example {i}: {example['title']} -----")
            print("\nCONTEXT:")
            print(example['context'])
            
            print("\nGenerating internal monologue...")
            
            try:
                # Use the agent to process the input
                result = monologue_agent.process({
                    "context": example['context'],
                    "state": {"example_id": i, "source": "research_context"}
                })
                
                # Display the generated monologue
                print("\nINTERNAL MONOLOGUE:")
                print(result['monologue'])
                
            except Exception as e:
                print(f"\nError processing example {i}: {str(e)}")
                # If API call fails, provide a fallback monologue
                fallback_monologue = (
                    "Let me think about this methodically... This appears to be about " + 
                    example['title'].lower() + ". I should consider multiple angles and analyze " +
                    "the potential implications. The key points to focus on might include connections " +
                    "between concepts and practical applications of these insights."
                )
                print("\nFALLBACK MONOLOGUE (due to API error):")
                print(fallback_monologue)
            
            print("\n" + "-"*50)
        
        # Show how to integrate with a research workflow
        print("\nIntegrating with Research Workflow Example:")
        research_flow = """
        1. User submits query: "How do transformers handle long-range dependencies?"
        2. Research Agent gathers relevant papers and information
        3. InternalMonologueAgent: <generates reasoning process about research findings>
        4. Answer Agent synthesizes final response based on research and reasoning
        5. User receives comprehensive answer with transparent thought process
        """
        print(research_flow)
        
        # Additional use cases
        print("\nReal-world use cases for InternalMonologueAgent:")
        print("1. Adding human-like reasoning to AI research assistants")
        print("2. Enhancing explainability by showing AI thought processes")
        print("3. Integrating with other agents to create more natural interactions")
        print("4. Creating educational content that demonstrates expert reasoning")
        print("5. Providing step-by-step problem-solving approaches in complex domains")
        
    except Exception as e:
        print(f"\nError during example execution: {str(e)}")
        print("\nThis example requires proper configuration of the Azure OpenAI client.")
        print("Please ensure your .env file contains the following variables:")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_DEPLOYMENT")
    
    print("\n===== End of Example =====")
