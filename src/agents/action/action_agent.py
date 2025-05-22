"""Action Agent for Agentic Researcher

This module implements the Action agent that determines and executes
the most appropriate action based on research results and user queries.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Use imports that work for both direct execution and when imported as a module
from src.agents.base_agent import BaseAgent
from src.utils.config import Config as config

class ActionAgent(BaseAgent):
    """Action agent that determines and executes appropriate actions.
    
    The Action agent takes the results of research and determines what actions
    to take, such as answering questions, writing code, or providing analysis.
    
    Attributes:
        config (ConfigLoader): Configuration loader
        action_types (List[str]): Available action types
    """
    
    def __init__(self):
        """Initialize the ActionAgent."""
        super().__init__(name="Action", description="Determines and executes appropriate actions")
        # Using the global config object imported at the top of the file
        self.config = config
        
        # Define available action types
        self.action_types = ["answer", "code", "analyze", "explain", "summarize"]
        
        self.logger.info("ActionAgent initialized")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input and determine the appropriate action.
        
        Args:
            input_data (Dict[str, Any]): Input data containing query and formatted content
            
        Returns:
            Dict[str, Any]: The result of the action
        """
        query = input_data.get("query", "")
        formatted_content = input_data.get("formatted_content", {})
        conversation_history = input_data.get("conversation_history", [])
        
        if not query:
            self.logger.error("No query provided to ActionAgent")
            raise ValueError("No query provided to ActionAgent")
        
        if not formatted_content:
            self.logger.error("No formatted content provided to ActionAgent")
            raise ValueError("No formatted content provided to ActionAgent")
        
        # Determine the best action type
        action_type = self._determine_action_type(query, formatted_content)
        self.log_activity(f"Determined action type: {action_type}")
        
        # Execute the action
        action_result = self.execute_action({
            "action_type": action_type,
            "query": query,
            "formatted_content": formatted_content,
            "conversation_history": conversation_history
        })
        
        # Update agent state
        self.update_state({
            "last_query": query,
            "last_action_type": action_type,
            "last_action_time": datetime.now().isoformat()
        })
        
        return {
            "query": query,
            "action_type": action_type,
            "action_result": action_result,
            "processed_at": datetime.now().isoformat()
        }
    
    def _determine_action_type(self, query: str, formatted_content: Dict[str, Any]) -> str:
        """Determine the most appropriate action type based on the query and content.
        
        Args:
            query (str): The user's query
            formatted_content (Dict[str, Any]): The formatted research content
            
        Returns:
            str: The determined action type
        """
        self.log_activity(f"Determining action type for query: {query}")
        
        # Check for code-related keywords
        code_keywords = ["code", "script", "program", "function", "class", "implement", "python", "algorithm"]
        if any(keyword in query.lower() for keyword in code_keywords):
            return "code"
        
        # Check for analysis keywords
        analysis_keywords = ["analyze", "analysis", "compare", "evaluate", "assess", "study"]
        if any(keyword in query.lower() for keyword in analysis_keywords):
            return "analyze"
        
        # Check for explanation keywords
        explanation_keywords = ["explain", "how", "why", "what is", "definition", "meaning"]
        if any(keyword in query.lower() for keyword in explanation_keywords):
            return "explain"
        
        # Check for summary keywords
        summary_keywords = ["summarize", "summary", "brief", "overview", "gist"]
        if any(keyword in query.lower() for keyword in summary_keywords):
            return "summarize"
        
        # Default to answer for general questions
        return "answer"
    
    def execute_action(self, action_data: Dict[str, Any]) -> str:
        """Execute the specified action based on action type.
        
        Args:
            action_data (Dict[str, Any]): Data containing action type and required information
            
        Returns:
            str: The result of the action
        """
        action_type = action_data.get("action_type", "answer")
        query = action_data.get("query", "")
        formatted_content = action_data.get("formatted_content", {})
        conversation_history = action_data.get("conversation_history", [])
        
        self.log_activity(f"Executing action: {action_type}")
        
        # Extract structured content from formatted content
        structured_content = ""
        if isinstance(formatted_content, dict):
            structured_content = formatted_content.get("structured_content", "")
        else:
            structured_content = str(formatted_content)
        
        # Use Azure OpenAI to execute the action
        return self._execute_with_llm(action_type, query, structured_content, conversation_history)
    
    def _execute_with_llm(self, action_type: str, query: str, content: str, conversation_history: List[Dict[str, str]]) -> str:
        """Use Azure OpenAI to execute the specified action.
        
        Args:
            action_type (str): The type of action to execute
            query (str): The user's query
            content (str): The content to use for the action
            conversation_history (List[Dict[str, str]]): Previous conversation history
            
        Returns:
            str: The result of the action
        """
        self.log_activity(f"Using LLM to execute {action_type} action")
        
        try:
            # Get Azure OpenAI credentials from config
            api_key = self.config.azure_openai_api_key
            api_version = self.config.azure_api_version_chat
            azure_endpoint = self.config.azure_openai_endpoint
            deployment_name = self.config.azure_openai_deployment
            
            # Set up Azure OpenAI client
            import openai
            client = openai.AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
            
            # Create the system prompt based on action type
            system_prompt = self._get_system_prompt(action_type)
            
            # Create messages array with conversation history if available
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history if available
            if conversation_history:
                for message in conversation_history:
                    messages.append(message)
            
            # Add the current query and content
            user_prompt = f"Query: {query}\n\nResearch Content:\n{content}"
            messages.append({"role": "user", "content": user_prompt})
            
            # Make API call to Azure OpenAI
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0.5,
                max_tokens=4000
            )
            
            result = response.choices[0].message.content.strip()
            self.log_activity(f"Successfully executed {action_type} action")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing action with LLM: {str(e)}")
            # Return a fallback message if LLM fails
            return self._get_fallback_response(action_type, query)
    
    def _get_system_prompt(self, action_type: str) -> str:
        """Get the appropriate system prompt for the specified action type.
        
        Args:
            action_type (str): The type of action to execute
            
        Returns:
            str: The system prompt
        """
        if action_type == "code":
            return """
            You are an expert coding assistant. Your task is to:
            1. Analyze the research content and the user's query
            2. Generate well-documented, functional code that addresses the query
            3. Include explanations and comments in the code
            4. Ensure the code is efficient, robust, and follows best practices
            5. Format the code properly using markdown code blocks
            6. Include example usage where appropriate
            7. Explain the reasoning behind implementation choices
            
            Focus on providing practical, working solutions based on the research.
            """
        
        elif action_type == "analyze":
            return """
            You are an expert analysis assistant. Your task is to:
            1. Analyze the research content in depth
            2. Identify key patterns, insights, and relationships
            3. Provide a structured analysis with clear sections
            4. Support your analysis with evidence from the research
            5. Consider alternative interpretations where appropriate
            6. Format your analysis using markdown for readability
            7. Conclude with the most significant insights
            
            Focus on providing a comprehensive, factual analysis based on the research.
            """
        
        elif action_type == "explain":
            return """
            You are an expert explanation assistant. Your task is to:
            1. Provide a clear, accessible explanation of the topic
            2. Break down complex concepts into understandable parts
            3. Use analogies and examples where helpful
            4. Structure your explanation logically
            5. Define technical terms when first used
            6. Format your explanation using markdown for readability
            7. Focus on accuracy while maintaining clarity
            
            Focus on making the subject matter understandable to someone unfamiliar with it.
            """
        
        elif action_type == "summarize":
            return """
            You are an expert summarization assistant. Your task is to:
            1. Distill the key points from the research content
            2. Create a concise yet comprehensive summary
            3. Maintain the essential meaning and context
            4. Organize the summary with clear sections
            5. Highlight the most important findings or concepts
            6. Format your summary using markdown for readability
            7. Ensure accuracy while eliminating redundancy
            
            Focus on creating a summary that captures the essence of the research in minimal space.
            """
        
        else:  # Default "answer" action type
            return """
            You are an expert research assistant. Your task is to:
            1. Answer the user's query based on the provided research content
            2. Provide comprehensive, accurate information
            3. Structure your answer clearly and logically
            4. Support your statements with evidence from the research
            5. Acknowledge limitations or uncertainties where they exist
            6. Format your answer using markdown for readability
            7. Be helpful, clear, and direct
            
            Focus on addressing the user's query precisely and thoroughly based on the research.
            """
    
    def _get_fallback_response(self, action_type: str, query: str) -> str:
        """Get a fallback response when LLM processing fails.
        
        Args:
            action_type (str): The type of action that was attempted
            query (str): The user's query
            
        Returns:
            str: A fallback response
        """
        return f"""
        # Research Response
        
        I've analyzed the available research on your query:  
        **{query}**
        
        Unfortunately, I encountered an issue processing the research data with the requested {action_type} action. 
        
        Here's what I can tell you based on the available information:
        
        - The research materials contain relevant information about your query
        - The {action_type} action was attempted but couldn't be completed due to a technical limitation
        - You can try rephrasing your query or selecting a different action type
        
        I apologize for the inconvenience. Would you like to try a different approach to address your research needs?
        """


if __name__ == "__main__":
    # Example usage
    action_agent = ActionAgent()
    
    try:
        # Sample query and formatted content for testing
        query = "What is the volatility index and how is it calculated?"
        formatted_content = {
            "structured_content": """# Volatility Index (VIX)

The Volatility Index, or VIX, is a real-time market index that represents the market's expectation of 30-day forward-looking volatility. Derived from the price inputs of the S&P 500 index options, it provides a measure of market risk and investor sentiment.

## Calculation
The VIX is calculated using a complex formula that weighs the prices of out-of-the-money put and call options on the S&P 500 index. The formula essentially measures the market's expectation of volatility implied by S&P 500 stock index option prices.

The general formula is based on the following formula:

VIX = 100 * √(τ * ∑(ΔK_i / K_i² * e^(RT) * Q(K_i)))

Where:
- τ is time to expiration
- K_i is the strike price of the i-th out-of-money option
- ΔK_i is the interval between strike prices
- R is the risk-free interest rate
- Q(K_i) is the midpoint of the bid-ask spread for each option with strike K_i
""",
            "sources": [
                {
                    "url": "https://example.com/volatility-index",
                    "title": "Understanding the Volatility Index"
                }
            ]
        }
        
        # Test the action agent
        result = action_agent.process({
            "query": query,
            "formatted_content": formatted_content
        })
        
        print("\nAction Result:")
        print(f"Query: {result['query']}")
        print(f"Action Type: {result['action_type']}")
        print(f"Processed At: {result['processed_at']}")
        print("\nResult Content:")
        # Handle potential encoding issues with Unicode characters
        try:
            print(result["action_result"])
        except UnicodeEncodeError:
            print("[Content contains special characters that can't be displayed in the console]")
            # Print a simplified version with problematic characters replaced
            print(result["action_result"].encode('ascii', 'replace').decode('ascii'))
        
        # Test code generation
        code_query = "Write Python code to calculate the VIX score based on options data"
        code_result = action_agent.process({
            "query": code_query,
            "formatted_content": formatted_content
        })
        
        print("\nCode Generation Result:")
        print(f"Query: {code_result['query']}")
        print(f"Action Type: {code_result['action_type']}")
        print("\nGenerated Code:")
        try:
            print(code_result["action_result"])
        except UnicodeEncodeError:
            print("[Content contains special characters that can't be displayed in the console]")
            # Print a simplified version with problematic characters replaced
            print(code_result["action_result"].encode('ascii', 'replace').decode('ascii'))
        
    except Exception as e:
        print(f"Error: {str(e)}")
