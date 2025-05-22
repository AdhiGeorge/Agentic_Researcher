"""Decision Agent for Agentic Researcher

This agent is responsible for analyzing options and making decisions based on the research data.
It helps determine the next best action based on the user query and research results.
"""

import json
import os
import sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import Dict, List, Any, Optional, Tuple

from src.db.sqlite_manager import SQLiteManager
from src.db.qdrant_manager import QdrantManager
from src.utils.openai_client import AzureOpenAIClient

class DecisionAgent:
    """
    Decision Agent that analyzes options and makes decisions
    
    This agent evaluates multiple paths or solutions and decides on the optimal approach
    based on the research findings and user requirements.
    """
    
    def __init__(self):
        """Initialize the DecisionAgent"""
        self.name = "decision"
        
        # Initialize database connections
        self.sqlite_manager = SQLiteManager()
        self.vector_db = QdrantManager()
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAIClient()
    
    def evaluate_options(self, query: str, options: List[Dict[str, Any]], 
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate multiple options and recommend the best approach
        
        Args:
            query: Original user query
            options: List of possible options or approaches
            context: Additional context information
            
        Returns:
            Dict: Evaluation results and recommendation
        """
        # Build the evaluation prompt
        options_text = ""
        for idx, option in enumerate(options, 1):
            name = option.get("name", f"Option {idx}")
            description = option.get("description", "No description provided")
            pros = option.get("pros", [])
            cons = option.get("cons", [])
            
            pros_text = "\n".join([f"  - {pro}" for pro in pros])
            cons_text = "\n".join([f"  - {con}" for con in cons])
            
            options_text += f"Option {idx}: {name}\n"
            options_text += f"Description: {description}\n"
            options_text += f"Pros:\n{pros_text}\n"
            options_text += f"Cons:\n{cons_text}\n\n"
        
        prompt = f"""You are an expert decision maker tasked with evaluating options and making decisions.

USER QUERY: {query}

OPTIONS TO EVALUATE:
{options_text}

Additional context:
{json.dumps(context) if context else 'No additional context provided'}

Your task is to analyze these options and provide a decision with clear reasoning.

Format your response as a JSON object with the following structure:
{{
    "recommendation": "The option name or number you recommend",
    "reasoning": "Detailed explanation of your reasoning",
    "strengths": ["List of key strengths of your chosen option"],
    "limitations": ["List of limitations or concerns about your chosen option"],
    "alternatives": ["Alternative options that could also work"],
    "next_steps": ["Concrete next steps to implement this decision"]
}}
"""

        # Call Azure OpenAI API
        response = self.openai_client.generate_completion(
            messages=[
                {"role": "system", "content": "You are an expert decision maker."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse and return the response
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except:
                    result = {
                        "recommendation": "Unable to determine",
                        "reasoning": "Failed to parse API response",
                        "error": "Response format error",
                        "raw_response": response
                    }
            else:
                result = {
                    "recommendation": "Unable to determine",
                    "reasoning": "Failed to parse API response",
                    "error": "Response format error",
                    "raw_response": response
                }
        
        return result
    
    def make_decision(self, query: str, research_results: Dict[str, Any], 
                    options: Optional[List[Dict[str, Any]]] = None, 
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a decision based on research results
        
        Args:
            query: Original user query
            research_results: Results from the research agent
            options: Optional predefined options, if not provided, will be inferred
            context: Additional context information
            
        Returns:
            Dict: Decision result with explanation and next steps
        """
        # If options not provided, generate them from research results
        if not options:
            options = self._generate_options(query, research_results)
        
        # Evaluate options and make decision
        decision = self.evaluate_options(query, options, context)
        
        # Save decision to database
        if context and "project_id" in context:
            self.sqlite_manager.save_agent_state(
                project_id=context["project_id"],
                agent_type="decision",
                state_data={
                    "decision": decision,
                    "options": options
                }
            )
        
        return decision
    
    def _generate_options(self, query: str, research_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate options based on research results
        
        Args:
            query: Original user query
            research_results: Results from the research agent
            
        Returns:
            List[Dict]: List of generated options
        """
        # Extract relevant chunks from research results
        chunks = research_results.get("web_content", [])
        
        # If no chunks in web_content, try pdf_content
        if not chunks:
            chunks = research_results.get("pdf_content", [])
        
        # Combine content from chunks
        content = ""
        for chunk in chunks:
            if isinstance(chunk, dict) and "content" in chunk:
                content += chunk["content"] + "\n\n"
            elif isinstance(chunk, str):
                content += chunk + "\n\n"
        
        # Build prompt for option generation
        prompt = f"""You are an expert analytical thinker. Based on the following research results, generate 3-5 possible options or approaches to address the user's query.

USER QUERY: {query}

RESEARCH RESULTS:
{content[:4000]}  # Truncate to avoid token limits

Generate a list of options in JSON format:
{{
    "options": [
        {{
            "name": "Option name",
            "description": "Brief description of this option",
            "pros": ["List of advantages"],
            "cons": ["List of disadvantages"]
        }},
        ...
    ]
}}
"""

        # Call Azure OpenAI API
        response = self.openai_client.generate_completion(
            messages=[
                {"role": "system", "content": "You are an expert analytical thinker."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse and return the options
        try:
            result = json.loads(response)
            return result.get("options", [])
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    return result.get("options", [])
                except:
                    # Return default options if extraction fails
                    return [
                        {
                            "name": "Default Option 1",
                            "description": "Generated due to parsing error",
                            "pros": ["Based on available research"],
                            "cons": ["Limited by parsing error"]
                        }
                    ]
            else:
                # Return default options
                return [
                    {
                        "name": "Default Option",
                        "description": "Generated due to parsing error",
                        "pros": ["Based on available research"],
                        "cons": ["Limited by parsing error"]
                    }
                ]


# Example usage
if __name__ == "__main__":
    decision_agent = DecisionAgent()
    
    # Example options
    sample_options = [
        {
            "name": "Standard VIX Calculation",
            "description": "Calculate VIX using the standard CBOE methodology",
            "pros": ["Industry standard approach", "Well-documented formula"],
            "cons": ["Computationally complex", "Requires option price data"]
        },
        {
            "name": "Simplified Volatility Estimation",
            "description": "Use a simplified approach based on historical price data",
            "pros": ["Easier to implement", "Works with limited data"],
            "cons": ["Less accurate than standard VIX", "May miss forward-looking signals"]
        }
    ]
    
    # Make a decision
    result = decision_agent.evaluate_options(
        query="What is the best way to calculate the VIX score?",
        options=sample_options
    )
    
    print(json.dumps(result, indent=2))
