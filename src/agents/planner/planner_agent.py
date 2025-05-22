"""Planner Agent for Agentic Researcher

This module implements the Planner agent that creates a structured research plan
based on the user's query. It breaks down complex research tasks into subtasks.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Union

# Add the project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Use imports that work for both direct execution and when imported as a module
from src.agents.base_agent import BaseAgent
from src.utils.config import config

class PlannerAgent(BaseAgent):
    """Planner agent that creates a structured research plan.
    
    The Planner agent breaks down complex research tasks into subtasks and 
    creates a step-by-step plan for addressing the query.
    
    Attributes:
        config (ConfigLoader): Configuration loader
        plan_template (str): Template for generating plans
    """
    
    def __init__(self):
        """Initialize the PlannerAgent."""
        super().__init__(name="Planner", description="Creates structured research plans")
        # Using the global config object imported at the top of the file
        self.config = config
        self.logger.info("PlannerAgent initialized")
        
        # Template for generating research plans
        self.plan_template = """
        # Research Plan for: {query}
        
        ## Main Objective
        {objective}
        
        ## Subtasks
        {subtasks}
        
        ## Key Areas to Research
        {key_areas}
        
        ## Expected Deliverables
        {deliverables}
        """
    
    async def execute(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Execute the planner agent with the given prompt and context.
        
        This method serves as a bridge between the SwarmOrchestrator and the agent's process method.
        
        Args:
            prompt (str): The research query or prompt
            context (Dict[str, Any], optional): Additional context information
            
        Returns:
            str: The research plan as a formatted string
        """
        if context is None:
            context = {}
            
        # Prepare input data for the process method
        input_data = {
            "query": prompt,
            **context
        }
        
        # Call the process method and get the results
        results = self.process(input_data)
        
        # Return the formatted research plan
        return results.get("plan", "No research plan generated")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input query and create a research plan.
        
        Args:
            input_data (Dict[str, Any]): Input data containing the query
            
        Returns:
            Dict[str, Any]: The research plan and metadata
        """
        query = input_data.get("query", "")
        if not query:
            self.logger.error("No query provided to PlannerAgent")
            raise ValueError("No query provided to PlannerAgent")
        
        self.log_activity(f"Creating research plan for query: {query}")
        
        # Generate a plan using Azure OpenAI
        plan = self.create_plan(query)
        
        # Update agent state
        self.update_state({
            "last_query": query,
            "last_plan": plan
        })
        
        return {
            "query": query,
            "plan": plan,
            "plan_metadata": self._extract_plan_metadata(plan)
        }
    
    def create_plan(self, query: str) -> str:
        """Create a structured research plan for the given query.
        
        Args:
            query (str): The research query to create a plan for
            
        Returns:
            str: A structured research plan
        """
        self.log_activity(f"Creating plan for: {query}")
        
        # Get Azure OpenAI credentials from config using direct attribute access
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
        
        # Import the prompt loader - use dual import approach to handle both module and direct execution
        try:
            # When imported as a module
            from src.utils.prompt_loader import PromptLoader
        except ModuleNotFoundError:
            # When run directly as a script
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from utils.prompt_loader import PromptLoader
        
        # Skip trying to load from prompts.yaml since there are issues with the PromptLoader
        # Just use our default prompt
        system_prompt = None
        
        # Use the default prompt since we're skipping the prompt loader
        if True:
            self.logger.warning("Failed to load planner system prompt from prompts.yaml, using default")
            system_prompt = """
            You are an expert AI research planner. Your job is to break down complex research queries into specific, actionable subtasks.
            Create detailed, structured research plans that are specific and directly applicable to the user's query.
            Be extremely specific, thorough, and practical in your planning. Think like you are coordinating a team of specialized agents who will execute this plan.
            """
        
        # Enhanced user prompt that encourages chain-of-thought reasoning
        user_prompt = f"""I need a comprehensive research plan for the following query:

"{query}"

Please think through this problem step-by-step using the chain-of-thought process. 
First analyze what the query is fundamentally asking, then identify the key components, 
and finally formulate a detailed plan that assigns specific tasks to appropriate agents.

Make sure to consider:
1. What information needs to be researched (websites, academic resources, documentation)
2. What conceptual understanding is required
3. What technical implementation will be needed (code samples, algorithms, data structures)
4. How the different components should be integrated

Please be thorough in your planning process and create a plan that could be executed by a team of specialized agents."""
        
        
        # Make API call to Azure OpenAI
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,  # Slightly higher temperature for more creative but still structured planning
                max_tokens=2000,  # Double the tokens to allow for detailed chain-of-thought planning
                top_p=0.95,       # Keep output focused but allow some creativity
                frequency_penalty=0.2  # Reduce repetition in the planning
            )
            
            plan = response.choices[0].message.content.strip()
            self.log_activity("Plan created successfully")
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating plan: {str(e)}")
            # Fallback to template-based plan if API fails
            return self._create_fallback_plan(query)
    
    def _create_fallback_plan(self, query: str) -> str:
        """Create a fallback plan if the API call fails.
        
        Args:
            query (str): The research query
            
        Returns:
            str: A basic research plan using the template
        """
        self.log_activity("Using fallback plan creation")
        
        # Basic fallback plan
        return self.plan_template.format(
            query=query,
            objective=f"Research and gather information about '{query}'",
            subtasks="1. Gather background information\n2. Identify key concepts\n3. Collect relevant data\n4. Analyze findings\n5. Present results",
            key_areas="- Background context\n- Core concepts\n- Related theories\n- Recent developments\n- Practical applications",
            deliverables="- Comprehensive research summary\n- Analysis of key findings\n- Actionable insights\n- Recommendations for further research"
        )
    
    def _extract_plan_metadata(self, plan: str) -> Dict[str, Any]:
        """Extract metadata from the generated plan for easier processing.
        
        Args:
            plan (str): The generated research plan
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        # Basic metadata extraction
        metadata = {
            "subtasks": [],
            "key_areas": [],
            "has_code_requirement": False
        }
        
        # Check for code requirement
        if "code" in plan.lower() or "python" in plan.lower() or "script" in plan.lower():
            metadata["has_code_requirement"] = True
        
        # Try to extract subtasks (basic approach)
        subtask_section = False
        for line in plan.split("\n"):
            if "subtask" in line.lower() or "step" in line.lower():
                subtask_section = True
                continue
            
            if subtask_section and line.strip() and not line.startswith("#"):
                # Remove list markers and numbers
                clean_line = line.strip()
                for marker in ["- ", "* ", "• "]:
                    if clean_line.startswith(marker):
                        clean_line = clean_line[len(marker):]
                        break
                
                # Remove leading numbers (e.g., "1. ")
                if clean_line and clean_line[0].isdigit() and ". " in clean_line[:4]:
                    clean_line = clean_line[clean_line.index(". ") + 2:]
                
                if clean_line:
                    metadata["subtasks"].append(clean_line)
            
            # Simple heuristic to detect end of subtasks section
            if subtask_section and line.startswith("##"):
                subtask_section = False
        
        return metadata


if __name__ == "__main__":
    # Example usage
    planner = PlannerAgent()
    
    # Test with a sample query
    query = "What is volatility index and what is the mathematical formula to calculate the VIX score. Also write a python code to calculate the vix score."
    
    try:
        plan = planner.create_plan(query)
        print("\nGenerated Research Plan:")
        print(plan)
        
        # Extract and show metadata
        metadata = planner._extract_plan_metadata(plan)
        print("\nExtracted Metadata:")
        print(json.dumps(metadata, indent=2))
        
        # Test with process method
        result = planner.process({"query": query})
        print("\nProcess Result:")
        print(f"Query: {result['query']}")
        print(f"Plan Metadata: {json.dumps(result['plan_metadata'], indent=2)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
