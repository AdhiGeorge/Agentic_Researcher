"""Swarm-based Orchestrator for Agentic Researcher

This module implements the main orchestrator using OpenAI's Swarm framework.
It coordinates the flow between different specialized agents to handle research requests.
Uses Azure OpenAI GPT-4o and text-embedding-3-small models for processing.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Union
from swarm import Swarm, Agent

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent modules
from src.agents.planner.planner_agent import PlannerAgent
from src.agents.researcher.researcher_agent import ResearcherAgent
from src.agents.formatter.formatter_agent import FormatterAgent
from src.agents.action.action_agent import ActionAgent
from src.agents.answer.answer_agent import AnswerAgent
from src.agents.runner.runner_agent import RunnerAgent
from src.agents.feature.feature_agent import FeatureAgent
from src.agents.patcher.patcher_agent import PatcherAgent
from src.agents.reporter.reporter_agent import ReporterAgent
from src.agents.decision.decision_agent import DecisionAgent

# Import database and utils
from src.db.sqlite_manager import SQLiteManager
from src.db.qdrant_manager import QdrantManager
from src.utils.config import config
from src.utils.keyword_extractor import KeywordExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SwarmOrchestrator:
    """Swarm-based Orchestrator for managing multi-agent interactions.
    
    This class uses the OpenAI Swarm framework to manage the flow between different
    specialized agents for handling complex research tasks.
    
    Attributes:
        config (ConfigLoader): Configuration loader
        logger (logging.Logger): Logger for the orchestrator
        swarm (Swarm): Swarm instance for agent orchestration
        agents (Dict[str, Agent]): Dictionary of available agents
        sqlite_manager (SQLiteManager): SQLite database manager
        qdrant_manager (QdrantManager): Qdrant vector database manager
        keyword_extractor (KeywordExtractor): Utility for extracting keywords
    """
    
    def __init__(self):
        """Initialize the SwarmOrchestrator with necessary components."""
        self.config = config
        self.logger = logging.getLogger("orchestrator.swarm")
        
        # Initialize database managers
        self.sqlite_manager = SQLiteManager()
        self.qdrant_manager = QdrantManager()
        
        # Initialize keyword extractor
        self.keyword_extractor = KeywordExtractor()
        
        # Initialize the agent instances
        self.agents = self._initialize_agents()
        
        # Create Swarm client with Azure OpenAI
        self.swarm = self._initialize_swarm()
        
        self.logger.info("SwarmOrchestrator initialized successfully")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all specialized agents.
        
        Returns:
            Dict[str, Any]: Dictionary of agent instances
        """
        agents = {
            "planner": PlannerAgent(),
            "researcher": ResearcherAgent(),
            "formatter": FormatterAgent(),
            "action": ActionAgent(),
            "answer": AnswerAgent(),
            "runner": RunnerAgent(),
            "feature": FeatureAgent(),
            "patcher": PatcherAgent(),
            "reporter": ReporterAgent(),
            "decision": DecisionAgent()
        }
        
        self.logger.info(f"Initialized {len(agents)} specialized agents")
        return agents
    
    def _initialize_swarm(self) -> Swarm:
        """Initialize the Swarm framework with Azure OpenAI.
        
        Returns:
            Swarm: Configured Swarm instance
        """
        # Use the config singleton for Azure OpenAI credentials
        api_key = self.config.azure_openai_api_key
        api_version = self.config.azure_api_version_chat  # Use the chat API version
        azure_endpoint = self.config.azure_openai_endpoint
        deployment_name = self.config.azure_openai_deployment  # Use GPT-4o deployment
        
        # Set up Azure OpenAI client for Swarm
        import openai
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        
        # Create Swarm instance
        swarm_instance = Swarm(client=client)
        self.logger.info("Swarm framework initialized with Azure OpenAI")
        
        return swarm_instance
    
    def _create_swarm_agent(self, agent_name: str, agent_description: str, functions: List = None) -> Agent:
        """Create a Swarm Agent with the given name and functions.
        
        Args:
            agent_name (str): Name of the agent
            agent_description (str): Description of the agent
            functions (List, optional): List of functions available to the agent
            
        Returns:
            Agent: Swarm Agent instance
        """
        functions = functions or []
        agent = Agent(
            name=agent_name, 
            description=agent_description,
            functions=functions
        )
        
        self.logger.debug(f"Created Swarm Agent: {agent_name}")
        return agent
    
    def process_research_query(self, query: str) -> Dict[str, Any]:
        """Process a research query using the agent swarm.
        
        This is the main entry point for the orchestrator. It takes a research query,
        delegates to appropriate agents in sequence, and returns the results.
        
        Args:
            query (str): The research query to process
            
        Returns:
            Dict[str, Any]: Results of the research process
        """
        self.logger.info(f"Processing research query: {query}")
        
        # Create a temporary project for this query if needed
        temp_project_name = f"Temp Project: {query[:30]}..."
        temp_project_id = self.sqlite_manager.create_project(temp_project_name, f"Temporary project for query: {query}")
        
        # Store query in SQLite for future reference
        query_id = self.sqlite_manager.store_query(project_id=temp_project_id, query=query)
        
        # Check if similar query exists in database
        similar_query = self.sqlite_manager.find_similar_query(query)
        if similar_query:
            self.logger.info(f"Found similar previous query: {similar_query.get('query', '')}")
            # You can also use this similar query for reference if needed
        
        # Step 1: Create research plan using the planner agent
        planner_agent = self._create_swarm_agent(
            "Planner", 
            "Creates a detailed research plan based on the query",
            [self.agents["planner"].create_plan]
        )
        
        # Initial message with the query
        initial_messages = [
            {"role": "system", "content": "You are a research planning assistant that creates detailed research plans."}, 
            {"role": "user", "content": f"Create a research plan for this query: {query}"}
        ]
        
        # Run the planner agent to create a plan
        planner_response = self.swarm.run(agent=planner_agent, messages=initial_messages)
        plan = planner_response.messages[-1]["content"]
        self.logger.info("Research plan created")
        
        # Extract keywords from the query and plan for better research
        keywords = self.keyword_extractor.extract_keywords(query + " " + plan)
        self.logger.info(f"Extracted keywords: {keywords}")
        
        # Step 2: Research using the researcher agent
        researcher_agent = self._create_swarm_agent(
            "Researcher",
            "Conducts web research based on the research plan",
            [self.agents["researcher"].search_web, self.agents["researcher"].scrape_content]
        )
        
        # Messages for the researcher with the plan and keywords
        researcher_messages = [
            {"role": "system", "content": "You are a research assistant that finds relevant information from the web."}, 
            {"role": "user", "content": f"Research for information based on this plan: {plan}\n\nFocus on these keywords: {', '.join(keywords)}"}
        ]
        
        # Run the researcher agent to conduct research
        researcher_response = self.swarm.run(agent=researcher_agent, messages=researcher_messages)
        research_results = researcher_response.messages[-1]["content"]
        self.logger.info("Web research completed")
        
        # Step 3: Format the research results
        formatter_agent = self._create_swarm_agent(
            "Formatter",
            "Formats and structures research results",
            [self.agents["formatter"].format_content]
        )
        
        # Messages for the formatter with research results
        formatter_messages = [
            {"role": "system", "content": "You are an assistant that formats and structures research findings."}, 
            {"role": "user", "content": f"Format these research results into a well-structured format: {research_results}"}
        ]
        
        # Run the formatter agent to structure the research
        formatter_response = self.swarm.run(agent=formatter_agent, messages=formatter_messages)
        formatted_results = formatter_response.messages[-1]["content"]
        self.logger.info("Research results formatted")
        
        # Step 4: Final action based on the research and original query
        action_agent = self._create_swarm_agent(
            "Action",
            "Takes appropriate action based on research results",
            [self.agents["action"].execute_action]
        )
        
        # Messages for the action agent with the formatted results
        action_messages = [
            {"role": "system", "content": "You are an assistant that takes appropriate actions based on research."}, 
            {"role": "user", "content": f"Based on this original query: {query}\n\nAnd these research findings: {formatted_results}\n\nTake appropriate action to address the query."}
        ]
        
        # Run the action agent to produce the final response
        action_response = self.swarm.run(agent=action_agent, messages=action_messages)
        final_result = action_response.messages[-1]["content"]
        self.logger.info("Final action completed")
        
        # Store results in database
        self.sqlite_manager.save_results(query_id, final_result)
        
        # Store formatted research in Qdrant for future retrieval
        self.qdrant_manager.store_research_data(query_id, formatted_results)
        
        return {
            "query": query,
            "plan": plan,
            "keywords": keywords,
            "research_results": research_results,
            "formatted_results": formatted_results,
            "final_result": final_result
        }
    
    def handle_followup_question(self, question: str, conversation_history: List[Dict[str, str]]) -> str:
        """Handle follow-up questions using the action agent.
        
        Args:
            question (str): The follow-up question
            conversation_history (List[Dict[str, str]]): Previous conversation history
            
        Returns:
            str: Response to the follow-up question
        """
        self.logger.info(f"Processing follow-up question: {question}")
        
        # Create action agent for follow-up
        action_agent = self._create_swarm_agent(
            "Action",
            "Handles follow-up questions based on previous context",
            [self.agents["action"].execute_action]
        )
        
        # Prepare messages with conversation history and follow-up question
        messages = [
            {"role": "system", "content": "You are an assistant that answers follow-up questions based on previous conversations."}
        ]
        
        # Add conversation history
        for msg in conversation_history:
            messages.append(msg)
        
        # Add the follow-up question
        messages.append({"role": "user", "content": question})
        
        # Run the action agent to handle the follow-up
        response = self.swarm.run(agent=action_agent, messages=messages)
        answer = response.messages[-1]["content"]
        self.logger.info("Follow-up question answered")
        
        return answer
    
    def get_agent_monologue(self, prompt: str) -> str:
        """Get an internal monologue for the current agent action.
        
        Args:
            prompt (str): The current prompt being processed
            
        Returns:
            str: A human-like internal monologue string
        """
        # Create a simple system prompt for the monologue
        messages = [
            {"role": "system", "content": "Create a short, human-like internal monologue about the current task."}, 
            {"role": "user", "content": f"Current task: {prompt[:100]}..."}
        ]
        
        # Import the AzureOpenAIClient
        from src.utils.azure_openai import AzureOpenAIClient
        
        # Use the AzureOpenAIClient for consistent settings
        client = AzureOpenAIClient()
        
        # Generate a completion with the monologue prompt
        monologue = client.generate_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=60
        )
        
        # Ensure we have a string, trimmed of any whitespace
        monologue = monologue.strip() if monologue else "Thinking about how to approach this task..."
        return monologue
    
    def __str__(self) -> str:
        """String representation of the orchestrator.
        
        Returns:
            str: String representation
        """
        return f"SwarmOrchestrator with {len(self.agents)} agents"


if __name__ == "__main__":
    # Example usage
    orchestrator = SwarmOrchestrator()
    
    # Example research query
    query = "What is volatility index and what is the mathematical formula to calculate the VIX score. Also write a python code to calculate the vix score."
    
    try:
        print(f"Processing query: {query}")
        
        # Get internal monologue for demonstration
        monologue = orchestrator.get_agent_monologue(f"Planning research for: {query}")
        print(f"\nInternal monologue: {monologue}\n")
        
        # Process the query
        results = orchestrator.process_research_query(query)
        
        print("\nFinal Result:")
        print(results["final_result"])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
    print("\nNote: This is a demonstration. In production, use this orchestrator through the Streamlit UI.")
