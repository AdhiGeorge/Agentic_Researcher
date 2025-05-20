"""
Swarm Orchestrator for Agentic Researcher

This module implements the core orchestration layer using OpenAI's Swarm framework.
It coordinates the agent flow in a hierarchical manner with advanced planning,
self-revisioning, and context-aware caching.
"""

import os
import logging
import sys
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Add project root to Python path to allow direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import OpenAI's Swarm framework
from swarm import Swarm, Agent

# Import utilities - wrap in try/except to handle missing dependencies for demos
try:
    from src.utils.config import config
    from src.utils.openai_client import get_chat_client
    from src.utils.semantic_cache import SemanticCache
    from src.utils.query_hash import QueryHashManager
    from src.utils.graph_builder import KnowledgeGraphBuilder
    
    # Import database managers
    from src.db.sqlite_manager import SQLiteManager
    from src.db.qdrant_manager import QdrantManager
    
    # Import agent interfaces
    from src.agents.base_agent import BaseAgent
    
    DEPENDENCIES_LOADED = True
except ImportError as e:
    print(f"Note: Some dependencies not available: {str(e)}")
    print("This is expected in demo mode - will use mock implementations")
    DEPENDENCIES_LOADED = False
    
    # Define minimal versions for demo purposes
    class Config:
        def __init__(self):
            self.api_key = "demo-key"
            self.embedding_model = "text-embedding-demo"
            self.model = "gpt-4-demo"
    
    config = Config()
    
    class BaseAgent:
        """Base agent class for demo purposes"""
        def __init__(self):
            pass
            
        async def execute(self, prompt, context=None):
            return {"response": f"Mock response to: {prompt}"}
            
        def get_status(self):
            return {"status": "ready"}

# Configure logging
logger = logging.getLogger(__name__)

class SwarmOrchestrator:
    """
    Swarm Orchestrator for Agentic Researcher.
    
    This class implements the core orchestration layer using OpenAI's Swarm framework.
    It provides:
    1. Hierarchical agent orchestration
    2. Self-revisioning chain of thought
    3. Entity-based contextual graphing
    4. Context-aware prompt caching
    5. Dynamic agent allocation
    
    Attributes:
        config: Configuration settings
        sqlite_manager: SQLite database manager
        qdrant_manager: Qdrant vector database manager
        swarm: OpenAI Swarm instance
        agents: Dictionary of registered agent instances
        semantic_cache: Semantic cache for prompt/response pairs
        graph_builder: Knowledge graph builder for contextual understanding
        query_hash_manager: Manager for efficient query deduplication
    """
    
    def __init__(self):
        """Initialize the SwarmOrchestrator with database connections and caching."""
        self.config = config
        
        # Initialize components if dependencies are available
        if DEPENDENCIES_LOADED:
            try:
                self.sqlite_manager = SQLiteManager()
                self.qdrant_manager = QdrantManager()
                
                # Initialize OpenAI Swarm
                self.swarm = Swarm()
                
                # Initialize semantic cache and knowledge graph
                self.semantic_cache = SemanticCache()
                self.graph_builder = KnowledgeGraphBuilder()
                self.query_hash_manager = QueryHashManager()
                
                logger.info("SwarmOrchestrator initialized successfully with all dependencies")
            except Exception as e:
                logger.warning(f"Partial initialization due to: {str(e)}")
                self._initialize_mock_components()
        else:
            # Use mock components for demo/example purposes
            self._initialize_mock_components()
            logger.info("SwarmOrchestrator initialized with mock components for demo purposes")
        
        # Initialize agent registry (always needed)
        self.agents = {}
    
    def _initialize_mock_components(self):
        """Initialize mock components for demo purposes"""
        # Define minimal mock classes for demo
        class MockDB:
            def __init__(self):
                self.projects = {}
                self.collections = {}
                self.queries = {}
                
            def get_connection(self): 
                return None
                
            def close(self): 
                pass
                
            def create_project(self, name, description=""):
                project_id = len(self.projects) + 1
                self.projects[project_id] = {
                    "id": project_id,
                    "name": name,
                    "description": description,
                    "created_at": datetime.now().isoformat()
                }
                return project_id
                
            def get_project(self, project_id):
                return self.projects.get(project_id)
                
            def store_query(self, project_id, query, response):
                query_id = len(self.queries) + 1
                self.queries[query_id] = {
                    "id": query_id,
                    "project_id": project_id,
                    "query": query,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                return query_id
            
        class MockCache:
            def __init__(self):
                self.cache = {}
                self.stats = {"hits": 0, "misses": 0}
                
            def get(self, key): 
                if key in self.cache:
                    self.stats["hits"] += 1
                    return self.cache[key]
                self.stats["misses"] += 1
                return None
                
            def set(self, key, value): 
                self.cache[key] = value
                
            def get_stats(self): 
                return self.stats
            
        class MockGraph:
            def __init__(self):
                self.entities = {}
                self.relations = []
                
            def add_entity(self, entity): 
                self.entities[entity.get("id", str(len(self.entities)))] = entity
                
            def get_context(self, query): 
                return []
                
            def get_stats(self): 
                return {"entities": len(self.entities), "relationships": len(self.relations)}
            
        class MockHashManager:
            def __init__(self):
                self.hashes = {}
                
            def get_hash(self, query): 
                return self.generate_hash(query, None)
                
            def generate_hash(self, query, prefix=None): 
                # Simple hash generation to provide uniqueness
                # prefix parameter added to match the real implementation signature
                if prefix:
                    hash_val = f"{prefix}-{hash(query) % 100000}"
                else:
                    hash_val = f"mock-hash-{hash(query) % 100000}"
                self.hashes[hash_val] = query
                return hash_val
            
        # Set up mock components
        self.sqlite_manager = MockDB()
        self.qdrant_manager = MockDB()
        self.swarm = None  # Swarm not available in demo mode
        self.semantic_cache = MockCache()
        self.graph_builder = MockGraph()
        self.query_hash_manager = MockHashManager()
    
    def register_agent(self, agent_instance: BaseAgent) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_instance: Instance of a BaseAgent subclass
        """
        agent_name = agent_instance.__class__.__name__
        self.agents[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name}")
    
    async def execute_with_self_revision(self, agent_name: str, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an agent with self-revision capability.
        
        This method invokes an agent, then allows it to revise its own response
        through a reflective process, improving the quality of the output.
        
        Args:
            agent_name: Name of the agent to execute
            prompt: The input prompt for the agent
            context: Additional context for the agent execution
            
        Returns:
            Dictionary containing the agent's response and revision history
        """
        # Handle the case where the agent is not registered
        if agent_name not in self.agents:
            logger.error(f"Agent {agent_name} not registered")
            return {"error": f"Agent {agent_name} not registered"}
            
        # For the example implementation, we'll provide a simpler version
        # that demonstrates the concept without all the caching complexity
        
        # Initialize context if none provided
        context = context or {}
        context["project_id"] = context.get("project_id", 0)
        
        try:
            # Generate a query hash for caching (for demo purposes)
            query_hash = self.query_hash_manager.generate_hash(prompt, "revision")
            
            # Check if result is in cache
            cached_result = self.semantic_cache.get(query_hash)
            if cached_result:
                logger.info(f"Retrieved cached result for agent {agent_name} execution")
                cached_result["from_cache"] = True
                return cached_result
                
            # Initialize revision tracking
            revision_history = []
            
            # Execute the agent to get the initial response
            logger.info(f"Executing initial response for agent {agent_name}")
            initial_result = await self.execute_agent(agent_name, prompt, context)
            
            if "error" in initial_result:
                return initial_result
                
            initial_response = initial_result.get("response", "")
            revision_history.append(initial_response)
            
            # Create the reflection prompt
            reflection_prompt = f"Please improve your previous response about {prompt}"
            
            # Set up revision context
            revision_context = context.copy() if context else {}
            revision_context["revision"] = True
            revision_context["previous_response"] = initial_response
            revision_context["revision_count"] = 1
            
            # Execute the revision
            logger.info(f"Executing revision for agent {agent_name}")
            revision_result = await self.execute_agent(agent_name, reflection_prompt, revision_context)
            
            if "error" not in revision_result:
                revision_response = revision_result.get("response", "")
                revision_history.append(revision_response)
                
                # Use the revised response as the final response
                result = {
                    "agent": agent_name,
                    "prompt": prompt,
                    "response": revision_response,
                    "revision_history": revision_history,
                    "revisions": len(revision_history) - 1,
                    "timestamp": datetime.now().isoformat(),
                    "from_cache": False
                }
            else:
                # If revision failed, use the initial response
                result = {
                    "agent": agent_name,
                    "prompt": prompt,
                    "response": initial_response,
                    "revision_history": revision_history,
                    "revisions": 0,
                    "timestamp": datetime.now().isoformat(),
                    "from_cache": False,
                    "revision_error": revision_result.get("error")
                }
            
            # Cache the result
            self.semantic_cache.set(query_hash, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in self-revision: {str(e)}")
            # Fallback to regular execution if revision fails
            return await self.execute_agent(agent_name, prompt, context)
    
    async def execute_agent(self, agent_name: str, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an agent without self-revision.
        
        Args:
            agent_name: Name of the agent to execute
            prompt: The input prompt for the agent
            context: Additional context for the agent execution
            
        Returns:
            Dictionary containing the agent's response
        """
        # Handle the case where the agent is not registered
        if agent_name not in self.agents:
            logger.error(f"Agent {agent_name} not registered")
            return {"error": f"Agent {agent_name} not registered"}
        
        # Get the agent instance
        agent = self.agents[agent_name]
        
        # Prepare context for agent
        if context is None:
            context = {}
        
        # Log the execution
        logger.info(f"Executing agent {agent_name} with prompt: {prompt[:50]}...")
        
        # Execute the agent
        start_time = time.time()
        response = await agent.execute(prompt, context)
        execution_time = time.time() - start_time
        
        # Log execution time
        logger.info(f"Agent {agent_name} execution completed in {execution_time:.2f}s")
        
        # Return the result
        return {
            "response": response,
            "source": "agent",
            "agent": agent_name,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
    
    async def execute_workflow(self, query: str, workflow_type: str = "research") -> Dict[str, Any]:
        """
        Execute a complete workflow using multiple agents.
        
        Args:
            query: The input query/prompt
            workflow_type: Type of workflow to execute (research, code, feature, etc.)
            
        Returns:
            Dictionary containing the workflow results
        """
        try:
            # Initialize context
            context = {
                "query": query,
                "workflow_type": workflow_type,
                "intermediate_results": {},
                "start_time": datetime.now().isoformat()
            }
            
            # Create project for the workflow
            project_name = f"{workflow_type.capitalize()}: {query[:30]}" + ("..." if len(query) > 30 else "")
            project_id = self.sqlite_manager.create_project(project_name, query)
            context["project_id"] = project_id
            
            logger.info(f"Starting {workflow_type} workflow for query: {query[:50]}...")
            
            # Execute workflow based on type
            if workflow_type == "research":
                return await self._research_workflow(query, context)
            elif workflow_type == "code":
                return await self._code_workflow(query, context)
            elif workflow_type == "feature":
                return await self._feature_workflow(query, context)
            elif workflow_type == "patch":
                return await self._patch_workflow(query, context)
            else:
                return {
                    "error": f"Unknown workflow type: {workflow_type}",
                    "supported_workflows": ["research", "code", "feature", "patch"],
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}")
            return {
                "error": f"Workflow execution failed: {str(e)}",
                "workflow_type": workflow_type,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _research_workflow(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research workflow.
        
        This workflow involves planning, research, formatting, and answer synthesis.
        
        Args:
            query: The research query
            context: Workflow context
            
        Returns:
            Dictionary containing the research results
        """
        try:
            # Initialize the context if not provided
            if context is None:
                context = {}
            
            # Create a project if not specified
            if "project_id" not in context:
                context["project_id"] = self.sqlite_manager.create_project(
                    f"Research: {query[:50]}", 
                    f"Auto-generated research project for query: {query}"
                )
            
            # For storing intermediate results
            if "intermediate_results" not in context:
                context["intermediate_results"] = {}
                
            logger.info(f"Starting research workflow for query: {query[:50]}...")
            
            # Step 1: Planning with PlannerAgent
            logger.info("Step 1: Planning")
            planning_result = await self.execute_agent("PlannerAgent", query, context)
            if "error" in planning_result:
                return {"error": f"Planning failed: {planning_result['error']}"}
                
            context["intermediate_results"]["plan"] = planning_result
            
            # Step 2: Research based on the plan
            logger.info("Step 2: Research")
            research_prompt = f"Research information to answer: {query}"
            research_result = await self.execute_agent("ResearcherAgent", research_prompt, context)
            if "error" in research_result:
                return {"error": f"Research failed: {research_result['error']}"}
                
            context["intermediate_results"]["research"] = research_result
            
            # Step 3: Answer synthesis with WriterAgent
            logger.info("Step 3: Writing answer")
            answer_prompt = f"Write a comprehensive answer to: {query}"
            answer_result = await self.execute_agent("WriterAgent", answer_prompt, context)
            if "error" in answer_result:
                return {"error": f"Answer synthesis failed: {answer_result['error']}"}
            
            # Store the query and result
            self.sqlite_manager.store_query(
                context["project_id"],
                query,
                answer_result["response"]
            )
            
            # Return the complete workflow result
            return {
                "query": query,
                "plan": planning_result["response"],
                "research": research_result["response"],
                "answer": answer_result["response"],
                "project_id": context["project_id"],
                "workflow_type": "research",
                "completion_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in research workflow: {str(e)}")
            return {"error": f"Research workflow failed: {str(e)}"}
    
    async def _code_workflow(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the code generation workflow.
        
        This workflow involves planning, research, code generation, and validation.
        
        Args:
            query: The code generation query
            context: Workflow context
            
        Returns:
            Dictionary containing the generated code
        """
        # Step 1: Planning with self-revision
        planning_result = await self.execute_with_self_revision("PlannerAgent", query, context)
        context["intermediate_results"]["plan"] = planning_result
        
        # Step 2: Research (if needed)
        research_prompt = f"""
        Based on the following code plan:
        {planning_result['response']}
        
        Research any necessary information to implement the code for: {query}
        """
        research_result = await self.execute_agent("ResearcherAgent", research_prompt, context)
        context["intermediate_results"]["research"] = research_result
        
        # Step 3: Code generation with self-revision
        code_prompt = f"""
        Based on:
        1. The plan: {planning_result['response']}
        2. The research: {research_result['response']}
        
        Generate code to address: {query}
        """
        code_result = await self.execute_with_self_revision("CodeAgent", code_prompt, context)
        context["intermediate_results"]["code"] = code_result
        
        # Step 4: Code execution/validation (if applicable)
        validation_prompt = f"""
        Validate the following code:
        {code_result['response']}
        
        Run the code if possible and report any issues or optimizations.
        """
        validation_result = await self.execute_agent("RunnerAgent", validation_prompt, context)
        
        # Return the complete workflow result
        return {
            "query": query,
            "code": code_result["response"],
            "plan": planning_result["response"],
            "research": research_result["response"],
            "validation": validation_result["response"],
            "project_id": context["project_id"],
            "workflow_type": "code",
            "completion_time": datetime.now().isoformat()
        }
    
    async def _feature_workflow(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature implementation workflow."""
        # Feature implementation workflow logic here
        # Similar structure to the other workflows
        return {"status": "Not yet implemented"}
    
    async def _patch_workflow(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bug fixing/patching workflow."""
        # Bug fixing workflow logic here
        # Similar structure to the other workflows
        return {"status": "Not yet implemented"}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get the status of all registered agents.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "agent_count": len(self.agents),
            "registered_agents": list(self.agents.keys()),
            "cache_stats": self.semantic_cache.get_stats(),
            "graph_stats": self.graph_builder.get_stats()
        }


# Example usage when run directly
if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*80)
    print("SWARM ORCHESTRATOR EXAMPLE USAGE")
    print("="*80)
    print("This example demonstrates how to use the SwarmOrchestrator to coordinate multiple agents")
    print()
    
    # Define proper agent classes that will register correctly with the orchestrator
    # The orchestrator uses the class name for registration, so we need separate classes
    
    class PlannerAgent(BaseAgent):
        """Agent responsible for planning research or other activities"""
        def __init__(self):
            self.call_count = 0
            super().__init__()
            
        async def execute(self, prompt, context=None):
            """Execute the planning agent with the given prompt"""
            self.call_count += 1
            context = context or {}
            
            # For a real implementation, this would call an LLM
            # We're simulating the response here to avoid API calls during example
            response = f"Response from PlannerAgent for: {prompt}"
            response += "\n\nResearch Plan:\n1. Identify key concepts\n2. Search latest papers\n3. Analyze findings\n4. Synthesize results"
            
            return {
                "agent": "PlannerAgent",
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
        def get_status(self):
            return {
                "agent_name": "PlannerAgent",
                "call_count": self.call_count,
                "status": "ready"
            }
    
    class ResearcherAgent(BaseAgent):
        """Agent responsible for conducting research"""
        def __init__(self):
            self.call_count = 0
            self.revision_count = 0
            super().__init__()
            
        async def execute(self, prompt, context=None):
            """Execute the researcher agent with the given prompt"""
            self.call_count += 1
            context = context or {}
            
            # Check if this is a revision request
            is_revision = False
            previous_response = ""
            if context and "revision" in context and context["revision"]:
                is_revision = True
                previous_response = context.get("previous_response", "")
                self.revision_count += 1
            
            response = f"Response from ResearcherAgent for: {prompt}"
            
            if is_revision:
                response += f"\n\nREVISION #{self.revision_count} - Improving previous response:\n"
                response += "\n\nImproved Research Findings:\n"
                response += "- Advanced Finding 1: Quantum Error Correction significantly improved with topological codes\n"
                response += "- Advanced Finding 2: Quantum Machine Learning applications in drug discovery showing practical results\n"
                response += "- Advanced Finding 3: Quantum Internet Protocols demonstrating secure communication over 100+ kilometers\n"
                response += "- New Finding 4: Quantum advantage demonstrated in specific optimization problems\n"
            else:
                response += "\n\nInitial Research Findings:\n"
                response += "- Advance 1: Quantum Error Correction\n"
                response += "- Advance 2: Quantum Machine Learning\n"
                response += "- Advance 3: Quantum Internet Protocols"
            
            return {
                "agent": "ResearcherAgent",
                "prompt": prompt,
                "response": response,
                "is_revision": is_revision,
                "previous_response": previous_response if is_revision else None,
                "timestamp": datetime.now().isoformat()
            }
            
        def get_status(self):
            return {
                "agent_name": "ResearcherAgent",
                "call_count": self.call_count,
                "status": "ready"
            }
    
    class WriterAgent(BaseAgent):
        """Agent responsible for writing and formatting content"""
        def __init__(self):
            self.call_count = 0
            super().__init__()
            
        async def execute(self, prompt, context=None):
            """Execute the writer agent with the given prompt"""
            self.call_count += 1
            context = context or {}
            
            response = f"Response from WriterAgent for: {prompt}"
            response += "\n\nQuantum computing has advanced significantly in recent years with breakthroughs in error correction, machine learning applications, and protocols for quantum internet."
            
            return {
                "agent": "WriterAgent",
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
        def get_status(self):
            return {
                "agent_name": "WriterAgent",
                "call_count": self.call_count,
                "status": "ready"
            }
    
    async def run_example():
        print("\n1. INITIALIZING ORCHESTRATOR\n" + "-"*30)
        # Create orchestrator with actual implementation
        orchestrator = SwarmOrchestrator()
        print("SwarmOrchestrator initialized")
        
        # Register agents
        print("\n2. REGISTERING AGENTS\n" + "-"*30)
        # Create and register agents
        orchestrator.register_agent(PlannerAgent())
        orchestrator.register_agent(ResearcherAgent())
        orchestrator.register_agent(WriterAgent())
        print(f"Registered {len(orchestrator.agents)} agents")
        
        # Show agent status
        agent_status = orchestrator.get_agent_status()
        print(f"Agent Status: {len(agent_status['registered_agents'])} agents ready")
        print(f"Registered Agents: {', '.join(agent_status['registered_agents'])}")
        
        # Example query
        research_query = "What are the latest advancements in quantum computing?"
        
        print("\n3. EXECUTING INDIVIDUAL AGENT\n" + "-"*30)
        print(f"Sending query to PlannerAgent: '{research_query}'")
        
        # Execute a single agent
        planner_result = await orchestrator.execute_agent(
            "PlannerAgent",
            research_query,
            {"project_id": 123}
        )
        
        if planner_result.get("error"):
            print(f"Error: {planner_result['error']}")
        else:
            print("\nPlannerAgent Response:")
            print("-" * 50)
            print(planner_result.get("response", "No response"))
            print("-" * 50)
        
        print("\n4. DEMONSTRATING SELF-REVISION CAPABILITY\n" + "-"*30)
        print(f"Executing ResearcherAgent with self-revision for: '{research_query}'")
        
        try:
            revision_result = await orchestrator.execute_with_self_revision(
                "ResearcherAgent",
                research_query,
                {"project_id": 123}
            )
            
            if revision_result.get("error"):
                print(f"Error: {revision_result['error']}")
            else:
                print("\nResearcherAgent Response with Self-Revision:")
                print("-" * 50)
                print(revision_result.get("response", "No response"))
                if "revision_history" in revision_result:
                    print("\nRevision History:")
                    for i, revision in enumerate(revision_result["revision_history"]):
                        if isinstance(revision, str):
                            print(f"Revision {i+1}: {revision[:100]}...")
                        else:
                            # Handle dictionary responses
                            rev_text = revision.get("response", "No content") if isinstance(revision, dict) else str(revision)
                            print(f"Revision {i+1}: {str(rev_text)[:100]}...")
                print("-" * 50)
        except Exception as e:
            print(f"Self-revision example could not complete: {str(e)}")
        
        print("\n5. WORKFLOW EXECUTION EXAMPLE\n" + "-"*30)
        print(f"Executing research workflow for: '{research_query}'")
        
        try:
            # Attempt to execute a workflow
            workflow_result = await orchestrator.execute_workflow(
                research_query,
                "research"
            )
            
            if workflow_result.get("error"):
                print(f"Error: {workflow_result['error']}")
            else:
                print("\nWorkflow Result:")
                print("-" * 50)
                print(f"Query: {workflow_result.get('query', 'N/A')}")
                print(f"Plan: {workflow_result.get('plan', 'N/A')}")
                print(f"Research: {workflow_result.get('research', 'N/A')}")
                print(f"Answer: {workflow_result.get('answer', 'N/A')}")
                print("-" * 50)
        except Exception as e:
            print(f"Workflow example could not complete: {str(e)}")
            print("This is expected as this example doesn't implement full workflow functionality")
            print("In a production environment, these workflow methods would call actual agents")
        
        print("\nSwarm Orchestrator example completed!")
        print("="*80)
        print("NOTE: This example shows the core orchestration structure.")
        print("In production, agents would call LLMs and use other external resources.")
        print("="*80)
    
    # Run the async example
    try:
        asyncio.run(run_example())
    except Exception as e:
        print(f"Example execution error: {str(e)}")
        print("="*80)
