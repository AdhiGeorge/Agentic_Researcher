#!/usr/bin/env python
"""Agentic Researcher - Console Interface

This is the main entry point for the Agentic Researcher
console application. It provides a command-line interface for research query execution,
followup questions, and project management without any UI components.

Usage examples:
    python main.py research "What is volatility index and what is the mathematical formula to calculate the VIX score?"
    python main.py code "Write a Python code to calculate the VIX score."
    python main.py list
    python main.py details 123
    python main.py followup "Can you explain the implications of a high VIX score?" --project-id 123
"""
import os
import sys
import argparse
import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agentic_researcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agentic_researcher")

# Make sure the package is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Agentic Researcher components
from src.orchestrator.swarm_orchestrator import SwarmOrchestrator
from src.db.sqlite_manager import SQLiteManager
from src.db.qdrant_manager import QdrantManager

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Agentic Researcher CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Execute a research query")
    research_parser.add_argument("query", type=str, help="Research query to execute")
    research_parser.add_argument("--project-name", type=str, help="Project name (generated from query if not provided)")
    research_parser.add_argument("--no-cache", action="store_true", help="Force fresh research even if cached results exist")
    
    # Code command for code-specific queries
    code_parser = subparsers.add_parser("code", help="Generate code for a specific task")
    code_parser.add_argument("query", type=str, help="Code generation query")
    code_parser.add_argument("--language", type=str, default="python", help="Programming language (default: python)")
    code_parser.add_argument("--project-id", type=int, help="Project ID for context")
    
    # Follow-up command
    followup_parser = subparsers.add_parser("followup", help="Ask a follow-up question")
    followup_parser.add_argument("query", type=str, help="Follow-up query")
    followup_parser.add_argument("--project-id", type=int, required=True, help="Project ID")
    followup_parser.add_argument("--action", type=str, default="answer", 
                               choices=["answer", "run", "feature", "bug", "report"],
                               help="Action type (answer, run, feature, etc.)")
    
    # List projects command
    list_parser = subparsers.add_parser("list", help="List all projects")
    
    # Project details command
    details_parser = subparsers.add_parser("details", help="Show project details")
    details_parser.add_argument("project_id", type=int, help="Project ID")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    
    return parser.parse_args()

class AgenticResearcherCLI:
    """Agentic Researcher Command Line Interface handler"""
    
    def __init__(self):
        """Initialize the Agentic Researcher CLI"""
        self.sqlite_manager = SQLiteManager()
        self.swarm_orchestrator = None  # Lazy initialization
        logger.info("Agentic Researcher CLI initialized")
    
    async def _ensure_swarm_orchestrator(self):
        """Ensure the Swarm Orchestrator instance is initialized"""
        if self.swarm_orchestrator is None:
            self.swarm_orchestrator = SwarmOrchestrator()
            logger.info("Swarm Orchestrator initialized")
    
    async def research(self, query, project_name=None, use_cache=True):
        """Execute a research query using available orchestrator methods"""
        # Ensure the swarm orchestrator is initialized
        await self._ensure_swarm_orchestrator()
        
        # Create project if name provided
        project_id = None
        if project_name:
            project_id = self.sqlite_manager.create_project(
                name=project_name,
                description=f"Research project for: {query}"
            )
            logger.info(f"Created new project with ID {project_id}")
        
        # Show spinner
        spinner_chars = ['|', '/', '-', '\\']
        start_time = time.time()
        i = 0
        
        print(f"Executing research for query: '{query}'")
        context = {
            "project_id": project_id,
            "use_cache": use_cache
        }
        
        try:
            # Step 1: Execute PlannerAgent to create research plan
            print("\nStep 1: Planning research approach...")
            planning_task = asyncio.create_task(self.swarm_orchestrator.execute_agent(
                "PlannerAgent",
                query,
                context
            ))
            
            # Show spinner while planning is processing
            while not planning_task.done():
                elapsed = time.time() - start_time
                spinner = spinner_chars[i % len(spinner_chars)]
                i += 1
                print(f"\rPlanning {spinner} (elapsed: {elapsed:.1f}s)", end="")
                await asyncio.sleep(0.1)
            
            # Get planning results
            planning_results = await planning_task
            print("\r" + " " * 50 + "\r", end="")
            print("Research plan created.")
            
            # Step 2: Execute ResearcherAgent with self-revision
            print("\nStep 2: Gathering and analyzing information...")
            research_start_time = time.time()
            research_task = asyncio.create_task(self.swarm_orchestrator.execute_with_self_revision(
                "ResearcherAgent",
                query,
                {**context, "plan": planning_results.get("response", "")}
            ))
            
            # Show spinner while research is processing
            while not research_task.done():
                elapsed = time.time() - research_start_time
                spinner = spinner_chars[i % len(spinner_chars)]
                i += 1
                print(f"\rResearching {spinner} (elapsed: {elapsed:.1f}s)", end="")
                await asyncio.sleep(0.1)
            
            # Get research results
            research_results = await research_task
            print("\r" + " " * 50 + "\r", end="")
            print("Information gathered and analyzed.")
            
            # Step 3: Execute WriterAgent to format final answer
            print("\nStep 3: Synthesizing final response...")
            writer_start_time = time.time()
            writer_task = asyncio.create_task(self.swarm_orchestrator.execute_agent(
                "WriterAgent",
                query,
                {**context, "research": research_results.get("response", "")}
            ))
            
            # Show spinner while writing is processing
            while not writer_task.done():
                elapsed = time.time() - writer_start_time
                spinner = spinner_chars[i % len(spinner_chars)]
                i += 1
                print(f"\rSynthesizing {spinner} (elapsed: {elapsed:.1f}s)", end="")
                await asyncio.sleep(0.1)
            
            # Get writer results
            writer_results = await writer_task
            print("\r" + " " * 50 + "\r", end="")
            
            # Combine all results
            total_time = time.time() - start_time
            results = {
                "project_id": project_id,
                "query": query,
                "results": {
                    "plan": planning_results.get("response", ""),
                    "research": research_results.get("response", ""),
                    "answer": writer_results.get("response", "")
                },
                "execution_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "from_cache": planning_results.get("from_cache", False) and 
                              research_results.get("from_cache", False) and 
                              writer_results.get("from_cache", False)
            }
            
            # Print summary
            print(f"\nResearch complete in {total_time:.2f} seconds")
            if results.get("from_cache", False):
                print("Results retrieved from cache")
            
            if project_id:
                print(f"Project ID: {project_id}")
                
            # Print the final answer
            print("\n" + "-" * 50)
            print("RESEARCH RESULTS")
            print("-" * 50)
            print(writer_results.get("response", "No results available"))
            print("-" * 50)
            
            # Check for code if present
            if writer_results.get("response", "").find("```") >= 0:
                print("\nCode blocks found in the response.")
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in research workflow: {str(e)}")
            print(f"\nError executing research: {str(e)}")
            return {
                "error": str(e),
                "project_id": project_id,
                "query": query,
                "execution_time": time.time() - start_time
            }
    
    async def code(self, query, language="python", project_id=None):
        """Generate code for a specific task"""
        await self._ensure_swarm_orchestrator()
        
        # Enhance query with language context
        enhanced_query = f"Write {language} code for the following task: {query}"
        
        print(f"Generating {language} code...")
        
        # Use the same pattern as research but with code focus
        spinner_chars = ['|', '/', '-', '\\']
        start_time = time.time()
        i = 0
        
        context = {
            "project_id": project_id,
            "use_cache": True,
            "language": language,
            "code_request": True
        }
        
        try:
            # Execute WriterAgent directly for code generation
            print("\nRequesting code generation...")
            code_task = asyncio.create_task(self.swarm_orchestrator.execute_with_self_revision(
                "WriterAgent",
                enhanced_query,
                context
            ))
            
            # Show spinner while code is generating
            while not code_task.done():
                elapsed = time.time() - start_time
                spinner = spinner_chars[i % len(spinner_chars)]
                i += 1
                print(f"\rGenerating code {spinner} (elapsed: {elapsed:.1f}s)", end="")
                await asyncio.sleep(0.1)
            
            # Get code results
            code_results = await code_task
            print("\r" + " " * 50 + "\r", end="")
            
            # Format results
            results = {
                "project_id": project_id,
                "query": query,
                "results": {
                    "code": code_results.get("response", "")
                },
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "from_cache": code_results.get("from_cache", False)
            }
            
            # Print summary
            print(f"\nCode generation complete in {results['execution_time']:.2f} seconds")
            
            # Print the code
            print("\n" + "-" * 50)
            print(f"GENERATED {language.upper()} CODE")
            print("-" * 50)
            print(code_results.get("response", "No code generated"))
            print("-" * 50)
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in code generation: {str(e)}")
            print(f"\nError generating code: {str(e)}")
            return {
                "error": str(e),
                "project_id": project_id,
                "query": query,
                "execution_time": time.time() - start_time
            }
    
    async def followup(self, query, project_id, action="answer"):
        """Ask a follow-up question about previous research"""
        await self._ensure_swarm_orchestrator()
        
        # Verify project exists
        project = self.sqlite_manager.get_project(project_id)
        if not project:
            print(f"Project with ID {project_id} not found")
            return {"error": "Project not found"}
        
        # Show spinner
        spinner_chars = ['|', '/', '-', '\\']
        start_time = time.time()
        i = 0
        
        print(f"Processing follow-up for project {project_id}...")
        
        # Prepare context with previous research
        context = {
            "project_id": project_id,
            "action": action,
            "previous_research": project.get("results", {})
        }
        
        try:
            # Execute WriterAgent for follow-up response
            print(f"\nGenerating {action} response to follow-up question...")
            followup_task = asyncio.create_task(self.swarm_orchestrator.execute_agent(
                "WriterAgent",
                query,
                context
            ))
            
            # Show spinner while generating response
            while not followup_task.done():
                elapsed = time.time() - start_time
                spinner = spinner_chars[i % len(spinner_chars)]
                i += 1
                print(f"\rProcessing {spinner} (elapsed: {elapsed:.1f}s)", end="")
                await asyncio.sleep(0.1)
            
            # Get followup results
            followup_results = await followup_task
            print("\r" + " " * 50 + "\r", end="")
            
            # Format results
            results = {
                "project_id": project_id,
                "query": query,
                "action": action,
                "results": {
                    "response": followup_results.get("response", "")
                },
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Print the response
            print(f"\n=== {action.capitalize()} Response ===\n")
            print(followup_results.get("response", "No response generated"))
            print("-" * 50)
            
            return results
            
        except Exception as e:
            logger.exception(f"Error processing follow-up: {str(e)}")
            print(f"\nError processing follow-up: {str(e)}")
            return {
                "error": str(e),
                "project_id": project_id,
                "query": query,
                "execution_time": time.time() - start_time
            }
    
    def list_projects(self):
        """List all projects"""
        projects = self.sqlite_manager.get_all_projects()
        
        if not projects:
            print("No projects found")
            return []
        
        print("\n=== Research Projects ===\n")
        for project in projects:
            created_at = project.get("created_at", "")
            if created_at:
                created_at = created_at.split("T")[0]  # Just show the date part
            
            print(f"ID: {project['id']} - {project['name']} (Created: {created_at})")
        
        return projects
    
    def project_details(self, project_id):
        """Show project details"""
        project = self.sqlite_manager.get_project(project_id)
        
        if not project:
            print(f"Project with ID {project_id} not found")
            return None
        
        print(f"\n=== Project Details: {project['name']} ===\n")
        print(f"ID: {project['id']}")
        print(f"Description: {project['description'] or 'No description'}")
        print(f"Created: {project['created_at']}")
        
        # Get project state
        state = self.sqlite_manager.get_project_state(project_id)
        if state:
            print("\n=== Project State ===\n")
            for agent_type, agent_state in state.items():
                print(f"Agent: {agent_type}")
                if isinstance(agent_state, dict) and "timestamp" in agent_state:
                    print(f"Last updated: {agent_state['timestamp']}")
        
        return project
    
    async def show_stats(self):
        """Show system statistics"""
        self._ensure_swarm_orchestrator()
        
        stats = self.swarm_orchestrator.get_execution_stats()
        
        print("\n=== System Statistics ===\n")
        
        # Agent stats
        if "stats" in stats:
            print("Agent Performance:")
            for agent, agent_stats in stats["stats"].items():
                print(f"  {agent}: {agent_stats['calls']} calls, {agent_stats['successes']} successes, "
                      f"{agent_stats['avg_time']:.2f}s avg time")
        
        # Cache stats
        if "cache_stats" in stats:
            cache_stats = stats["cache_stats"]
            hit_ratio = cache_stats.get("hit_ratio", 0) * 100
            print(f"\nCache Performance:")
            print(f"  Hit ratio: {hit_ratio:.1f}%")
            print(f"  Exact hits: {cache_stats.get('exact_hits', 0)}")
            print(f"  Semantic hits: {cache_stats.get('semantic_hits', 0)}")
            print(f"  Misses: {cache_stats.get('misses', 0)}")
            print(f"  Threshold range: {cache_stats.get('threshold_range', [0.85, 0.95])}")
        
        return stats

async def main():
    """Main entry point that supports both interactive and command-line modes"""
    cli = AgenticResearcherCLI()
    
    # Print welcome message with simple ASCII art logo
    print("""
    +--------------------------------------------------+
    |               AGENTIC RESEARCHER                |
    |     Orchestrated Research & Knowledge Assistant |
    +--------------------------------------------------+
    """)
    
    # Detect if arguments were provided
    try:
        # First try to parse command-line arguments
        args = parse_arguments()
        
        # If a command was specified, use command-line mode
        if args.command:
            print(f"Running in command-line mode: '{args.command}'")
            return await run_command_line_mode(cli, args)
            
        # If no command specified, fall back to interactive mode
        print("No command specified. Starting interactive mode...")
        return await run_interactive_mode(cli)
        
    except (AttributeError, SystemExit):
        # If argument parsing fails, try interactive mode
        print("Welcome to Agentic Researcher interactive CLI.")
        try:
            return await run_interactive_mode(cli)
        except (EOFError, KeyboardInterrupt):
            print("\nInteractive mode not available in this environment.")
            print("Use command-line arguments instead (e.g., 'python main.py research \"What is VIX?\"')")
            print("Run 'python main.py --help' for usage information.")
            return 1

async def run_interactive_mode(cli):
    """Run the CLI in interactive mode with user prompts"""
    print("Type 'help' for available commands or 'exit' to quit.")
    
    # Main interactive loop
    running = True
    
    try:
        while running:
            print("\n" + "=" * 50)
            # Get user input with prompt
            user_input = input("AR> ").strip()
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting Agentic Researcher. Goodbye!")
                running = False
                continue
                
            # Check for help command
            elif user_input.lower() in ['help', 'h', '?']:
                print("""
Available commands:
  research <query>   - Execute a research query
  code <query>       - Generate code for a specific task
  followup <query>   - Ask a follow-up question about previous research
  list               - List all projects
  details <id>       - Show project details
  stats              - Show system statistics
  exit               - Exit the application
                """)
                continue
                
            # Parse the command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower() if parts else ""
            
            # Handle empty input
            if not command:
                continue
                
            # Process commands
            if command == "research":
                if len(parts) < 2:
                    query = input("Enter your research query: ")
                else:
                    query = parts[1]
                    
                project_name = input("Project name (press Enter to use default): ") or None
                use_cache = input("Use cache? (y/n, default: y): ").lower() != 'n'
                
                print("\nProcessing research query...")
                await cli.research(query, project_name, use_cache)
                
            elif command == "code":
                if len(parts) < 2:
                    query = input("Enter your code request: ")
                else:
                    query = parts[1]
                    
                language = input("Programming language (default: python): ") or "python"
                project_id = input("Project ID (press Enter for none): ") or None
                if project_id and project_id.isdigit():
                    project_id = int(project_id)
                    
                print("\nGenerating code...")
                await cli.code(query, language, project_id)
                
            elif command == "followup":
                if len(parts) < 2:
                    query = input("Enter your follow-up question: ")
                else:
                    query = parts[1]
                    
                project_id_input = input("Project ID: ")
                if not project_id_input or not project_id_input.isdigit():
                    print("Error: Project ID is required and must be a number")
                    continue
                    
                project_id = int(project_id_input)
                action = input("Action (answer/extend/critique, default: answer): ") or "answer"
                
                print("\nProcessing follow-up question...")
                await cli.followup(query, project_id, action)
                
            elif command == "list":
                cli.list_projects()
                
            elif command == "details":
                if len(parts) < 2:
                    project_id_input = input("Enter project ID: ")
                else:
                    project_id_input = parts[1]
                    
                if not project_id_input or not project_id_input.isdigit():
                    print("Error: Project ID must be a number")
                    continue
                    
                cli.project_details(int(project_id_input))
                
            elif command == "stats":
                await cli.show_stats()
                
            else:
                print(f"Unknown command: '{command}'. Type 'help' for available commands.")
    except (EOFError, KeyboardInterrupt):
        print("\nInteractive session terminated.")
    
    print("\nExiting Agentic Researcher")
    return 0

async def run_command_line_mode(cli, args):
    """Run the CLI using command-line arguments"""
    try:
        if args.command == "research":
            await cli.research(args.query, args.project_name, not args.no_cache)
            
        elif args.command == "code":
            await cli.code(args.query, args.language, args.project_id)
            
        elif args.command == "followup":
            await cli.followup(args.query, args.project_id, args.action)
            
        elif args.command == "list":
            cli.list_projects()
            
        elif args.command == "details":
            cli.project_details(args.project_id)
            
        elif args.command == "stats":
            await cli.show_stats()
            
        else:
            # No command or unknown command
            print("Please specify a command. Use --help for available commands.")
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.exception(f"Error in command execution: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    finally:    
        print("\nExiting Agentic Researcher")
        
    return 0

if __name__ == "__main__":
    # Run the async main function
    try:
        import asyncio
        sys.exit(asyncio.run(main()))
    except Exception as e:
        print(f"Error running async main: {str(e)}")
        sys.exit(1)
