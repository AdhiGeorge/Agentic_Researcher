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
    
    def _ensure_swarm_orchestrator(self):
        """Ensure the Swarm Orchestrator instance is initialized"""
        if self.swarm_orchestrator is None:
            self.swarm_orchestrator = SwarmOrchestrator()
            logger.info("Swarm Orchestrator initialized")
    
    async def research(self, query, project_name=None, use_cache=True):
        """Execute a research query"""
        self._ensure_swarm_orchestrator()
        
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
        
        # Start research in background
        research_task = asyncio.create_task(self.swarm_orchestrator.process_research_query(
            query=query,
            project_id=project_id,
            use_cache=use_cache
        ))
        
        # Show spinner while research is processing
        while not research_task.done():
            elapsed = time.time() - start_time
            spinner = spinner_chars[i % len(spinner_chars)]
            i += 1
            print(f"\rResearching {spinner} (elapsed: {elapsed:.1f}s)", end="")
            await asyncio.sleep(0.1)
        
        # Get results
        results = await research_task
        
        # Clear spinner line
        print("\r" + " " * 50 + "\r", end="")
        
        # Print summary
        print(f"\nResearch complete in {results.get('execution_time', 0):.2f} seconds")
        if results.get("from_cache", False):
            print("Results retrieved from cache")
        
        if "project_id" in results:
            print(f"Project ID: {results['project_id']}")
        
        # Print results
        if "results" in results and "answer" in results["results"]:
            print("\n=== Research Answer ===\n")
            print(results["results"]["answer"])
            
            # Check for code
            if "code" in results["results"]:
                print("\n=== Generated Code ===\n")
                print(results["results"]["code"])
        
        return results
    
    async def code(self, query, language="python", project_id=None):
        """Generate code for a specific task"""
        self._ensure_orka_swarm()
        
        # Enhance query with language context
        enhanced_query = f"Write {language} code for the following task: {query}"
        
        print(f"Generating {language} code...")        
        # Handle the same way as research but with code focus
        results = await self.swarm_orchestrator.process_research_query(
            query=enhanced_query,
            project_id=project_id,
            use_cache=True
        )
        
        # Print results
        if "results" in results and "answer" in results["results"]:
            print("\n=== Explanation ===\n")
            print(results["results"]["answer"])
            
            # Extract code blocks from the answer
            import re
            code_blocks = re.findall(r'```(?:' + language + r')?([\s\S]*?)```', results["results"]["answer"])
            
            if code_blocks:
                print("\n=== Generated Code ===\n")
                print(code_blocks[0].strip())
                
                # Save to file
                extension = ".py" if language == "python" else f".{language}"
                filename = f"generated_code{extension}"
                with open(filename, "w") as f:
                    f.write(code_blocks[0].strip())
                print(f"\nCode saved to {filename}")
            else:
                print("\nNo code block found in the response.")
        
        return results
    
    async def followup(self, query, project_id, action="answer"):
        """Ask a follow-up question"""
        self._ensure_orka_swarm()
        
        # Verify project exists
        project = self.sqlite_manager.get_project(project_id)
        if not project:
            print(f"Project with ID {project_id} not found")
            return {"error": "Project not found"}
        
        print(f"Processing follow-up for project {project_id}...")
        
        # Process follow-up
        results = await self.orka_swarm.handle_followup(
            query=query,
            project_id=project_id,
            action_type=action
        )
        
        # Print results
        if "results" in results and "response" in results["results"]:
            print(f"\n=== {action.capitalize()} Response ===\n")
            print(results["results"]["response"])
        
        return results
    
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
    """Main entry point for the CLI application"""
    args = parse_arguments()
    cli = AgenticResearcherCLI()
    
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
        logger.exception(f"Error in main application: {str(e)}")
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
