
import logging
from typing import Dict, List, Any, Optional
import time
import json
import os
from datetime import datetime

from src.config.system_config import SystemConfig
from src.agents.search_agent import WebSearchAgent
from src.agents.scraper_agent import ScraperAgent
from src.agents.rag_agent import RAGAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.validator import ValidationAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.memory.memory_manager import MemoryManager
from src.swarm.swarm_protocol import SwarmCoordinator, SwarmAgent, AgentStatus, Message

logger = logging.getLogger(__name__)

class TaskOrchestrator:
    """
    Orchestrates the entire research process by coordinating multiple agents.
    Implements a lightweight state machine and swarm coordination.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.memory = MemoryManager(config.memory_path)
        
        # Initialize swarm coordinator
        self.swarm = SwarmCoordinator(config.swarm)
        
        # Initialize agents
        self.search_agent = self._create_search_agent()
        self.scraper_agent = self._create_scraper_agent()
        self.rag_agent = self._create_rag_agent()
        self.code_generator = self._create_code_generator_agent()
        self.validator = self._create_validator_agent()
        self.report_generator = self._create_report_generator_agent()
        
        # Register agents with swarm
        self.swarm.register_agent(self.search_agent)
        self.swarm.register_agent(self.scraper_agent)
        self.swarm.register_agent(self.rag_agent)
        self.swarm.register_agent(self.code_generator)
        self.swarm.register_agent(self.validator)
        self.swarm.register_agent(self.report_generator)
        
        # State tracking
        self.state = "initialized"
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def _create_search_agent(self):
        """Create search agent with swarm integration"""
        agent = WebSearchAgentSwarm("search", self.swarm, self.config)
        return agent
    
    def _create_scraper_agent(self):
        """Create scraper agent with swarm integration"""
        agent = ScraperAgentSwarm("scraper", self.swarm, self.config)
        return agent
    
    def _create_rag_agent(self):
        """Create RAG agent with swarm integration"""
        agent = RAGAgentSwarm("rag", self.swarm, self.config)
        return agent
    
    def _create_code_generator_agent(self):
        """Create code generator agent with swarm integration"""
        agent = CodeGeneratorAgentSwarm("code_generator", self.swarm, self.config)
        return agent
    
    def _create_validator_agent(self):
        """Create validator agent with swarm integration"""
        agent = ValidationAgentSwarm("validator", self.swarm, self.config)
        return agent
    
    def _create_report_generator_agent(self):
        """Create report generator agent with swarm integration"""
        agent = ReportGeneratorAgentSwarm("report_generator", self.swarm, self.config)
        return agent
    
    def _update_agent_status(self, agent_name: str, status: str):
        """Update the status of an agent in the session state"""
        import streamlit as st
        if 'agent_status' in st.session_state:
            st.session_state.agent_status[agent_name] = status
    
    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to process a research query through the agent pipeline.
        Implements a state machine for the research workflow.
        """
        self.start_time = time.time()
        self.state = "started"
        self.results = {
            "query": query_data["query"],
            "timestamp": datetime.now().isoformat(),
            "research_area": query_data.get("research_area", "General")
        }
        
        try:
            # Update orchestrator status
            self._update_agent_status("orchestrator", "running")
            
            # Step 1: Web Search
            search_payload = {
                "query": query_data["query"],
                "depth": query_data.get("search_depth", 3),
                "include_academic": query_data.get("include_academic", True)
            }
            
            logger.info(f"Starting web search for: {query_data['query']}")
            
            # Run search agent
            self.swarm.run_agent("search", **search_payload)
            
            # Wait for search to complete
            self.swarm.wait_for_agents(["search"], timeout=90)
            
            # Get search results
            search_results = self.swarm.get_agent_result("search")
            if not search_results:
                raise Exception("Search agent failed to produce results")
            
            self.results["search_results"] = search_results
            self.state = "search_completed"
            
            # Step 2: Web Scraping
            scraper_payload = {
                "search_results": search_results
            }
            
            logger.info(f"Scraping {len(search_results)} search results")
            
            # Run scraper agent
            self.swarm.run_agent("scraper", **scraper_payload)
            
            # Wait for scraping to complete
            self.swarm.wait_for_agents(["scraper"], timeout=180)
            
            # Get scraped content
            scraped_content = self.swarm.get_agent_result("scraper")
            if not scraped_content:
                raise Exception("Scraper agent failed to produce results")
            
            self.results["scraped_content"] = {
                "num_sources": len(scraped_content),
                "total_tokens": sum(item.get("token_count", 0) for item in scraped_content)
            }
            self.state = "scraping_completed"
            
            # Step 3: RAG Processing
            rag_payload = {
                "query": query_data["query"],
                "content": scraped_content
            }
            
            logger.info("Processing content with RAG engine")
            
            # Run RAG agent
            self.swarm.run_agent("rag", **rag_payload)
            
            # Wait for RAG to complete
            self.swarm.wait_for_agents(["rag"], timeout=120)
            
            # Get RAG results
            rag_results = self.swarm.get_agent_result("rag")
            if not rag_results:
                raise Exception("RAG agent failed to produce results")
            
            self.results["rag_results"] = rag_results
            self.state = "rag_completed"
            
            # Step 4: Generate code if requested
            if query_data.get("include_code", True):
                code_generator_payload = {
                    "query": query_data["query"],
                    "context": rag_results
                }
                
                logger.info("Generating code based on research")
                
                # Run code generator agent
                self.swarm.run_agent("code_generator", **code_generator_payload)
                
                # Wait for code generation to complete
                self.swarm.wait_for_agents(["code_generator"], timeout=120)
                
                # Get code generation results
                generated_code = self.swarm.get_agent_result("code_generator")
                if not generated_code:
                    logger.warning("Code generator failed to produce results")
                    generated_code = []
                
                self.results["generated_code"] = generated_code
                self.state = "code_generated"
                
                # Step 5: Validate results
                validator_payload = {
                    "results": self.results
                }
                
                logger.info("Validating generated content")
                
                # Run validator agent
                self.swarm.run_agent("validator", **validator_payload)
                
                # Wait for validation to complete
                self.swarm.wait_for_agents(["validator"], timeout=90)
                
                # Get validation results
                validation_results = self.swarm.get_agent_result("validator")
                if not validation_results:
                    logger.warning("Validator failed to produce results")
                    validation_results = {"overall_score": 0.0, "checks": [], "issues": []}
                
                self.results["validation"] = validation_results
                self.state = "validated"
            
            # Step 6: Generate final report
            report_generator_payload = {
                "results": self.results
            }
            
            logger.info("Generating final report")
            
            # Run report generator agent
            self.swarm.run_agent("report_generator", **report_generator_payload)
            
            # Wait for report generation to complete
            self.swarm.wait_for_agents(["report_generator"], timeout=90)
            
            # Get report generation results
            report = self.swarm.get_agent_result("report_generator")
            if not report:
                logger.warning("Report generator failed to produce results")
                report = {"summary": "Failed to generate report", "key_insights": [], "detailed_findings": [], "sources": []}
            
            # Add report to results
            self.results.update(report)
            
            # Save results to disk
            self._save_results()
            
            # Update final state
            self.state = "completed"
            self._update_agent_status("orchestrator", "completed")
            
        except Exception as e:
            logger.error(f"Error during query processing: {str(e)}", exc_info=True)
            self.state = "error"
            self.results["error"] = str(e)
            self._update_agent_status("orchestrator", "error")
        
        self.end_time = time.time()
        self.results["processing_time"] = self.end_time - self.start_time
        
        # Add swarm visualization data
        self.results["swarm_visualization"] = self.swarm.visualize_swarm()
        
        return self.results
    
    def _save_results(self):
        """Save results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.output_path}/research_{timestamp}.json"
        
        os.makedirs(self.config.output_path, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        self.results["saved_to"] = filename


# Swarm Agent implementations for each agent type
class WebSearchAgentSwarm(SwarmAgent):
    def __init__(self, agent_id: str, swarm: SwarmCoordinator, config: SystemConfig):
        super().__init__(agent_id, swarm)
        self.config = config
        self.search_agent = WebSearchAgent(config)
        
        # Register message handlers
        self.register_handler("cancel", self._handle_cancel)
    
    def run(self, **kwargs):
        """Run the search agent"""
        self.update_status(AgentStatus.RUNNING)
        
        try:
            query = kwargs.get("query", "")
            depth = kwargs.get("depth", 3)
            include_academic = kwargs.get("include_academic", True)
            
            if not query:
                raise ValueError("No query provided")
            
            # Perform search
            search_results = self.search_agent.search(
                query=query,
                depth=depth,
                include_academic=include_academic
            )
            
            # Process any messages that arrived during search
            self.process_messages()
            
            self.update_status(AgentStatus.COMPLETED)
            return search_results
            
        except Exception as e:
            logger.error(f"Error in search agent: {str(e)}")
            self.update_status(AgentStatus.ERROR)
            self.error = str(e)
            return None
    
    def _handle_cancel(self, message: Message):
        """Handle cancel message"""
        logger.info(f"Search agent received cancel message: {message.payload}")
        # Implementation would cancel any ongoing tasks


class ScraperAgentSwarm(SwarmAgent):
    def __init__(self, agent_id: str, swarm: SwarmCoordinator, config: SystemConfig):
        super().__init__(agent_id, swarm)
        self.config = config
        self.scraper_agent = ScraperAgent(config)
        
        # Register message handlers
        self.register_handler("cancel", self._handle_cancel)
    
    def run(self, **kwargs):
        """Run the scraper agent"""
        self.update_status(AgentStatus.RUNNING)
        
        try:
            search_results = kwargs.get("search_results", [])
            
            if not search_results:
                raise ValueError("No search results provided")
            
            # Perform scraping
            scraped_content = self.scraper_agent.scrape(search_results)
            
            # Process any messages that arrived during scraping
            self.process_messages()
            
            self.update_status(AgentStatus.COMPLETED)
            return scraped_content
            
        except Exception as e:
            logger.error(f"Error in scraper agent: {str(e)}")
            self.update_status(AgentStatus.ERROR)
            self.error = str(e)
            return None
    
    def _handle_cancel(self, message: Message):
        """Handle cancel message"""
        logger.info(f"Scraper agent received cancel message: {message.payload}")
        # Implementation would cancel any ongoing tasks


class RAGAgentSwarm(SwarmAgent):
    def __init__(self, agent_id: str, swarm: SwarmCoordinator, config: SystemConfig):
        super().__init__(agent_id, swarm)
        self.config = config
        self.rag_agent = RAGAgent(config)
        
        # Register message handlers
        self.register_handler("cancel", self._handle_cancel)
    
    def run(self, **kwargs):
        """Run the RAG agent"""
        self.update_status(AgentStatus.RUNNING)
        
        try:
            query = kwargs.get("query", "")
            content = kwargs.get("content", [])
            
            if not query:
                raise ValueError("No query provided")
            
            if not content:
                raise ValueError("No content provided")
            
            # Process with RAG
            rag_results = self.rag_agent.process(query, content)
            
            # Process any messages that arrived during RAG processing
            self.process_messages()
            
            self.update_status(AgentStatus.COMPLETED)
            return rag_results
            
        except Exception as e:
            logger.error(f"Error in RAG agent: {str(e)}")
            self.update_status(AgentStatus.ERROR)
            self.error = str(e)
            return None
    
    def _handle_cancel(self, message: Message):
        """Handle cancel message"""
        logger.info(f"RAG agent received cancel message: {message.payload}")
        # Implementation would cancel any ongoing tasks


class CodeGeneratorAgentSwarm(SwarmAgent):
    def __init__(self, agent_id: str, swarm: SwarmCoordinator, config: SystemConfig):
        super().__init__(agent_id, swarm)
        self.config = config
        self.code_generator = CodeGeneratorAgent(config)
        
        # Register message handlers
        self.register_handler("cancel", self._handle_cancel)
    
    def run(self, **kwargs):
        """Run the code generator agent"""
        self.update_status(AgentStatus.RUNNING)
        
        try:
            query = kwargs.get("query", "")
            context = kwargs.get("context", {})
            
            if not query:
                raise ValueError("No query provided")
            
            if not context:
                raise ValueError("No context provided")
            
            # Generate code
            generated_code = self.code_generator.generate(query, context)
            
            # Process any messages that arrived during code generation
            self.process_messages()
            
            self.update_status(AgentStatus.COMPLETED)
            return generated_code
            
        except Exception as e:
            logger.error(f"Error in code generator agent: {str(e)}")
            self.update_status(AgentStatus.ERROR)
            self.error = str(e)
            return None
    
    def _handle_cancel(self, message: Message):
        """Handle cancel message"""
        logger.info(f"Code generator agent received cancel message: {message.payload}")
        # Implementation would cancel any ongoing tasks


class ValidationAgentSwarm(SwarmAgent):
    def __init__(self, agent_id: str, swarm: SwarmCoordinator, config: SystemConfig):
        super().__init__(agent_id, swarm)
        self.config = config
        self.validator = ValidationAgent(config)
        
        # Register message handlers
        self.register_handler("cancel", self._handle_cancel)
    
    def run(self, **kwargs):
        """Run the validator agent"""
        self.update_status(AgentStatus.RUNNING)
        
        try:
            results = kwargs.get("results", {})
            
            if not results:
                raise ValueError("No results provided")
            
            # Validate results
            validation_results = self.validator.validate(results)
            
            # Process any messages that arrived during validation
            self.process_messages()
            
            self.update_status(AgentStatus.COMPLETED)
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in validator agent: {str(e)}")
            self.update_status(AgentStatus.ERROR)
            self.error = str(e)
            return None
    
    def _handle_cancel(self, message: Message):
        """Handle cancel message"""
        logger.info(f"Validator agent received cancel message: {message.payload}")
        # Implementation would cancel any ongoing tasks


class ReportGeneratorAgentSwarm(SwarmAgent):
    def __init__(self, agent_id: str, swarm: SwarmCoordinator, config: SystemConfig):
        super().__init__(agent_id, swarm)
        self.config = config
        self.report_generator = ReportGeneratorAgent(config)
        
        # Register message handlers
        self.register_handler("cancel", self._handle_cancel)
    
    def run(self, **kwargs):
        """Run the report generator agent"""
        self.update_status(AgentStatus.RUNNING)
        
        try:
            results = kwargs.get("results", {})
            
            if not results:
                raise ValueError("No results provided")
            
            # Generate report
            report = self.report_generator.generate(results)
            
            # Process any messages that arrived during report generation
            self.process_messages()
            
            self.update_status(AgentStatus.COMPLETED)
            return report
            
        except Exception as e:
            logger.error(f"Error in report generator agent: {str(e)}")
            self.update_status(AgentStatus.ERROR)
            self.error = str(e)
            return None
    
    def _handle_cancel(self, message: Message):
        """Handle cancel message"""
        logger.info(f"Report generator agent received cancel message: {message.payload}")
        # Implementation would cancel any ongoing tasks
