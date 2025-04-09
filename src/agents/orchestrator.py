
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

logger = logging.getLogger(__name__)

class TaskOrchestrator:
    """
    Orchestrates the entire research process by coordinating multiple agents.
    Implements a lightweight state machine to track the progress.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.memory = MemoryManager(config.memory_path)
        
        # Initialize agents
        self.search_agent = WebSearchAgent(config)
        self.scraper_agent = ScraperAgent(config)
        self.rag_agent = RAGAgent(config)
        self.code_generator = CodeGeneratorAgent(config)
        self.validator = ValidationAgent(config)
        self.report_generator = ReportGeneratorAgent(config)
        
        # State tracking
        self.state = "initialized"
        self.results = {}
        self.start_time = None
        self.end_time = None
    
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
            # Step 1: Web Search
            self._update_agent_status("orchestrator", "running")
            self._update_agent_status("search", "running")
            
            logger.info(f"Starting web search for: {query_data['query']}")
            search_results = self.search_agent.search(
                query=query_data["query"],
                depth=query_data.get("search_depth", 3),
                include_academic=query_data.get("include_academic", True)
            )
            
            self._update_agent_status("search", "completed")
            self.results["search_results"] = search_results
            self.state = "search_completed"
            
            # Step 2: Web Scraping
            self._update_agent_status("scraper", "running")
            
            logger.info(f"Scraping {len(search_results)} search results")
            scraped_content = self.scraper_agent.scrape(search_results)
            
            self._update_agent_status("scraper", "completed")
            self.results["scraped_content"] = {
                "num_sources": len(scraped_content),
                "total_tokens": sum(item.get("token_count", 0) for item in scraped_content)
            }
            self.state = "scraping_completed"
            
            # Step 3: RAG Processing
            self._update_agent_status("rag", "running")
            
            logger.info("Processing content with RAG engine")
            rag_results = self.rag_agent.process(
                query=query_data["query"],
                content=scraped_content
            )
            
            self._update_agent_status("rag", "completed")
            self.results["rag_results"] = rag_results
            self.state = "rag_completed"
            
            # Step 4: Generate code if requested
            if query_data.get("include_code", True):
                self._update_agent_status("code_generator", "running")
                
                logger.info("Generating code based on research")
                generated_code = self.code_generator.generate(
                    query=query_data["query"],
                    context=rag_results
                )
                
                self._update_agent_status("code_generator", "completed")
                self.results["generated_code"] = generated_code
                self.state = "code_generated"
                
                # Step 5: Validate results
                self._update_agent_status("validator", "running")
                
                logger.info("Validating generated content")
                validation_results = self.validator.validate(
                    results=self.results
                )
                
                self._update_agent_status("validator", "completed")
                self.results["validation"] = validation_results
                self.state = "validated"
            
            # Step 6: Generate final report
            self._update_agent_status("report_generator", "running")
            
            logger.info("Generating final report")
            report = self.report_generator.generate(self.results)
            
            self._update_agent_status("report_generator", "completed")
            
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
