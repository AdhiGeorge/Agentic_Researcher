
import logging
from typing import Dict, List, Any, Optional
import json
import time
import os
from datetime import datetime

from src.config.system_config import SystemConfig
from src.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class ReportGeneratorAgent:
    """
    Agent responsible for generating final research reports.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.llm = LLMManager(config.llm)
        self.output_dir = os.path.join(config.output_path, "reports")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured report from the research results
        
        Args:
            results: The research results
            
        Returns:
            Dictionary with report sections
        """
        logger.info("Generating final report")
        
        # Extract key information from results
        query = results.get("query", "Unknown query")
        
        rag_results = results.get("rag_results", {})
        answer = rag_results.get("answer", "")
        insights = rag_results.get("insights", [])
        
        validation = results.get("validation", {})
        
        # Generate report sections
        summary = self._generate_summary(query, answer, insights)
        
        detailed_findings = self._generate_detailed_findings(
            query, 
            answer, 
            results.get("scraped_content", {}),
            results.get("search_results", [])
        )
        
        key_insights = self._process_insights(insights)
        
        sources = self._extract_sources(results)
        
        # Prepare final report
        report = {
            "summary": summary,
            "key_insights": key_insights,
            "detailed_findings": detailed_findings,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save report to file
        self._save_report(report, query)
        
        logger.info("Report generation complete")
        return report
    
    def _generate_summary(self, query: str, answer: str, insights: List[str]) -> str:
        """Generate an executive summary"""
        # This could use the LLM to generate a summary, but for the demo we'll create one directly
        
        if not answer:
            return f"The research on '{query}' did not yield comprehensive results."
        
        # Extract first paragraph or up to 500 chars from answer
        summary_text = answer.split("\n\n")[0] if "\n\n" in answer else answer[:500]
        
        # Add insights summary if available
        if insights:
            summary_text += "\n\nKey findings include: " + "; ".join(insights[:3]) + "."
        
        return summary_text
    
    def _generate_detailed_findings(
        self, 
        query: str, 
        answer: str, 
        scraped_content: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate detailed findings sections"""
        findings = []
        
        # Main findings section from the answer
        if answer:
            # Split the answer into paragraphs
            paragraphs = answer.split("\n\n")
            
            # Create main findings section
            findings.append({
                "title": "Main Findings",
                "content": answer
            })
            
            # If the answer is long enough, create additional sections
            if len(paragraphs) > 3:
                # Create a methodology section from relevant search info
                methodology = self._generate_methodology_section(query, search_results, scraped_content)
                findings.append(methodology)
                
                # Create an analysis section from the later paragraphs
                if len(paragraphs) > 4:
                    analysis = {
                        "title": "Detailed Analysis",
                        "content": "\n\n".join(paragraphs[2:])
                    }
                    findings.append(analysis)
        else:
            # Fallback if no answer is available
            findings.append({
                "title": "Research Results",
                "content": f"The research on '{query}' did not yield sufficient results for analysis."
            })
        
        return findings
    
    def _generate_methodology_section(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]],
        scraped_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a methodology section"""
        num_sources = scraped_content.get("num_sources", 0)
        total_tokens = scraped_content.get("total_tokens", 0)
        
        content = f"""
        This research was conducted using automated information retrieval and analysis techniques. 
        The system searched for information related to "{query}" across multiple sources.
        
        The research methodology included:
        
        1. Information gathering from {num_sources} distinct sources
        2. Content extraction and processing of approximately {total_tokens} tokens of text
        3. Retrieval-augmented generation to synthesize findings
        4. Fact checking and validation of key claims
        """
        
        if search_results:
            content += "\n\nTop sources by relevance included:\n"
            # Get top 3 sources by relevance
            top_sources = sorted(search_results, key=lambda x: x.get("relevance_score", 0), reverse=True)[:3]
            for i, source in enumerate(top_sources, 1):
                content += f"\n{i}. {source.get('title', 'Unknown source')}"
        
        return {
            "title": "Methodology",
            "content": content
        }
    
    def _process_insights(self, insights: List[str]) -> List[str]:
        """Process and clean up insights"""
        cleaned_insights = []
        
        for insight in insights:
            # Remove numbering if present
            cleaned = re.sub(r"^\d+\.\s*", "", insight) if re.match(r"^\d+\.\s*", insight) else insight
            
            # Clean up formatting
            cleaned = cleaned.strip()
            
            if cleaned:
                cleaned_insights.append(cleaned)
        
        return cleaned_insights
    
    def _extract_sources(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and format sources used in the research"""
        sources = []
        
        # Get search results if available
        search_results = results.get("search_results", [])
        
        if search_results:
            # Sort by relevance score
            sorted_results = sorted(search_results, key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Take the top N sources
            top_sources = sorted_results[:10]
            
            for source in top_sources:
                sources.append({
                    "title": source.get("title", "Unknown source"),
                    "url": source.get("url", ""),
                    "relevance_score": round(source.get("relevance_score", 0) * 10, 1),
                    "accessed_date": datetime.now().strftime("%Y-%m-%d"),
                    "summary": source.get("snippet", "No summary available")
                })
        
        return sources
    
    def _save_report(self, report: Dict[str, Any], query: str):
        """Save the report to a file"""
        import re
        
        # Create a filename from the query
        query_slug = re.sub(r"[^a-zA-Z0-9]", "_", query)[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{query_slug}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {file_path}")
        report["saved_to"] = file_path
