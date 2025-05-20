"""Reporter Agent for Agentic Researcher

This agent is responsible for generating comprehensive reports based on research results.
It processes research data, structures findings, and creates well-formatted reports.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Use imports that work for both direct execution and when imported as a module
from src.utils.config import config
from src.db.sqlite_manager import SQLiteManager
from src.db.qdrant_manager import QdrantManager
from src.utils.openai_client import AzureOpenAIClient

class ReporterAgent:
    """
    Reporter Agent that generates comprehensive research reports
    
    This agent processes research data, structures findings, and creates
    well-formatted reports with citations and analysis.
    """
    
    def __init__(self):
        """Initialize the ReporterAgent"""
        self.name = "reporter"
        
        # Initialize database connections
        self.sqlite_manager = SQLiteManager()
        self.vector_db = QdrantManager()
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAIClient()
    
    def generate_report(self, query: str, research_results: Dict[str, Any], project_id: int,
                      format_type: str = "markdown",
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report based on research results
        
        Args:
            query: Original user query
            research_results: Results from the research agent
            project_id: Project ID
            format_type: Output format (markdown, html, or text)
            context: Additional context information
            
        Returns:
            Dict: Generated report with metadata
        """
        # Log the start of report generation
        self.sqlite_manager.save_agent_state(
            project_id=project_id,
            agent_type="reporter",
            state_data={
                "status": "generating",
                "query": query,
                "format": format_type
            }
        )
        
        try:
            # Extract content from research results
            web_content = research_results.get("web_content", [])
            pdf_content = research_results.get("pdf_content", [])
            search_results = research_results.get("search_results", [])
            
            # Combine content chunks
            all_chunks = []
            
            # Process web content
            for item in web_content:
                if isinstance(item, dict):
                    content = item.get("content", "")
                    source = item.get("url", "") or item.get("source", "")
                    all_chunks.append({"content": content, "source": source, "type": "web"})
                elif isinstance(item, str):
                    all_chunks.append({"content": item, "source": "Unknown web source", "type": "web"})
            
            # Process PDF content
            for item in pdf_content:
                if isinstance(item, dict):
                    content = item.get("content", "")
                    source = item.get("source", "") or "PDF document"
                    all_chunks.append({"content": content, "source": source, "type": "pdf"})
                elif isinstance(item, str):
                    all_chunks.append({"content": item, "source": "Unknown PDF source", "type": "pdf"})
            
            # Process search results for additional context
            search_context = ""
            for result in search_results:
                if isinstance(result, dict):
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    url = result.get("link", "") or result.get("url", "")
                    search_context += f"Title: {title}\nSnippet: {snippet}\nURL: {url}\n\n"
            
            # Generate content for the report prompt
            chunks_content = ""
            sources = set()
            
            for chunk in all_chunks:
                content = chunk.get("content", "")[:1000]  # Limit chunk size
                source = chunk.get("source", "")
                chunks_content += f"SOURCE: {source}\n\nCONTENT: {content}\n\n"
                sources.add(source)
            
            # Prepare sources for citation
            sources_list = "\n".join([f"- {source}" for source in sources if source])
            
            # Determine report structure based on format
            structure_instructions = ""
            if format_type.lower() == "markdown":
                structure_instructions = """Format the report using Markdown. Include:
- # for main headings
- ## for section headings
- ### for subsection headings
- Bullet points with -
- Numbered lists with 1., 2., etc.
- *italic* or **bold** for emphasis
- [text](URL) for hyperlinks
- > for blockquotes
- ```code blocks``` for code
- Tables using | syntax"""
            elif format_type.lower() == "html":
                structure_instructions = """Format the report using HTML. Include proper HTML structure with:
- <h1>, <h2>, <h3> for headings
- <p> for paragraphs
- <ul>/<li> for bullet points
- <ol>/<li> for numbered lists
- <em> or <strong> for emphasis
- <a href="URL">text</a> for hyperlinks
- <blockquote> for quotes
- <pre><code> for code blocks
- <table>, <tr>, <td> for tables"""
            else:  # text
                structure_instructions = """Format the report as plain text. Use:
- ALL CAPS for main headings
- Underlines (======) for section headings
- Indentation for structure
- Asterisks * for bullet points
- Numbers for ordered lists
- URLs in full form: https://...
- Clear paragraph breaks"""
            
            # Build the report generation prompt
            prompt = f"""You are an expert research report writer. Generate a comprehensive research report based on the following query and research findings.

QUERY: {query}

RESEARCH FINDINGS:
{chunks_content}

ADDITIONAL SEARCH CONTEXT:
{search_context}

REPORT FORMAT REQUIREMENTS:
{structure_instructions}

Report requirements:
1. Create a structured report with clear sections including an executive summary, introduction, main findings, analysis, and conclusion
2. Include proper citations to the sources provided
3. Synthesize information from multiple sources
4. Be objective and analytical in your presentation
5. Include a references/sources section at the end
6. Keep the report concise but comprehensive, focusing on addressing the query
7. If there are mathematical formulas or code examples relevant to the query, include them properly formatted

SOURCES TO CITE:
{sources_list}
"""

            # Call Azure OpenAI API
            response = self.openai_client.generate_completion(
                messages=[
                    {"role": "system", "content": "You are an expert research report writer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000
            )
            
            # Prepare the report result
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_result = {
                "title": f"Research Report: {query}",
                "query": query,
                "content": response,
                "format": format_type,
                "sources": list(sources),
                "timestamp": timestamp,
                "project_id": project_id
            }
            
            # Save report to database
            report_id = self.sqlite_manager.save_report(
                project_id=project_id,
                title=report_result["title"],
                content=report_result["content"],
                format_type=format_type,
                sources=json.dumps(list(sources)),
                query=query
            )
            
            report_result["report_id"] = report_id
            
            # Log completion of report generation
            self.sqlite_manager.save_agent_state(
                project_id=project_id,
                agent_type="reporter",
                state_data={
                    "status": "completed",
                    "report_id": report_id,
                    "word_count": len(response.split())
                }
            )
            
            return report_result
            
        except Exception as e:
            # Log error
            self.sqlite_manager.save_agent_state(
                project_id=project_id,
                agent_type="reporter",
                state_data={
                    "status": "error",
                    "error": str(e)
                }
            )
            
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    def summarize_report(self, report_content: str, max_length: int = 500) -> str:
        """
        Generate a concise summary of a research report
        
        Args:
            report_content: Full report content
            max_length: Maximum summary length in words
            
        Returns:
            str: Concise report summary
        """
        prompt = f"""You are an expert research summarizer. Create a concise summary of the following research report, highlighting the most important findings and conclusions.

REPORT CONTENT:
{report_content[:7000]}  # Truncated to fit within token limits

Provide a concise summary in approximately {max_length} words. Focus on the key findings, insights, and conclusions while maintaining clarity and coherence.
"""

        # Call Azure OpenAI API
        response = self.openai_client.generate_completion(
            messages=[
                {"role": "system", "content": "You are an expert research summarizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        return response
    
    def extract_key_insights(self, report_content: str, num_insights: int = 5) -> List[str]:
        """
        Extract key insights from a research report
        
        Args:
            report_content: Full report content
            num_insights: Number of key insights to extract
            
        Returns:
            List[str]: List of key insights
        """
        prompt = f"""You are an expert research analyst. Extract {num_insights} key insights from the following research report. These should be the most important discoveries, conclusions, or findings.

REPORT CONTENT:
{report_content[:7000]}  # Truncated to fit within token limits

Extract exactly {num_insights} key insights from this report. Format your response as a JSON array of strings with each insight clearly articulated in a complete sentence.
"""

        # Call Azure OpenAI API
        response = self.openai_client.generate_completion(
            messages=[
                {"role": "system", "content": "You are an expert research analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse insights from response
        try:
            insights = json.loads(response)
            if isinstance(insights, list):
                return insights
            # If response is a JSON object with insights key
            elif isinstance(insights, dict) and "insights" in insights:
                return insights["insights"]
            else:
                # Fallback parsing
                return ["Failed to parse insights properly"]
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response)
            if json_match:
                try:
                    insights = json.loads(json_match.group(1))
                    if isinstance(insights, list):
                        return insights
                    elif isinstance(insights, dict) and "insights" in insights:
                        return insights["insights"]
                except:
                    pass
            
            # If all parsing fails, extract lines that look like insights
            lines = response.strip().split('\n')
            extracted_insights = []
            for line in lines:
                # Look for numbered or bullet point lines
                if re.match(r'^\d+\.\s+|^\*\s+|^-\s+', line.strip()):
                    # Remove the numbering/bullet and trim
                    insight = re.sub(r'^\d+\.\s+|^\*\s+|^-\s+', '', line.strip())
                    if insight:
                        extracted_insights.append(insight)
            
            if extracted_insights:
                return extracted_insights[:num_insights]
            else:
                # Last resort: split by periods and take the first few sentences
                sentences = re.split(r'(?<=[.!?])\s+', response.strip())
                return [s for s in sentences[:num_insights] if len(s) > 10]


# Example usage
if __name__ == "__main__":
    print("\n===== ReporterAgent Initialization =====\n")
    
    # Create a simplified version of ReporterAgent that doesn't use SQLiteManager
    class SimpleReporterAgent:
        def __init__(self):
            self.name = "reporter"
            # Skip SQLiteManager initialization - just use the OpenAI client
            from src.utils.openai_client import AzureOpenAIClient
            self.openai_client = AzureOpenAIClient()
            print("Initialized simplified ReporterAgent for testing")
        
        def extract_key_insights(self, report_content, num_insights=5):
            print(f"\nExtracting {num_insights} key insights from report content...")
            # Return some dummy insights as a test
            insights = [
                "The VIX measures market's expectation of future volatility",
                "It is derived from S&P 500 options prices",
                "The formula uses weighted average of out-of-the-money option prices",
                "VIX is often referred to as the 'fear index'",
                "It provides a 30-day expectation of volatility"
            ]
            return insights[:num_insights]
    
    # Use the simplified agent
    reporter_agent = SimpleReporterAgent()
    
    # Example of what the reporter would analyze
    sample_report = """
    # Volatility Index (VIX) Analysis
    
    ## Introduction
    The Volatility Index (VIX), often referred to as the 'fear index', measures the market's expectation of future volatility. 
    It is derived from S&P 500 options prices. The higher the VIX, the greater the expected market volatility.
    
    ## Mathematical Formula
    The formula for calculating VIX involves measuring the implied volatility of a range of S&P 500 options, both calls and puts. 
    It specifically uses a weighted average of out-of-the-money option prices to derive an expectation of volatility over the next 30 days.
    """
    
    # Extract insights
    insights = reporter_agent.extract_key_insights(sample_report)
    
    print("\nExample Report Content:\n")
    print(sample_report[:300] + "...\n")  # Print first 300 chars
    
    print("Key Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
