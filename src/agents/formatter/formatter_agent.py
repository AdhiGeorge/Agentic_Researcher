"""Formatter Agent for Agentic Researcher

This module implements the Formatter agent that processes and structures
raw content from scraped sources into well-organized research data.
"""

import os
import sys
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.agents.base_agent import BaseAgent
from src.utils.config import Config as config

class FormatterAgent(BaseAgent):
    """Formatter agent that processes and structures raw content.
    
    The Formatter agent transforms the scraped content into well-structured,
    readable formats suitable for analysis and presentation.
    
    Attributes:
        config (ConfigLoader): Configuration loader
    """
    
    def __init__(self):
        """Initialize the FormatterAgent."""
        super().__init__(name="Formatter", description="Formats and structures raw content")
        # Using the global config object imported at the top of the file
        self.config = config
        self.logger.info("FormatterAgent initialized")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw content and format it for better readability and structure.
        
        Args:
            input_data (Dict[str, Any]): Input data containing raw content to format
            
        Returns:
            Dict[str, Any]: The formatted content and metadata
        """
        raw_content = input_data.get("raw_content", [])
        query = input_data.get("query", "")
        keywords = input_data.get("keywords", [])
        
        if not raw_content:
            self.logger.error("No content provided to FormatterAgent")
            raise ValueError("No content provided to FormatterAgent")
        
        self.log_activity(f"Formatting content for query: {query}")
        
        # Format the content
        formatted_content = self.format_content(raw_content, query, keywords)
        
        # Update agent state
        self.update_state({
            "last_query": query,
            "last_formatted": datetime.now().isoformat()
        })
        
        return {
            "query": query,
            "keywords": keywords,
            "formatted_content": formatted_content,
            "source_count": len(raw_content),
            "formatted_at": datetime.now().isoformat()
        }
    
    def format_content(self, raw_content: List[Dict[str, Any]], query: str = "", keywords: List[str] = None) -> Dict[str, Any]:
        """Format raw content into structured research data.
        
        Args:
            raw_content (List[Dict[str, Any]]): The raw content to format
            query (str, optional): The original query
            keywords (List[str], optional): Keywords to highlight
            
        Returns:
            Dict[str, Any]: Formatted and structured content
        """
        self.log_activity(f"Formatting {len(raw_content)} content items")
        keywords = keywords or []
        
        # Extract text content from each item
        all_text = ""
        sources = []
        
        for item in raw_content:
            text = item.get("text", "")
            url = item.get("url", "")
            title = item.get("title", "")
            
            if text:
                all_text += f"\n\n{text}"
                
                # Add source metadata
                sources.append({
                    "url": url,
                    "title": title,
                    "snippet": text[:200] + "..." if len(text) > 200 else text
                })
        
        # Use Azure OpenAI to summarize and structure the content
        structured_content = self._structure_with_llm(all_text, query, keywords)
        
        # Combine the formatted content with metadata
        return {
            "structured_content": structured_content,
            "sources": sources,
            "word_count": len(all_text.split()),
            "source_count": len(sources)
        }
    
    def _structure_with_llm(self, text: str, query: str, keywords: List[str]) -> str:
        """Use Azure OpenAI to structure and format the content.
        
        Args:
            text (str): The combined text to structure
            query (str): The original query
            keywords (List[str]): Keywords to highlight
            
        Returns:
            str: Structured and formatted content
        """
        self.log_activity("Using LLM to structure content")
        
        # Truncate text if too long (context window limitation)
        max_tokens = 60000  # Conservative limit
        approx_token_count = len(text.split())
        
        if approx_token_count > max_tokens:
            self.logger.warning(f"Text too long ({approx_token_count} words), truncating")
            words = text.split()
            text = " ".join(words[:max_tokens]) + "\n\n[Content truncated due to length...]"
        
        try:
            # Get Azure OpenAI credentials from config
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
            
            # Create the system prompt for formatting
            system_prompt = f"""
            You are a research assistant that formats and structures information.
            
            You will receive raw text content scraped from various web sources related to this query: "{query}"
            
            Your task is to:
            1. Organize and structure this information into a comprehensive, well-formatted research document
            2. Include relevant sections with clear headings
            3. Eliminate redundancy and irrelevant information
            4. Highlight key concepts and findings
            5. Include a summary section that synthesizes the main points
            6. Format the output using Markdown
            7. Pay special attention to these keywords: {', '.join(keywords)}
            
            The output should be factual, well-structured, and comprehensive. Do not add speculative information.
            Cite sources where appropriate using footnote references [n].
            """
            
            # User prompt with the content
            user_prompt = f"Here is the raw content to format and structure:\n\n{text}"
            
            # Make API call to Azure OpenAI
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more structured output
                max_tokens=4000
            )
            
            structured_content = response.choices[0].message.content.strip()
            self.log_activity("Successfully structured content with LLM")
            return structured_content
            
        except Exception as e:
            self.logger.error(f"Error structuring content with LLM: {str(e)}")
            # Fallback to basic formatting if LLM fails
            return self._basic_format(text, query, keywords)
    
    def _basic_format(self, text: str, query: str, keywords: List[str]) -> str:
        """Basic formatting when LLM is unavailable.
        
        Args:
            text (str): The text to format
            query (str): The original query
            keywords (List[str]): Keywords to highlight
            
        Returns:
            str: Basic formatted text
        """
        self.log_activity("Using basic formatting as fallback")
        
        # Create sections based on paragraphs
        paragraphs = text.split("\n\n")
        
        # Basic structure with sections
        formatted = f"# Research Results for: {query}\n\n"
        formatted += "## Summary\n\n"
        formatted += "This document contains research findings from multiple sources.\n\n"
        
        # Add content sections
        formatted += "## Main Content\n\n"
        
        # Add paragraphs (limited to reasonable number)
        for i, para in enumerate(paragraphs[:50]):
            if len(para.strip()) > 100:  # Only add substantial paragraphs
                formatted += f"### Section {i+1}\n\n"
                formatted += para.strip() + "\n\n"
        
        # Highlight keywords
        if keywords:
            formatted += "## Key Concepts\n\n"
            for keyword in keywords:
                formatted += f"- **{keyword}**\n"
            formatted += "\n"
        
        # Add timestamp
        formatted += f"\n\n---\nResearch compiled on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return formatted


if __name__ == "__main__":
    # Example usage
    formatter = FormatterAgent()
    
    try:
        # Sample raw content for testing
        sample_content = [
            {
                "url": "https://example.com/page1",
                "title": "Example Page 1",
                "text": "This is sample text from the first page. It contains information about the topic."
            },
            {
                "url": "https://example.com/page2",
                "title": "Example Page 2",
                "text": "This is sample text from the second page. It has different information about the same topic."
            }
        ]
        
        # Test query and keywords
        query = "Sample research topic"
        keywords = ["sample", "information", "topic"]
        
        # Format the content
        formatted = formatter.format_content(sample_content, query, keywords)
        
        print("\nFormatted Content:")
        print(json.dumps(formatted, indent=2))
        
        # Test with process method
        result = formatter.process({
            "raw_content": sample_content,
            "query": query,
            "keywords": keywords
        })
        
        print("\nProcess Result:")
        print(f"Query: {result['query']}")
        print(f"Source count: {result['source_count']}")
        print(f"Formatted at: {result['formatted_at']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
