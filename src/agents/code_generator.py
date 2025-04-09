
import logging
from typing import Dict, List, Any, Optional
import json
import time
import os

from src.config.system_config import SystemConfig
from src.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class CodeGeneratorAgent:
    """
    Agent responsible for generating code based on research results.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.llm = LLMManager(config.llm)
        self.output_dir = os.path.join(config.output_path, "code")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate code based on research results
        
        Args:
            query: The original research query
            context: The RAG results and context
            
        Returns:
            List of code blocks with descriptions
        """
        logger.info(f"Generating code for query: {query}")
        
        # Extract relevant information from context
        answer = context.get("answer", "")
        insights = context.get("insights", [])
        top_chunks = context.get("top_chunks", [])
        
        # Combine context information
        context_text = f"""
        Research Summary:
        {answer[:1000]}...
        
        Key Insights:
        {' '.join(insights)}
        
        Additional Context:
        {' '.join([chunk['text'][:200] + '...' for chunk in top_chunks[:3]])}
        """
        
        # Determine what types of code to generate
        code_blocks = []
        
        # Generate main code implementation
        implementation = self._generate_implementation(query, context_text)
        if implementation:
            code_blocks.append(implementation)
        
        # Generate data visualization if appropriate
        visualization = self._generate_visualization(query, context_text)
        if visualization:
            code_blocks.append(visualization)
        
        # Generate utility functions if appropriate
        utilities = self._generate_utilities(query, context_text)
        if utilities:
            code_blocks.append(utilities)
        
        # Save code blocks to files
        self._save_code_blocks(code_blocks, query)
        
        logger.info(f"Generated {len(code_blocks)} code blocks")
        return code_blocks
    
    def _generate_implementation(self, query: str, context: str) -> Optional[Dict[str, Any]]:
        """Generate main implementation code"""
        prompt = f"""
        Research Query: {query}
        
        Context Information:
        {context}
        
        Task: Generate Python code that implements a solution related to the research query.
        Your code should be well-structured, include comments, and use best practices.
        
        Create a main implementation that solves a key aspect of the research query.
        Include:
        1. Necessary imports
        2. Well-named functions and classes
        3. Docstrings explaining purpose and usage
        4. Example usage at the end
        
        Only respond with the Python code, nothing else.
        """
        
        # Get response from LLM
        code = self.llm.generate(prompt)
        
        if not code:
            return None
        
        # Get a title for this code block
        title_prompt = f"""
        Research Query: {query}
        
        Code:
        {code[:500]}...
        
        Task: Provide a short, descriptive title for this code implementation.
        The title should be concise (5-8 words) and describe what the code does.
        Only respond with the title, nothing else.
        """
        
        title = self.llm.generate(title_prompt).strip()
        
        # Get a description for this code block
        desc_prompt = f"""
        Research Query: {query}
        
        Code:
        {code[:500]}...
        
        Task: Provide a brief description (2-3 sentences) explaining what this code does
        and how it relates to the research query.
        Only respond with the description, nothing else.
        """
        
        description = self.llm.generate(desc_prompt).strip()
        
        return {
            "title": title or "Main Implementation",
            "description": description or "Implementation related to the research query.",
            "code": code,
            "language": "python",
            "type": "implementation"
        }
    
    def _generate_visualization(self, query: str, context: str) -> Optional[Dict[str, Any]]:
        """Generate data visualization code"""
        prompt = f"""
        Research Query: {query}
        
        Context Information:
        {context}
        
        Task: Generate Python code for a visualization related to the research query.
        Use matplotlib, seaborn, or plotly to create a visualization that illustrates
        an important aspect of the research topic.
        
        Your code should:
        1. Include sample data that represents the research domain
        2. Create a clear, informative visualization
        3. Include proper labels, title, and styling
        4. Have comments explaining the visualization choices
        
        Only respond with the Python code, nothing else.
        """
        
        # Get response from LLM
        code = self.llm.generate(prompt)
        
        if not code:
            return None
        
        return {
            "title": "Data Visualization",
            "description": "Visualization illustrating key aspects of the research topic.",
            "code": code,
            "language": "python",
            "type": "visualization"
        }
    
    def _generate_utilities(self, query: str, context: str) -> Optional[Dict[str, Any]]:
        """Generate utility functions"""
        prompt = f"""
        Research Query: {query}
        
        Context Information:
        {context}
        
        Task: Generate Python utility functions related to the research query.
        Create helper functions that would be useful for working with data or concepts
        from the research domain.
        
        Your utility code should:
        1. Include 2-4 useful helper functions
        2. Each function should have a clear purpose related to the research
        3. Include docstrings and type hints
        4. Demonstrate usage with example calls
        
        Only respond with the Python code, nothing else.
        """
        
        # Get response from LLM
        code = self.llm.generate(prompt)
        
        if not code:
            return None
        
        return {
            "title": "Utility Functions",
            "description": "Helper functions for working with concepts from the research domain.",
            "code": code,
            "language": "python",
            "type": "utilities"
        }
    
    def _save_code_blocks(self, code_blocks: List[Dict[str, Any]], query: str):
        """Save code blocks to files"""
        if not code_blocks:
            return
        
        # Create a directory for this query
        query_slug = "".join(c if c.isalnum() else "_" for c in query)[:30]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{query_slug}"
        
        output_dir = os.path.join(self.output_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each code block to a separate file
        for i, block in enumerate(code_blocks):
            file_name = None
            
            if block["type"] == "implementation":
                file_name = "main.py"
            elif block["type"] == "visualization":
                file_name = "visualization.py"
            elif block["type"] == "utilities":
                file_name = "utils.py"
            else:
                file_name = f"code_{i+1}.py"
            
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, 'w') as f:
                f.write(block["code"])
            
            logger.info(f"Saved code block to {file_path}")
            
            # Update the block with file path
            block["file_path"] = file_path
