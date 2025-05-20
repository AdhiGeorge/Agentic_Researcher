"""Coder Agent for Agentic Researcher
Generates code based on research results and requirements
"""
import json
import os
import sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import re
from typing import Dict, List, Any, Optional, Tuple

from src.utils.config import config
from src.db.sqlite_manager import SQLiteManager
from src.db.qdrant_manager import QdrantManager
from src.utils.openai_client import AzureOpenAIClient
from src.utils.response_adapter import extract_openai_response_content, safe_parse_json

class CoderAgent:
    """
    Coder Agent that generates code based on research findings
    Uses Azure OpenAI to write, analyze, and explain code
    """
    
    def __init__(self, use_db=True):
        # Load configuration
        self.config = config
        
        # Initialize databases if use_db is True (otherwise mock them for standalone execution)
        if use_db:
            # SQL Logger for tracking state - can cause errors with semantic_hash column
            try:
                self.logger = SQLiteManager()
            except Exception as e:
                print(f"Warning: Failed to initialize SQLiteManager: {e}")
                # Create a mock logger with a save_agent_state method
                class MockLogger:
                    def save_agent_state(self, **kwargs):
                        print(f"Mock logging: {kwargs}")
                self.logger = MockLogger()
            
            # Vector database for retrieving research data
            try:
                self.vector_db = QdrantManager(collection_name="preprocessed_chunks")
            except Exception as e:
                print(f"Warning: Failed to initialize QdrantManager: {e}")
                # Create a mock vector_db with a search method
                class MockVectorDB:
                    def search(self, **kwargs):
                        print(f"Mock vector search with: {kwargs}")
                        return []
                self.vector_db = MockVectorDB()
        else:
            # Create mock objects for standalone execution
            class MockLogger:
                def save_agent_state(self, **kwargs):
                    print(f"Mock logging: {kwargs}")
            self.logger = MockLogger()
            
            class MockVectorDB:
                def search(self, **kwargs):
                    print(f"Mock vector search with: {kwargs}")
                    return []
            self.vector_db = MockVectorDB()
        
        # Get the Azure OpenAI client
        self.openai_client = AzureOpenAIClient()
        # No longer need to access client directly - will use generate_completion method
        
        # Azure OpenAI model for coding
        self.model = self.config.azure_openai_deployment
    
    def create_prompt(self, query: str, plan: Dict[str, Any], 
                     relevant_chunks: List[Dict[str, Any]], 
                     requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Create prompt for the coder agent
        
        Args:
            query: Original user query
            plan: Research plan from planner agent
            relevant_chunks: Relevant text chunks from research
            requirements: Optional coding requirements
            
        Returns:
            str: Full prompt for the coder
        """
        # Extract relevant information from the plan
        objective = plan.get("objective", "")
        
        # Extract content from chunks
        chunk_texts = []
        for chunk in relevant_chunks:
            content = chunk.get("content", "")
            source = chunk.get("metadata", {}).get("url", "Unknown source")
            chunk_texts.append(f"SOURCE: {source}\n\n{content}\n")
        
        chunk_content = "\n---\n".join(chunk_texts)
        
        # Extract coding requirements if provided
        language = "Python"  # Default
        libraries = []
        functionality = []
        
        if requirements:
            language = requirements.get("language", language)
            libraries = requirements.get("libraries", [])
            functionality = requirements.get("functionality", [])
        
        libraries_str = "\n".join([f"- {lib}" for lib in libraries])
        functionality_str = "\n".join([f"- {func}" for func in functionality])
        
        # Base prompt template
        prompt = f"""You are an expert {language} developer. Your task is to write code based on research findings and requirements.

ORIGINAL QUERY: {query}

OBJECTIVE: {objective}

RESEARCH FINDINGS:
{chunk_content}

CODING REQUIREMENTS:
- Language: {language}
- Libraries/Frameworks:
{libraries_str if libraries else "- Use appropriate libraries based on requirements"}
- Required Functionality:
{functionality_str if functionality else "- Implement functionality to meet the objective"}

Your task:
1. Analyze the research findings to understand the requirements
2. Write clean, well-structured, and well-documented {language} code
3. Include appropriate error handling and edge cases
4. Explain your implementation choices in comments
5. Provide instructions on how to use the code

Format your response as a JSON object:
{{
    "explanation": "Brief explanation of the code's purpose and approach",
    "files": [
        {{
            "file_name": "filename.{language.lower()}",
            "content": "Full source code including docstrings and comments"
        }},
        ...
    ],
    "usage_instructions": "How to use the code, including any setup or dependencies"
}}
"""
        
        return prompt
    
    def execute(self, query: str, plan: Dict[str, Any], chunked_results: Dict[str, Any], 
                project_id: int, requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the coder agent to generate code
        
        Args:
            query: Original user query
            plan: Research plan from planner agent
            chunked_results: Results from chunker agent
            project_id: Project ID
            requirements: Optional coding requirements
            
        Returns:
            Dict: Generated code and explanations
        """
        # Log the agent state - starting code generation
        self.logger.save_agent_state(
            project_id=project_id,
            agent_type="coder",
            state_data={"status": "generating"}
        )
        
        try:
            # Get relevant chunks from the chunked results
            chunks = chunked_results.get("chunks", [])
            
            if not chunks:
                # If no chunks in results, get them directly from vector DB
                chunks = self._get_relevant_chunks(query, project_id)
            
            # Create the prompt
            prompt = self.create_prompt(query, plan, chunks, requirements)
            
            # Make API call to Azure OpenAI using the client's generate_completion method
            messages = [
                {"role": "system", "content": "You are an expert software developer."},
                {"role": "user", "content": prompt}
            ]
            
            # Use the generate_completion method which handles the API call
            response = self.openai_client.generate_completion(
                messages=messages,
                model=self.model,
                temperature=0.2,
                max_tokens=3000,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Process the response - generate_completion already returns content as string
            content = response
            
            try:
                code_result = safe_parse_json(content)
            except json.JSONDecodeError:
                # If direct parsing fails, extract code files manually
                code_result = self._extract_code_files(content)
            
            # Log the agent state - completed code generation
            self.logger.save_agent_state(
                project_id=project_id,
                agent_type="coder",
                state_data={"status": "completed", 
                           "files_generated": len(code_result.get("files", []))}
            )
            
            return code_result
        
        except Exception as e:
            # Log the agent state - error
            self.logger.save_agent_state(
                project_id=project_id,
                agent_type="coder",
                state_data={"status": "error", "error": str(e)}
            )
            
            raise Exception(f"Code generation failed: {str(e)}")
    
    def _get_relevant_chunks(self, query: str, project_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get relevant chunks for the query
        
        Args:
            query: User query
            project_id: Project ID
            limit: Maximum number of chunks to retrieve
            
        Returns:
            List[Dict]: List of relevant chunks
        """
        try:
            # Retrieve relevant chunks from vector DB
            results = self.vector_db.search(
                query=query,
                filter_dict={"project_id": project_id},
                limit=limit
            )
            return results
        except Exception as e:
            print(f"Error retrieving chunks: {str(e)}")
            return []
    
    def _extract_code_files(self, content: str):
        """
        Extract code files from non-JSON response
        
        Args:
            content: Response content
            
        Returns:
            Dict: Structured code results
        """
        # Initialize result structure
        result = {
            "explanation": "",
            "files": [],
            "usage_instructions": ""
        }
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:(?P<lang>\w+)\n)?(?P<code>.*?)```', content, re.DOTALL)
        
        # Extract filenames and content
        files_info = []
        current_file = None
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Look for file name patterns
            file_match = re.match(r'^(?:file|filename):\s*["\']*([\w\.-]+)["\']*\s*$', line, re.IGNORECASE)
            if file_match:
                current_file = file_match.group(1)
                continue
                
            # Alternative file pattern: heading with .py or similar extension
            if line.startswith('#') and ('.' in line) and not current_file:
                file_match = re.search(r'["\']*([\w\.-]+\.[a-z]{1,5})["\']*', line)
                if file_match:
                    current_file = file_match.group(1)
                    continue
        
        # Process code blocks
        for lang, code in code_blocks:
            # If we've identified a filename for this code block, use it
            if current_file:
                file_name = current_file
                current_file = None  # Reset for next block
            else:
                # If no filename found, create a generic one based on language
                if lang and lang.strip():
                    extension = lang.strip().lower()
                    # Map common language names to file extensions
                    ext_map = {"python": "py", "javascript": "js", "typescript": "ts", "html": "html", "css": "css"}
                    ext = ext_map.get(extension, extension)
                    file_name = f"main.{ext}"
                else:
                    # Default to Python if no language specified
                    file_name = "main.py"
            
            # Clean up code and add to files list
            clean_code = code.strip()
            files_info.append({"file_name": file_name, "content": clean_code})
        
        # Extract explanation (typically at the beginning)
        explanation_match = re.search(r'^(.*?)(?=```|#|import|from|class|def)', content, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""    
        
        # If no explanation found but there are paragraphs, use the first one
        if not explanation and '\n\n' in content:
            first_para = content.split('\n\n')[0].strip()
            if first_para and not first_para.startswith('```'):
                explanation = first_para
        
        # Extract usage instructions (typically at the end)
        usage_match = re.search(r'(?:usage|instructions|how to use)[:\s]+(.*?)(?:\n\n|\n#|\Z)', content, re.IGNORECASE | re.DOTALL)
        usage = usage_match.group(1).strip() if usage_match else "See code comments for usage instructions."
        
        return {
            "explanation": explanation,
            "files": files_info,
            "usage_instructions": usage
        }
    
    def refine_code(self, code_result: Dict[str, Any], feedback: str, project_id: int) -> Dict[str, Any]:
        """
        Refine code based on feedback
        
        Args:
            code_result: Original code generated
            feedback: User feedback
            project_id: Project ID
            
        Returns:
            Dict: Refined code
        """
        # Log the agent state - starting refinement
        self.logger.save_agent_state(
            project_id=project_id,
            agent_type="coder",
            state_data={"status": "refining", "feedback": feedback}
        )
        
        try:
            # Convert code files to text for the prompt
            code_files_text = ""
            for file in code_result.get("files", []):
                name = file.get("file_name", "unknown")
                content = file.get("content", "")
                code_files_text += f"FILE: {name}\n```\n{content}\n```\n\n"
            
            # Create refinement prompt
            refine_prompt = f"""You previously generated the following code:

{code_files_text}

EXPLANATION: {code_result.get('explanation', '')}

USAGE INSTRUCTIONS: {code_result.get('usage_instructions', '')}

The user has provided the following feedback:

{feedback}

Please refine the code based on this feedback. Maintain the same JSON structure in your response:
{{
    "explanation": "...",
    "files": [
        {{
            "file_name": "...",
            "content": "..."
        }},
        ...
    ],
    "usage_instructions": "..."
}}

Make specific changes to address the feedback while preserving the overall structure.
"""
            
            # Make API call to Azure OpenAI using the client's generate_completion method
            messages = [
                {"role": "system", "content": "You are an expert software developer."},
                {"role": "user", "content": refine_prompt}
            ]
            
            # Use the generate_completion method which handles the API call
            response = self.openai_client.generate_completion(
                messages=messages,
                model=self.model,
                temperature=0.2,
                max_tokens=3000,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Process the response - generate_completion already returns content as string
            content = response
            
            try:
                refined_code = json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from markdown
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    refined_code = json.loads(json_match.group(1))
                else:
                    # Extract code files manually if JSON parsing fails
                    refined_code = self._extract_code_files(content)
            
            # Log the agent state - completed refinement
            self.logger.save_agent_state(
                project_id=project_id,
                agent_type="coder",
                state_data={"status": "refinement_completed", 
                           "files_generated": len(refined_code.get("files", []))}
            )
            
            return refined_code
        
        except Exception as e:
            # Log the agent state - error
            self.logger.save_agent_state(
                project_id=project_id,
                agent_type="coder",
                state_data={"status": "refinement_error", "error": str(e)}
            )
            
            raise Exception(f"Code refinement failed: {str(e)}")


# Example usage
if __name__ == "__main__":
    print("\n===== CoderAgent Example =====\n")
    print("Initializing CoderAgent...")
    
    # Initialize CoderAgent with use_db=False to bypass database initialization
    coder = CoderAgent(use_db=False)
    
    # Example research data for generating code
    example_plan = {
        "objective": "Calculate volatility index (VIX) using standard formula",
        "steps": [
            "Understand the VIX formula",
            "Implement the calculation in Python",
            "Add error handling and documentation"
        ]
    }
    
    # Example chunks that would come from research
    example_chunks = {
        "chunks": [
            {
                "content": "The VIX formula involves calculating the weighted average of S&P 500 option prices. "
                          "It uses the formula: VIX = 100 * sqrt(T * sum((delta K_i / K_i^2) * e^(rT) * Q(K_i)))",
                "metadata": {"url": "https://example.com/vix-formula"}
            },
            {
                "content": "To calculate VIX in Python, you will need to: \n"
                          "1. Gather option prices for S&P 500\n"
                          "2. Calculate the time to expiration (T)\n"
                          "3. Use the risk-free interest rate (r)\n"
                          "4. Find the intervals between strike prices (delta K_i)\n"
                          "5. Compute the weighted contribution of each option\n"
                          "6. Apply the final formula",
                "metadata": {"url": "https://example.com/vix-implementation"}
            }
        ]
    }
    
    print("\nExecuting CoderAgent...")
    print("Query: Write Python code to calculate VIX score")
    print(f"Plan: {example_plan['objective']}")
    print(f"Chunks: {len(example_chunks['chunks'])} research sources")
    
    # Instead of making actual API calls, use pre-defined example results
    print("\nGenerating code based on research...")
    
    # Print the explanation
    print("\nEXPLANATION:")
    explanation = ("This code implements the VIX calculation formula based on the provided research. "
                  "It calculates the Volatility Index (VIX) using S&P 500 option prices, applying "
                  "the standard formula: VIX = 100 * sqrt(T * sum((delta K_i / K_i^2) * e^(rT) * Q(K_i))). "
                  "The implementation handles option data processing, interval calculations, and weighted averages.")
    print(explanation)
    
    # Print the generated code files
    print("\n" + "-"*50)
    print("GENERATED CODE:")
    print("-"*50)
    
    # First file: vix_calculator.py
    vix_calculator_content = '''import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class VIXCalculator:
    """Calculator for the CBOE Volatility Index (VIX) using S&P 500 options data."""
    
    def __init__(self, risk_free_rate=0.01):
        """Initialize the VIX calculator with a risk-free interest rate.
        
        Args:
            risk_free_rate (float): Annual risk-free interest rate, default is 1%
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_vix(self, options_data, current_price, days_to_expiration=30):
        """Calculate the VIX score based on S&P 500 options data.
        
        Args:
            options_data (list): List of dictionaries containing options data with fields:
                - 'strike': Strike price
                - 'bid': Bid price
                - 'ask': Ask price
                - 'type': 'call' or 'put'
            current_price (float): Current S&P 500 index price
            days_to_expiration (int): Days until option expiration, default is 30
            
        Returns:
            float: Calculated VIX score
        """
        # Convert days to years for time to expiration (T)
        time_to_expiration = days_to_expiration / 365.0
        
        try:
            # Sort options by strike price
            sorted_options = sorted(options_data, key=lambda x: x['strike'])
            
            # Calculate option intervals (delta K)
            delta_ks = self._calculate_intervals(sorted_options)
            
            # Exponential term for time value adjustment
            e_rt = math.exp(self.risk_free_rate * time_to_expiration)
            
            # Calculate the weighted sum component
            weighted_sum = 0
            for i, option in enumerate(sorted_options):
                # Skip options without interval information
                if i not in delta_ks:
                    continue
                    
                # Get strike price (K) and interval (delta K)
                strike = option['strike']
                delta_k = delta_ks[i]
                
                # Only include out-of-the-money options
                is_otm = (option['type'] == 'call' and strike > current_price) or \
                         (option['type'] == 'put' and strike < current_price)
                         
                if not is_otm:
                    continue
                
                # Calculate midpoint of bid-ask spread (Q(K))
                midpoint = (option['bid'] + option['ask']) / 2
                if midpoint <= 0:
                    continue  # Skip options with zero or negative midpoints
                
                # Apply the VIX formula component: (delta K / K^2) * e^(rT) * Q(K)
                contribution = (delta_k / (strike**2)) * e_rt * midpoint
                weighted_sum += contribution
            
            # Apply final VIX formula: 100 * sqrt(T * weighted_sum)
            vix = 100 * math.sqrt(time_to_expiration * weighted_sum)
            return vix
            
        except Exception as e:
            raise ValueError(f"Error calculating VIX: {str(e)}")
    
    def _calculate_intervals(self, sorted_options):
        """Calculate the intervals between strike prices (delta K_i).
        
        Args:
            sorted_options (list): Options sorted by strike price
            
        Returns:
            dict: Dictionary mapping option index to its delta K value
        """
        intervals = {}
        
        # Handle special cases for first and last options
        if len(sorted_options) > 0:
            # First option
            if len(sorted_options) > 1:
                intervals[0] = sorted_options[1]['strike'] - sorted_options[0]['strike']
            
            # Middle options
            for i in range(1, len(sorted_options) - 1):
                intervals[i] = (sorted_options[i+1]['strike'] - sorted_options[i-1]['strike']) / 2
            
            # Last option
            if len(sorted_options) > 1:
                last_idx = len(sorted_options) - 1
                intervals[last_idx] = sorted_options[last_idx]['strike'] - sorted_options[last_idx-1]['strike']
        
        return intervals


def calculate_vix_from_csv(csv_file, risk_free_rate=0.01, days_to_expiration=30):
    """Calculate VIX from options data in a CSV file.
    
    Args:
        csv_file (str): Path to CSV file with options data
        risk_free_rate (float): Risk-free interest rate
        days_to_expiration (int): Days to expiration
        
    Returns:
        float: Calculated VIX score
    """
    try:
        # Read options data from CSV
        df = pd.read_csv(csv_file)
        
        # Convert DataFrame to list of dictionaries
        options_data = df.to_dict('records')
        
        # Get current S&P 500 price (assuming it's in the file or could be extracted)
        current_price = df['current_price'].iloc[0] if 'current_price' in df.columns else 4000
        
        # Calculate VIX
        calculator = VIXCalculator(risk_free_rate=risk_free_rate)
        vix = calculator.calculate_vix(
            options_data=options_data,
            current_price=current_price,
            days_to_expiration=days_to_expiration
        )
        
        return vix
        
    except Exception as e:
        print(f"Error calculating VIX from CSV: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    # Sample options data
    sample_options = [
        {'strike': 3900, 'bid': 15.0, 'ask': 16.0, 'type': 'put'},
        {'strike': 3950, 'bid': 12.5, 'ask': 13.5, 'type': 'put'},
        {'strike': 4000, 'bid': 10.0, 'ask': 11.0, 'type': 'put'},
        {'strike': 4050, 'bid': 8.0, 'ask': 9.0, 'type': 'call'},
        {'strike': 4100, 'bid': 10.5, 'ask': 11.5, 'type': 'call'},
        {'strike': 4150, 'bid': 13.0, 'ask': 14.0, 'type': 'call'}
    ]
    
    # Current S&P 500 index price
    current_price = 4025.0
    
    # Calculate VIX
    calculator = VIXCalculator(risk_free_rate=0.015)  # 1.5% risk-free rate
    vix = calculator.calculate_vix(
        options_data=sample_options,
        current_price=current_price,
        days_to_expiration=30
    )
    
    print(f"Calculated VIX: {vix:.2f}")
'''
    
    print("\nFILE: vix_calculator.py")
    print("-"*40)
    print(vix_calculator_content)
    
    # Second file: vix_example.csv
    vix_example_csv = '''strike,bid,ask,type,current_price
3900,15.0,16.0,put,4025
3950,12.5,13.5,put,4025
4000,10.0,11.0,put,4025
4050,8.0,9.0,call,4025
4100,10.5,11.5,call,4025
4150,13.0,14.0,call,4025
'''
    
    print("\nFILE: vix_example.csv")
    print("-"*40)
    print(vix_example_csv)
    
    # Print usage instructions
    print("\nUSAGE INSTRUCTIONS:")
    usage_instructions = '''To use the VIX calculator:

1. Install the required dependencies:
   ```
   pip install numpy pandas
   ```

2. Import the VIXCalculator class:
   ```python
   from vix_calculator import VIXCalculator, calculate_vix_from_csv
   ```

3. Use the calculator with your own options data:
   ```python
   calculator = VIXCalculator(risk_free_rate=0.01)
   vix = calculator.calculate_vix(
       options_data=your_options_data,
       current_price=current_index_price,
       days_to_expiration=30
   )
   ```

4. Alternatively, use the CSV helper function:
   ```python
   vix = calculate_vix_from_csv('path_to_your_data.csv')
   ```

The included vix_example.csv file provides a sample format for CSV input data.'''
    print(usage_instructions)
    
    print("\n" + "-"*50)
    
    # Log the agent state - example of what would happen in real execution
    print("\nExample logging: CoderAgent completed code generation with 2 files produced")
    print("(This would be saved to the database in a real execution)")
    print("\n" + "-"*50)
