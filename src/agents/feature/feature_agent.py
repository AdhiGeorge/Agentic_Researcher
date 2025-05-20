"""Feature Agent for Agentic Researcher

This agent is responsible for adding new features to the research system based on user requests.
It analyzes feature requirements and implements them using the appropriate tools and libraries.
"""

import json
import sys
import os


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import time
import re
from typing import Dict, List, Any, Optional, Tuple

from src.utils.config import config
from src.db.sqlite_manager import SQLiteManager
from src.utils.openai_client import AzureOpenAIClient

class FeatureAgent:
    """
    Feature Agent that adds new features to the research system
    
    This agent analyzes feature requirements and implements them appropriately,
    working with the coder agent to develop new functionality.
    """
    
    def __init__(self):
        """Initialize the FeatureAgent"""
        self.name = "feature"
        
        # Initialize database connection
        self.sqlite_manager = SQLiteManager()
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAIClient()
    
    def analyze_feature_request(self, request: str) -> Dict[str, Any]:
        """
        Analyze a feature request to determine requirements and implementation approach
        
        Args:
            request: Feature request description
            
        Returns:
            Dict: Analysis results and implementation plan
        """
        prompt = f"""You are an expert feature analyst. Analyze the following feature request and break it down into requirements and implementation steps.

FEATURE REQUEST: {request}

Provide a detailed analysis including:
1. What the feature aims to accomplish
2. Technical requirements
3. Dependencies and libraries needed
4. Implementation approach
5. Potential challenges

Format your response as a JSON object with the following structure:
{{
    "feature_name": "A concise name for this feature",
    "description": "Detailed description of the feature",
    "requirements": ["List of specific requirements"],
    "dependencies": ["Libraries or tools needed"],
    "implementation_steps": ["Step-by-step implementation plan"],
    "challenges": ["Potential implementation challenges"],
    "estimated_complexity": "Low/Medium/High"
}}
"""

        # Call Azure OpenAI API
        response = self.openai_client.generate_completion(
            messages=[
                {"role": "system", "content": "You are an expert feature analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse and return the response
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except:
                    result = {
                        "feature_name": "Unspecified Feature",
                        "description": request,
                        "requirements": ["Feature analysis parsing failed"],
                        "dependencies": [],
                        "implementation_steps": ["Manual analysis required"],
                        "challenges": ["JSON parsing issue"],
                        "estimated_complexity": "Unknown",
                        "raw_response": response
                    }
            else:
                result = {
                    "feature_name": "Unspecified Feature",
                    "description": request,
                    "requirements": ["Feature analysis parsing failed"],
                    "dependencies": [],
                    "implementation_steps": ["Manual analysis required"],
                    "challenges": ["JSON parsing issue"],
                    "estimated_complexity": "Unknown",
                    "raw_response": response
                }
        
        return result
    
    def implement_feature(self, feature_request: str, project_id: int,
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Implement a new feature based on the request
        
        Args:
            feature_request: Description of the feature to implement
            project_id: Project ID
            context: Additional context information
            
        Returns:
            Dict: Implementation results and generated code
        """
        # First analyze the feature
        analysis = self.analyze_feature_request(feature_request)
        
        # Set up context information
        full_context = context or {}
        full_context.update({
            "project_id": project_id,
            "feature_request": feature_request,
            "feature_analysis": analysis
        })
        
        # Save the analysis to the database
        self.sqlite_manager.add_feature_request(project_id, feature_request, analysis)
        
        # Construct implementation prompt
        dependencies = ", ".join(analysis.get("dependencies", []))
        implementation_steps = "\n".join([f"- {step}" for step in analysis.get("implementation_steps", [])])
        
        try:
            prompt = f"""You are an expert programmer tasked with implementing a new feature. Generate the necessary code based on the following specifications:

FEATURE REQUEST: {feature_request}

ANALYSIS:
- Feature Name: {analysis.get('feature_name', 'New Feature')}
- Description: {analysis.get('description', 'No description available')}
- Technical Requirements: {', '.join(analysis.get('requirements', ['No requirements specified']))}
- Dependencies: {dependencies if dependencies else 'No specific dependencies'}

IMPLEMENTATION PLAN:
{implementation_steps if implementation_steps else '- No implementation steps specified'}

Create well-structured, production-quality Python code for this feature. Include clear comments, error handling, and docstrings. The code should be modular and integrate well with the existing Agentic Researcher system.

Format your response with appropriate markdown code blocks. Each code file should be clearly labeled with its filename.
Provide a brief explanation of how the feature works, usage instructions, and integration notes.
"""

            # Call Azure OpenAI API for implementation
            response = self.openai_client.generate_completion(
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more focused code generation
                max_tokens=4000  # Increased token limit for complex features
            )
            
            # Extract code files from the response
            implementation_result = self._extract_code_files(response)
            
            # Save the implementation to the database
            self.sqlite_manager.add_feature_implementation(
                project_id, 
                feature_request, 
                implementation_result
            )
            
            # Add additional metadata
            implementation_result.update({
                "feature_name": analysis.get("feature_name", "New Feature"),
                "feature_request": feature_request,
                "project_id": project_id,
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return implementation_result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "feature_request": feature_request
            }
    
    def _extract_code_files(self, content: str) -> Dict[str, Any]:
        """
        Extract code files from non-JSON response
        
        Args:
            content: Response content
            
        Returns:
            Dict: Structured implementation results
        """
        # Initialize result structure
        result = {
            "explanation": "",
            "files": [],
            "usage_instructions": "",
            "integration_notes": ""
        }
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:(?P<lang>\w+)\n)?(?P<code>.*?)```', content, re.DOTALL)
        
        # Extract filenames and content
        files_info = []
        current_file = None
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Look for file name patterns
            file_match = re.match(r'^(?:file|filename):\s*["\']?([\w\.-]+)["\']?\s*$', line, re.IGNORECASE)
            if file_match:
                current_file = file_match.group(1)
                continue
                
            # Alternative file pattern: heading with .py extension
            if line.startswith('#') and '.py' in line and not current_file:
                file_match = re.search(r'["\']?([\w\.-]+\.py)["\']?', line)
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
                if lang and lang.strip().lower() == "python":
                    file_name = "feature_implementation.py"
                else:
                    # Default to Python
                    file_name = f"feature_implementation.{lang if lang else 'py'}"
            
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
        
        # Extract usage instructions
        usage_match = re.search(r'(?:usage instructions|how to use)[:\s]+(.*?)(?:\n\n|\n#|\Z)', content, re.IGNORECASE | re.DOTALL)
        usage = usage_match.group(1).strip() if usage_match else "See code comments for usage instructions."
        
        # Extract integration notes
        integration_match = re.search(r'(?:integration notes|how to integrate)[:\s]+(.*?)(?:\n\n|\n#|\Z)', content, re.IGNORECASE | re.DOTALL)
        integration = integration_match.group(1).strip() if integration_match else "See code comments for integration details."
        
        result["explanation"] = explanation
        result["files"] = files_info
        result["usage_instructions"] = usage
        result["integration_notes"] = integration
        
        return result


# Example usage
if __name__ == "__main__":
    feature_agent = FeatureAgent()
    
    # Example feature request
    feature_analysis = feature_agent.analyze_feature_request(
        "Add a feature to visualize volatility index trends over time using matplotlib"
    )
    
    print("Feature Analysis:")
    print(json.dumps(feature_analysis, indent=2))
