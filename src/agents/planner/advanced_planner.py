"""
Advanced Planner Agent for Agentic Researcher

This module implements an enhanced planner agent with sophisticated task decomposition,
resource allocation, and detailed research planning capabilities.
"""

import os
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Import project utilities
from src.utils.config import config
from src.utils.openai_client import get_chat_client, generate_chat_completion

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedPlannerAgent:
    """
    Enhanced planner agent for creating detailed, actionable research plans
    
    Capabilities:
    - Task decomposition: Breaking complex queries into subtasks
    - Resource allocation: Specifying tools and agents needed for each task
    - Dependency tracking: Managing task dependencies
    - Time estimation: Estimating the time required for tasks
    """
    
    def __init__(self):
        """Initialize the advanced planner agent"""
        self.client = get_chat_client()
        self.system_prompt = self._get_system_prompt()
        
        logger.info("Advanced Planner Agent initialized")
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the planner agent
        
        Returns:
            System prompt string
        """
        return """You are an Advanced Research Planner, designed to create detailed, structured research plans.

Your task is to analyze research queries and break them down into logical, well-organized steps that will lead to a comprehensive answer.

For each research query, you will:
1. Analyze the query to identify the key topics, questions, and requirements
2. Break down the research process into discrete tasks and subtasks
3. Specify which agent should handle each task (Researcher, Analyst, Coder, or AnswerAgent)
4. Indicate any dependencies between tasks
5. Allocate appropriate tools and resources for each task
6. Provide estimated time requirements for complex tasks

Your output should be a structured research plan in the following format:

```json
{
  "query_analysis": {
    "key_topics": ["topic1", "topic2"],
    "research_questions": ["question1", "question2"],
    "requirements": ["requirement1", "requirement2"]
  },
  "tasks": [
    {
      "id": "task1",
      "description": "Description of the task",
      "agent": "ResearcherAgent",
      "tools_needed": ["web_search", "content_scraping"],
      "dependencies": [],
      "estimated_time": "2 minutes"
    },
    {
      "id": "task2",
      "description": "Description of the task",
      "agent": "AnalystAgent",
      "tools_needed": ["data_analysis"],
      "dependencies": ["task1"],
      "estimated_time": "3 minutes"
    }
  ],
  "execution_strategy": "parallel_with_dependencies",
  "expected_output": "Detailed description of what the final output should include"
}
```

Additional Guidelines:
- If the query involves data analysis, include specific analytical approaches
- If code generation is needed, specify the programming language and libraries
- Always consider the most efficient research strategy
- For tasks involving uploaded files, specify how the file data should be incorporated
- Prioritize tasks based on importance and dependencies

Your goal is to create a plan that is both comprehensive and executable, ensuring that all aspects of the research query are thoroughly addressed.
"""

    async def create_research_plan(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a detailed research plan for a query
        
        Args:
            query: Research query
            context: Additional context (e.g., file data)
            
        Returns:
            Dictionary containing the research plan
        """
        try:
            # Prepare context information
            context_info = ""
            if context and "file_data" in context:
                file_data = context["file_data"]
                context_info += f"\n\nAdditional Context - File Data:"
                context_info += f"\n- File Name: {file_data.get('file_name', 'Unknown')}"
                context_info += f"\n- File Type: {file_data.get('file_type', 'Unknown')}"
                context_info += f"\n- Data Type: {file_data.get('data_type', 'Unknown')}"
                
                # Add more specific information based on data type
                if file_data.get("data_type") == "tabular" and "columns" in file_data:
                    context_info += f"\n- Columns: {', '.join(file_data['columns'])}"
                    if "data_shape" in file_data:
                        context_info += f"\n- Shape: {file_data['data_shape'][0]} rows × {file_data['data_shape'][1]} columns"
                
                elif file_data.get("data_type") in ["text", "document"] and "text_length" in file_data:
                    context_info += f"\n- Text Length: {file_data['text_length']} characters"
            
            # Create input prompt
            input_prompt = f"""
Research Query: {query}
{context_info}

Please create a detailed research plan for addressing this query. Follow the format specified in the system instructions.
"""
            
            # Generate the research plan
            response = await generate_chat_completion(
                client=self.client,
                system_message=self.system_prompt,
                user_message=input_prompt,
                temperature=0.7,
                max_tokens=3000
            )
            
            # Extract the JSON research plan
            plan_text = response.get("content", "")
            
            # Try to extract JSON
            try:
                # Look for JSON pattern between triple backticks
                import re
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", plan_text)
                
                if json_match:
                    plan_json = json.loads(json_match.group(1))
                else:
                    # Try to parse the entire text as JSON
                    plan_json = json.loads(plan_text)
                    
                logger.info("Successfully created research plan")
                return {
                    "plan": plan_json,
                    "raw_plan": plan_text
                }
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                logger.warning("Failed to parse research plan as JSON")
                return {
                    "plan": None,
                    "raw_plan": plan_text,
                    "error": "Failed to parse research plan as JSON"
                }
            
        except Exception as e:
            logger.error(f"Error creating research plan: {str(e)}")
            return {
                "plan": None,
                "raw_plan": None,
                "error": f"Error creating research plan: {str(e)}"
            }
    
    async def decompose_task(self, task_description: str) -> Dict[str, Any]:
        """
        Decompose a complex task into subtasks
        
        Args:
            task_description: Description of the complex task
            
        Returns:
            Dictionary containing the decomposed subtasks
        """
        try:
            # Create input prompt for task decomposition
            input_prompt = f"""
Task Description: {task_description}

Please decompose this task into smaller, more manageable subtasks. For each subtask, specify:
1. A clear description
2. The agent responsible for executing it
3. Any tools or resources needed
4. Dependencies on other subtasks
5. Estimated time for completion

Format the output as a JSON array of subtasks.
"""
            
            # Generate the subtasks
            response = await generate_chat_completion(
                client=self.client,
                system_message=self.system_prompt,
                user_message=input_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract the subtasks
            subtasks_text = response.get("content", "")
            
            # Try to extract JSON
            try:
                # Look for JSON pattern between triple backticks
                import re
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", subtasks_text)
                
                if json_match:
                    subtasks_json = json.loads(json_match.group(1))
                else:
                    # Try to parse the entire text as JSON
                    subtasks_json = json.loads(subtasks_text)
                    
                logger.info(f"Successfully decomposed task into {len(subtasks_json)} subtasks")
                return {
                    "subtasks": subtasks_json,
                    "raw_response": subtasks_text
                }
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                logger.warning("Failed to parse subtasks as JSON")
                return {
                    "subtasks": None,
                    "raw_response": subtasks_text,
                    "error": "Failed to parse subtasks as JSON"
                }
            
        except Exception as e:
            logger.error(f"Error decomposing task: {str(e)}")
            return {
                "subtasks": None,
                "raw_response": None,
                "error": f"Error decomposing task: {str(e)}"
            }
    
    async def allocate_resources(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Allocate resources for a set of tasks
        
        Args:
            tasks: List of tasks
            
        Returns:
            Dictionary containing resource allocation
        """
        try:
            # Format tasks as JSON
            tasks_json = json.dumps(tasks, indent=2)
            
            # Create input prompt for resource allocation
            input_prompt = f"""
Tasks:
{tasks_json}

Please analyze these tasks and allocate appropriate resources for each one. For each task, specify:
1. The priority level (high, medium, low)
2. Specific tools and APIs needed
3. The most suitable agent type
4. Whether the task can be executed in parallel with others

Format the output as a JSON object with task IDs as keys and resource allocations as values.
"""
            
            # Generate the resource allocation
            response = await generate_chat_completion(
                client=self.client,
                system_message=self.system_prompt,
                user_message=input_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract the resource allocation
            allocation_text = response.get("content", "")
            
            # Try to extract JSON
            try:
                # Look for JSON pattern between triple backticks
                import re
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", allocation_text)
                
                if json_match:
                    allocation_json = json.loads(json_match.group(1))
                else:
                    # Try to parse the entire text as JSON
                    allocation_json = json.loads(allocation_text)
                    
                logger.info("Successfully allocated resources for tasks")
                return {
                    "allocation": allocation_json,
                    "raw_response": allocation_text
                }
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                logger.warning("Failed to parse resource allocation as JSON")
                return {
                    "allocation": None,
                    "raw_response": allocation_text,
                    "error": "Failed to parse resource allocation as JSON"
                }
            
        except Exception as e:
            logger.error(f"Error allocating resources: {str(e)}")
            return {
                "allocation": None,
                "raw_response": None,
                "error": f"Error allocating resources: {str(e)}"
            }
    
    async def estimate_time_requirements(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate time requirements for a research plan
        
        Args:
            plan: Research plan
            
        Returns:
            Dictionary containing time estimates
        """
        try:
            # Format plan as JSON
            plan_json = json.dumps(plan, indent=2)
            
            # Create input prompt for time estimation
            input_prompt = f"""
Research Plan:
{plan_json}

Please analyze this research plan and provide time estimates for each task and the overall plan. Consider:
1. The complexity of each task
2. Dependencies between tasks
3. Parallel execution opportunities
4. Any potential bottlenecks

Format the output as a JSON object with task IDs as keys and time estimates as values, plus an "overall" estimate.
"""
            
            # Generate the time estimates
            response = await generate_chat_completion(
                client=self.client,
                system_message=self.system_prompt,
                user_message=input_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract the time estimates
            estimates_text = response.get("content", "")
            
            # Try to extract JSON
            try:
                # Look for JSON pattern between triple backticks
                import re
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", estimates_text)
                
                if json_match:
                    estimates_json = json.loads(json_match.group(1))
                else:
                    # Try to parse the entire text as JSON
                    estimates_json = json.loads(estimates_text)
                    
                logger.info("Successfully estimated time requirements")
                return {
                    "estimates": estimates_json,
                    "raw_response": estimates_text
                }
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                logger.warning("Failed to parse time estimates as JSON")
                return {
                    "estimates": None,
                    "raw_response": estimates_text,
                    "error": "Failed to parse time estimates as JSON"
                }
            
        except Exception as e:
            logger.error(f"Error estimating time requirements: {str(e)}")
            return {
                "estimates": None,
                "raw_response": None,
                "error": f"Error estimating time requirements: {str(e)}"
            }
    
    async def create_file_analysis_plan(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a plan specifically for analyzing an uploaded file
        
        Args:
            file_data: File metadata and content
            
        Returns:
            Dictionary containing the file analysis plan
        """
        try:
            # Extract file information
            file_name = file_data.get("file_name", "Unknown")
            file_type = file_data.get("file_type", "Unknown")
            data_type = file_data.get("data_type", "Unknown")
            
            # Create context information string
            context_info = f"File Name: {file_name}\nFile Type: {file_type}\nData Type: {data_type}\n"
            
            # Add more specific information based on data type
            if data_type == "tabular" and "columns" in file_data:
                context_info += f"Columns: {', '.join(file_data['columns'])}\n"
                if "data_shape" in file_data:
                    context_info += f"Shape: {file_data['data_shape'][0]} rows × {file_data['data_shape'][1]} columns\n"
            
            elif data_type in ["text", "document"] and "text_length" in file_data:
                context_info += f"Text Length: {file_data['text_length']} characters\n"
            
            # Create input prompt for file analysis planning
            input_prompt = f"""
File Information:
{context_info}

Please create a detailed plan for analyzing this file. The plan should include:
1. Initial data exploration steps
2. Key analyses to perform
3. Visualizations to generate (if applicable)
4. Insights to extract
5. Tools and techniques to use

Consider the file type and data type in your plan. Format the output as a structured JSON plan.
"""
            
            # Generate the file analysis plan
            response = await generate_chat_completion(
                client=self.client,
                system_message=self.system_prompt,
                user_message=input_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract the file analysis plan
            plan_text = response.get("content", "")
            
            # Try to extract JSON
            try:
                # Look for JSON pattern between triple backticks
                import re
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", plan_text)
                
                if json_match:
                    plan_json = json.loads(json_match.group(1))
                else:
                    # Try to parse the entire text as JSON
                    plan_json = json.loads(plan_text)
                    
                logger.info(f"Successfully created file analysis plan for {file_name}")
                return {
                    "plan": plan_json,
                    "raw_plan": plan_text
                }
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                logger.warning("Failed to parse file analysis plan as JSON")
                return {
                    "plan": None,
                    "raw_plan": plan_text,
                    "error": "Failed to parse file analysis plan as JSON"
                }
            
        except Exception as e:
            logger.error(f"Error creating file analysis plan: {str(e)}")
            return {
                "plan": None,
                "raw_plan": None,
                "error": f"Error creating file analysis plan: {str(e)}"
            }
