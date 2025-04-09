
import os
import json
import logging
from typing import Dict, List, Any, Optional
import time

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages persistent memory for the research assistant.
    Stores search history, tools, and other data.
    """
    
    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self.history_file = os.path.join(memory_path, "search_history.json")
        self.tools_file = os.path.join(memory_path, "tools.json")
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_path, exist_ok=True)
        
        # Initialize history and tools files if they don't exist
        self._init_files()
    
    def _init_files(self):
        """Initialize memory files if they don't exist"""
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.tools_file):
            with open(self.tools_file, 'w') as f:
                json.dump([], f)
    
    def add_search_history(self, query: str, result_summary: Dict[str, Any]):
        """
        Add a search query and its results to history
        
        Args:
            query: The search query
            result_summary: Summary of the search results
        """
        try:
            # Load existing history
            history = self.get_search_history()
            
            # Add new entry
            history.append({
                "query": query,
                "timestamp": time.time(),
                "summary": result_summary
            })
            
            # Save history
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            logger.info(f"Added query to search history: {query}")
            
        except Exception as e:
            logger.error(f"Error adding to search history: {str(e)}")
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get the search history"""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading search history: {str(e)}")
            return []
    
    def add_tool(self, tool_data: Dict[str, Any]):
        """
        Add a tool to the tool registry
        
        Args:
            tool_data: Tool definition data
        """
        try:
            # Load existing tools
            tools = self.get_tools()
            
            # Check if tool already exists
            for i, tool in enumerate(tools):
                if tool.get("name") == tool_data.get("name"):
                    # Update existing tool
                    tools[i] = tool_data
                    break
            else:
                # Add new tool
                tools.append(tool_data)
            
            # Save tools
            with open(self.tools_file, 'w') as f:
                json.dump(tools, f, indent=2)
                
            logger.info(f"Added/updated tool: {tool_data.get('name')}")
            
        except Exception as e:
            logger.error(f"Error adding tool: {str(e)}")
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the registered tools"""
        try:
            with open(self.tools_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading tools: {str(e)}")
            return []
    
    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool by name"""
        tools = self.get_tools()
        for tool in tools:
            if tool.get("name") == name:
                return tool
        return None
