"""
File utility functions for Agentic Researcher
Handles directory creation, path management, and file operations
"""
import os
import logging
from pathlib import Path
from typing import List

# Configure logging
logger = logging.getLogger(__name__)

def ensure_directories(directories: List[str]) -> None:
    """
    Ensure all required directories exist
    
    Args:
        directories: List of directory paths to create if they don't exist
    """
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

def get_project_dir(project_id: int, base_dir: str, create: bool = True) -> str:
    """
    Get project directory path and create it if needed
    
    Args:
        project_id: Project ID
        base_dir: Base directory for projects
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        str: Path to the project directory
    """
    project_dir = os.path.join(base_dir, f"project_{project_id}")
    
    if create and not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)
        logger.info(f"Created project directory: {project_dir}")
        
    return project_dir

def write_json_file(file_path: str, data: dict, pretty: bool = True) -> None:
    """
    Write data to a JSON file
    
    Args:
        file_path: Path to the JSON file
        data: Data to write
        pretty: Whether to format the JSON with indentation
    """
    import json
    
    # Ensure the directory exists
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    # Write the data to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)
    
    logger.info(f"Wrote data to file: {file_path}")

def read_json_file(file_path: str, default=None):
    """
    Read data from a JSON file
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return if the file doesn't exist or can't be parsed
        
    Returns:
        Data from the JSON file, or the default value
    """
    import json
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read JSON file {file_path}: {str(e)}")
    
    return default
