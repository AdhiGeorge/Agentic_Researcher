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


# Example usage
if __name__ == "__main__":
    import tempfile
    import shutil
    from pprint import pprint
    from datetime import datetime
    
    # Configure logging to show info messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("\n" + "=" * 80)
    print("File Utilities - Example Usage")
    print("=" * 80)
    
    # Create a temporary directory for our examples
    temp_base_dir = tempfile.mkdtemp(prefix="file_utils_example_")
    print(f"\nCreated temporary directory for examples: {temp_base_dir}")
    
    try:
        # Example 1: Creating directory structure for a research project
        print("\nExample 1: Creating directory structure for a research project")
        print("-" * 60)
        
        # Define the directories we need
        research_dirs = [
            os.path.join(temp_base_dir, "data"),
            os.path.join(temp_base_dir, "results"),
            os.path.join(temp_base_dir, "models"),
            os.path.join(temp_base_dir, "figures")
        ]
        
        # Ensure all directories exist
        ensure_directories(research_dirs)
        
        # Check that directories were created
        print("Created research directory structure:")
        for directory in research_dirs:
            print(f"  - {os.path.basename(directory)}: {os.path.exists(directory)}")
        
        # Example 2: Managing project directories
        print("\nExample 2: Managing project directories")
        print("-" * 60)
        
        # Create project directories for different research projects
        projects = [
            {"id": 101, "name": "Quantum Computing Research"},
            {"id": 102, "name": "NLP Sentiment Analysis"},
            {"id": 103, "name": "Computer Vision Object Detection"}
        ]
        
        # Create project directories and store their paths
        project_paths = {}
        for project in projects:
            project_dir = get_project_dir(project["id"], temp_base_dir)
            project_paths[project["id"]] = project_dir
            print(f"Created project directory for '{project['name']}': {project_dir}")
        
        # Example 3: Writing and reading JSON files
        print("\nExample 3: Writing and reading JSON files")
        print("-" * 60)
        
        # Create some research project metadata
        project_id = 101  # Quantum Computing project
        project_dir = project_paths[project_id]
        
        # Create research metadata
        research_metadata = {
            "project_id": project_id,
            "title": "Quantum Algorithm Performance Analysis",
            "researchers": ["Dr. Quantum Researcher", "Dr. Algorithm Expert"],
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "description": "Analysis of performance characteristics of quantum algorithms compared to classical counterparts",
            "keywords": ["quantum computing", "algorithm complexity", "performance analysis"],
            "datasets": [
                {"name": "quantum_bench_v1", "size": "2.3GB", "records": 15000},
                {"name": "classical_bench_v1", "size": "1.8GB", "records": 15000}
            ]
        }
        
        # Write the metadata to a JSON file
        metadata_file = os.path.join(project_dir, "metadata.json")
        write_json_file(metadata_file, research_metadata)
        print(f"Wrote research metadata to: {metadata_file}")
        
        # Create some research results
        research_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "Shor's Algorithm",
            "input_size": 1024,
            "execution_time": {
                "quantum": 0.02,
                "classical": 4.5
            },
            "accuracy": 0.997,
            "notes": "Quantum speedup observed as expected for integer factorization"
        }
        
        # Write the results to a JSON file
        results_file = os.path.join(project_dir, "results.json")
        write_json_file(results_file, research_results)
        print(f"Wrote research results to: {results_file}")
        
        # Read back the metadata file
        print("\nReading back the metadata file:")
        loaded_metadata = read_json_file(metadata_file)
        if loaded_metadata:
            print(f"Project title: {loaded_metadata.get('title')}")
            print(f"Keywords: {', '.join(loaded_metadata.get('keywords', []))}")
            print(f"Researchers: {', '.join(loaded_metadata.get('researchers', []))}")
        
        # Example 4: Handling missing files gracefully
        print("\nExample 4: Handling missing files gracefully")
        print("-" * 60)
        
        # Try to read a file that doesn't exist
        nonexistent_file = os.path.join(project_dir, "nonexistent.json")
        default_config = {"status": "default", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        print(f"Attempting to read nonexistent file: {os.path.basename(nonexistent_file)}")
        config = read_json_file(nonexistent_file, default=default_config)
        print("Result from read_json_file:")
        pprint(config)
        print(f"Successfully used default value for missing file.")
        
        # Example 5: Working with multiple project files
        print("\nExample 5: Working with multiple project files")
        print("-" * 60)
        
        # Create a configuration file for each project
        for project_id, project_path in project_paths.items():
            config_data = {
                "project_id": project_id,
                "max_threads": 4,
                "output_format": "json",
                "save_intermediates": True,
                "logging_level": "info",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            config_file = os.path.join(project_path, "config.json")
            write_json_file(config_file, config_data)
            print(f"Created configuration for project {project_id}")
        
        # Read all project configurations and display them
        print("\nReading all project configurations:")
        all_configs = {}
        for project_id, project_path in project_paths.items():
            config_file = os.path.join(project_path, "config.json")
            config = read_json_file(config_file)
            if config:
                all_configs[project_id] = config
                print(f"Project {project_id} config loaded successfully")
        
        print(f"Loaded {len(all_configs)} project configurations")
        
        print("\n" + "=" * 80)
        print("File Utilities example completed successfully!")
        print("These utilities provide essential file operations for")
        print("managing research projects and their associated data.")
        print("=" * 80)
    
    finally:
        # Clean up: remove the temporary directory
        print(f"\nCleaning up temporary directory: {temp_base_dir}")
        shutil.rmtree(temp_base_dir)
