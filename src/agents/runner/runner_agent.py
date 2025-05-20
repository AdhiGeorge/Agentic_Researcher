"""Runner Agent for Agentic Researcher

This agent executes generated code, captures outputs, and reports results.
It provides a sandboxed environment for running research-related code.
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Union


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.utils.config import config
from src.db.sqlite_manager import SQLiteManager
from src.utils.openai_client import AzureOpenAIClient

class RunnerAgent:
    """
    Runner Agent that executes generated code and captures outputs
    
    This agent provides a controlled environment for executing code generated
    by the Coder Agent, capturing outputs and errors, and providing detailed
    execution reports.
    """
    
    def __init__(self):
        """Initialize the RunnerAgent"""
        self.name = "runner"
        
        # Initialize database connection
        self.sqlite_manager = SQLiteManager()
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAIClient()
        
        # Set up execution environment
        self.temp_dir = tempfile.mkdtemp(prefix="agentic_runner_")
        print(f"Runner agent initialized with temp directory: {self.temp_dir}")
        
        # Example code for testing
        self.example_code = [
            {
                "file_name": "vix_calculator.py",
                "content": '''# VIX Calculator - Simple Implementation
import numpy as np
import matplotlib.pyplot as plt

def calculate_historical_volatility(prices, window=30):
    """Calculate historical volatility from price series."""
    # Calculate log returns
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Calculate rolling standard deviation
    volatility = np.std(log_returns[-window:]) * np.sqrt(252)  # Annualized
    
    return volatility

def simulate_price_data(days=252, seed=42):
    """Simulate price data for demonstration."""
    np.random.seed(seed)
    price = 100
    prices = [price]
    for _ in range(days-1):
        change = np.random.normal(0, 0.01)  # 1% daily volatility
        price *= (1 + change)
        prices.append(price)
    return np.array(prices)

# Generate sample data
prices = simulate_price_data(252)

# Calculate volatility
vol = calculate_historical_volatility(prices)
print(f"Simulated VIX (30-day historical volatility): {vol*100:.2f}%")

# Plot prices
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(prices)
plt.title("Simulated Price Data")
plt.ylabel("Price")

# Calculate rolling volatility
volatilities = []
for i in range(30, len(prices)):
    vol = calculate_historical_volatility(prices[:i+1])
    volatilities.append(vol*100)

# Plot volatility
plt.subplot(2, 1, 2)
plt.plot(range(30, len(prices)), volatilities)
plt.title("30-Day Historical Volatility (Annualized)")
plt.ylabel("Volatility (%)")
plt.xlabel("Trading Day")
plt.tight_layout()

# Save the plot
plt.savefig("vix_simulation.png")
print("Plot saved to 'vix_simulation.png'")
'''
            }
        ]
    
    def execute_code(self, code_files: List[Dict[str, str]], project_id: int,
                   requirements: Optional[List[str]] = None,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the provided code files, capturing outputs and errors
        
        Args:
            code_files: List of dictionaries with file_name and content
            project_id: Project ID
            requirements: Optional list of package requirements
            context: Additional context information
            
        Returns:
            Dict: Execution results including outputs and errors
        """
        start_time = time.time()
        result = {
            "project_id": project_id,
            "execution_start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": False,
            "output": "",
            "error": "",
            "files": [],
            "execution_time": 0
        }
        
        try:
            # Create a subdirectory for this specific execution
            execution_dir = os.path.join(self.temp_dir, f"proj_{project_id}_{int(time.time())}")
            os.makedirs(execution_dir, exist_ok=True)
            
            # Write code files to disk
            file_paths = []
            main_file = None
            
            for file_info in code_files:
                file_name = file_info["file_name"]
                content = file_info["content"]
                
                # Save the file
                file_path = os.path.join(execution_dir, file_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                file_paths.append(file_path)
                
                # Keep track of potential main files for execution
                if "main" in file_name.lower() or (not main_file and file_name.endswith(".py")):
                    main_file = file_path
            
            # Save file list in result
            result["files"] = [os.path.basename(p) for p in file_paths]
            
            # Install requirements if provided
            if requirements and len(requirements) > 0:
                req_str = " ".join(requirements)
                try:
                    pip_process = subprocess.run(
                        [sys.executable, "-m", "pip", "install", *requirements],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout for installations
                    )
                    if pip_process.returncode != 0:
                        result["error"] += f"Error installing dependencies: {pip_process.stderr}\n"
                        result["requirements_installed"] = False
                    else:
                        result["requirements_installed"] = True
                except Exception as e:
                    result["error"] += f"Error installing requirements: {str(e)}\n"
                    result["requirements_installed"] = False
            
            # Choose the file to execute
            if not main_file and file_paths:
                # If no main file identified, choose the first .py file
                python_files = [f for f in file_paths if f.endswith(".py")]
                if python_files:
                    main_file = python_files[0]
            
            if main_file:
                # Execute the main file
                try:
                    process = subprocess.run(
                        [sys.executable, main_file],
                        capture_output=True,
                        text=True,
                        cwd=execution_dir,
                        timeout=60  # 1 minute timeout for execution
                    )
                    
                    # Capture output and error
                    result["output"] = process.stdout
                    if process.stderr:
                        result["error"] += process.stderr
                    
                    # Set success flag
                    result["success"] = process.returncode == 0
                    result["execution_code"] = process.returncode
                    
                except subprocess.TimeoutExpired:
                    result["error"] += "Execution timed out after 60 seconds.\n"
                    result["success"] = False
                except Exception as e:
                    result["error"] += f"Error during execution: {str(e)}\n"
                    result["success"] = False
            else:
                result["error"] += "No executable Python file found.\n"
                result["success"] = False
            
            # Check for generated files/outputs
            try:
                output_files = []
                for root, _, files in os.walk(execution_dir):
                    for file in files:
                        if file.endswith((".png", ".jpg", ".jpeg", ".csv", ".json", ".html")):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, execution_dir)
                            output_files.append(rel_path)
                
                result["output_files"] = output_files
            except Exception as e:
                result["error"] += f"Error checking for output files: {str(e)}\n"
        
        except Exception as e:
            result["error"] += f"Runner agent execution error: {str(e)}\n"
            result["success"] = False
        
        # Calculate execution time
        result["execution_time"] = time.time() - start_time
        result["execution_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Store execution result
        self.sqlite_manager.add_execution_result(project_id, result)
        
        return result




# Example usage
if __name__ == "__main__":
    runner = RunnerAgent()
    
    # Run example code
    result = runner.execute_code(
        code_files=runner.example_code,
        project_id=1,
        requirements=["numpy", "matplotlib"]
    )
    
    print("\nExecution Result:")
    print(json.dumps(result, indent=2))
