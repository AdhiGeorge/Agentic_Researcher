"""
Interactive Code Execution for Agentic Researcher

This module provides a secure environment for executing Python code
within the research context, with support for data analysis and visualization.
"""

import os
import sys
import io
import logging
import traceback
import ast
import json
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import threading
import time
import contextlib
import uuid

# Import common data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)

class CodeExecutor:
    """
    Secure Python code execution environment
    
    Features:
    - Sandboxed execution
    - Output capturing
    - Plot generation
    - Timeout enforcement
    - Predefined imports
    """
    
    def __init__(self, 
                timeout: int = 30,
                max_output_length: int = 100000,
                allowed_modules: Optional[List[str]] = None):
        """
        Initialize the code executor
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum length of captured output
            allowed_modules: List of allowed modules (None for default set)
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        
        # Default allowed modules (common data science stack)
        self.default_allowed_modules = [
            "pandas", "numpy", "matplotlib", "seaborn", 
            "math", "datetime", "json", "re", "collections",
            "itertools", "functools", "random", "statistics"
        ]
        
        # Set allowed modules
        self.allowed_modules = allowed_modules or self.default_allowed_modules
        
        # Initialize execution environment
        self.globals = {}
        self.locals = {}
        
        # Set up plotting backend
        plt.switch_backend('agg')
        
        logger.info("Code Executor initialized")
    
    def _is_safe_code(self, code: str) -> Tuple[bool, str]:
        """
        Check if code is safe to execute
        
        Args:
            code: Python code string
            
        Returns:
            Tuple of (is_safe, reason)
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Check for prohibited imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name.split('.')[0] not in self.allowed_modules:
                            return False, f"Import of '{name.name}' is not allowed"
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] not in self.allowed_modules:
                        return False, f"Import from '{node.module}' is not allowed"
                
                # Check for prohibited function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                            return False, f"Call to '{node.func.id}' is prohibited"
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['system', 'popen', 'call', 'check_output', 'run']:
                            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                                return False, f"Call to 'os.{node.func.attr}' is prohibited"
                        elif node.func.attr == 'open' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                            return False, f"Call to 'os.open' is prohibited"
            
            return True, ""
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Error analyzing code: {str(e)}"
    
    def execute_code(self, 
                    code: str, 
                    data: Optional[Dict[str, Any]] = None,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Python code and capture results
        
        Args:
            code: Python code to execute
            data: Optional data to make available to the code
            context: Additional context information
            
        Returns:
            Dictionary with execution results
        """
        # Check if code is safe
        is_safe, reason = self._is_safe_code(code)
        if not is_safe:
            return {
                "success": False,
                "error": f"Code security check failed: {reason}",
                "output": "",
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Set up execution environment
        self._setup_environment(data, context)
        
        # Create output buffer
        output_buffer = io.StringIO()
        
        # Result container for thread
        result = {
            "success": False,
            "error": None,
            "output": "",
            "plots": [],
            "variables": {},
            "execution_time": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Execute code in a separate thread with timeout
        def execution_thread():
            start_time = time.time()
            
            try:
                # Redirect stdout/stderr to our buffer
                with contextlib.redirect_stdout(output_buffer):
                    with contextlib.redirect_stderr(output_buffer):
                        # Execute the code
                        exec(code, self.globals, self.locals)
                
                # Capture execution time
                execution_time = time.time() - start_time
                
                # Get output and truncate if necessary
                output = output_buffer.getvalue()
                if len(output) > self.max_output_length:
                    output = output[:self.max_output_length] + f"\n... Output truncated (exceeded {self.max_output_length} characters)"
                
                # Capture generated plots
                plots = self._capture_plots()
                
                # Capture variables
                variables = self._capture_variables()
                
                # Update result
                result["success"] = True
                result["output"] = output
                result["plots"] = plots
                result["variables"] = variables
                result["execution_time"] = execution_time
                
            except Exception as e:
                # Capture error and traceback
                error_msg = str(e)
                tb = traceback.format_exc()
                
                # Get output so far
                output = output_buffer.getvalue()
                
                # Update result with error
                result["success"] = False
                result["error"] = f"{error_msg}\n\n{tb}"
                result["output"] = output
                result["execution_time"] = time.time() - start_time
        
        # Create and start thread
        thread = threading.Thread(target=execution_thread)
        thread.daemon = True
        thread.start()
        
        # Wait for thread to complete or timeout
        thread.join(self.timeout)
        
        # Check if thread is still alive (timeout occurred)
        if thread.is_alive():
            return {
                "success": False,
                "error": f"Execution timed out after {self.timeout} seconds",
                "output": output_buffer.getvalue(),
                "execution_time": self.timeout,
                "timestamp": datetime.now().isoformat()
            }
        
        return result
    
    def _setup_environment(self, 
                          data: Optional[Dict[str, Any]] = None,
                          context: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the execution environment with required imports and data
        
        Args:
            data: Data to make available in the environment
            context: Additional context information
        """
        # Reset environment
        self.globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "datetime": datetime,
        }
        
        self.locals = {}
        
        # Add common modules
        for module_name in self.allowed_modules:
            try:
                if module_name not in self.globals:
                    self.globals[module_name] = __import__(module_name)
            except ImportError:
                logger.warning(f"Could not import {module_name}")
        
        # Add data to environment
        if data:
            for key, value in data.items():
                # Special handling for DataFrames
                if isinstance(value, dict) and "data_type" in value and value["data_type"] == "tabular" and "data" in value:
                    # Add the actual DataFrame
                    self.globals[f"{key}_df"] = value["data"]
                    # Also add metadata
                    self.globals[f"{key}_meta"] = {k: v for k, v in value.items() if k != "data"}
                else:
                    self.globals[key] = value
        
        # Add context to environment
        if context:
            self.globals["context"] = context
    
    def _capture_plots(self) -> List[Dict[str, Any]]:
        """
        Capture and encode any plots generated during execution
        
        Returns:
            List of dictionaries with plot data
        """
        plots = []
        
        # Check if any figures were created
        for i, fig in enumerate(plt.get_fignums()):
            try:
                # Get the figure
                figure = plt.figure(fig)
                
                # Save figure to buffer
                buf = io.BytesIO()
                figure.savefig(buf, format='png')
                plt.close(fig)
                
                # Encode the image
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                
                # Add to plots list
                plots.append({
                    "id": f"plot_{i+1}",
                    "format": "png",
                    "data": img_str
                })
                
            except Exception as e:
                logger.error(f"Error capturing plot {i+1}: {str(e)}")
        
        return plots
    
    def _capture_variables(self) -> Dict[str, Any]:
        """
        Capture variables created during execution
        
        Returns:
            Dictionary of variable names and their string representations
        """
        variables = {}
        
        for name, value in self.locals.items():
            # Skip private variables and modules
            if name.startswith('_') or name in self.allowed_modules:
                continue
            
            # Try to get a string representation
            try:
                # Handle different types appropriately
                if isinstance(value, pd.DataFrame):
                    # For DataFrames, capture shape and preview
                    variables[name] = {
                        "type": "DataFrame",
                        "shape": value.shape,
                        "preview": value.head(5).to_dict(),
                        "columns": list(value.columns)
                    }
                elif isinstance(value, np.ndarray):
                    # For NumPy arrays, capture shape and preview
                    variables[name] = {
                        "type": "ndarray",
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "preview": value.tolist() if value.size < 100 else "Array too large to display"
                    }
                elif isinstance(value, (pd.Series, list, dict, str, int, float, bool)):
                    # For basic types, capture directly
                    variables[name] = {
                        "type": type(value).__name__,
                        "value": value if not isinstance(value, pd.Series) else value.to_dict()
                    }
                else:
                    # For other objects, just capture the type
                    variables[name] = {
                        "type": type(value).__name__,
                        "summary": str(value)[:100] + ("..." if len(str(value)) > 100 else "")
                    }
            except Exception as e:
                variables[name] = {
                    "type": type(value).__name__,
                    "error": f"Could not capture: {str(e)}"
                }
        
        return variables
    
    def execute_data_analysis(self, 
                             file_data: Dict[str, Any],
                             analysis_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute data analysis on an uploaded file
        
        Args:
            file_data: File data from FileUploadProcessor
            analysis_code: Optional custom analysis code
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Check if file data is valid
            if not file_data or "data_type" not in file_data:
                return {
                    "success": False,
                    "error": "Invalid file data",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Prepare data for different file types
            data_type = file_data.get("data_type", "")
            
            if data_type == "tabular" and "data" in file_data:
                # For tabular data (CSV, Excel)
                df = file_data["data"]
                
                # Generate default analysis code if not provided
                if not analysis_code:
                    analysis_code = self._generate_tabular_analysis(df, file_data)
                
                # Execute the analysis
                results = self.execute_code(
                    code=analysis_code,
                    data={"file": file_data}
                )
                
                return results
                
            elif data_type in ["text", "document"] and "text_content" in file_data:
                # For text data (TXT, PDF)
                text = file_data["text_content"]
                
                # Generate default analysis code if not provided
                if not analysis_code:
                    analysis_code = self._generate_text_analysis(text, file_data)
                
                # Execute the analysis
                results = self.execute_code(
                    code=analysis_code,
                    data={"file": file_data}
                )
                
                return results
                
            else:
                return {
                    "success": False,
                    "error": f"Unsupported data type: {data_type}",
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error in execute_data_analysis: {str(e)}")
            return {
                "success": False,
                "error": f"Error executing data analysis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_tabular_analysis(self, df: pd.DataFrame, file_data: Dict[str, Any]) -> str:
        """
        Generate default analysis code for tabular data
        
        Args:
            df: DataFrame
            file_data: File metadata
            
        Returns:
            Python code string
        """
        # Get file info
        file_name = file_data.get("file_name", "data")
        
        # Generate analysis code
        code = f"""
# Analysis of {file_name}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Access the DataFrame
df = file['data']

# Display basic information
print("Dataset Information:")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\\nColumn Data Types:")
print(df.dtypes)

# Display summary statistics
print("\\nSummary Statistics:")
print(df.describe().T)

# Check for missing values
print("\\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found")

# For numerical columns, create histograms
num_cols = df.select_dtypes(include=['number']).columns[:5]  # First 5 numerical columns
if len(num_cols) > 0:
    print("\\nGenerating histograms for numerical columns...")
    for col in num_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

# For categorical columns, create bar plots
cat_cols = df.select_dtypes(include=['object', 'category']).columns[:3]  # First 3 categorical columns
if len(cat_cols) > 0:
    print("\\nGenerating bar plots for categorical columns...")
    for col in cat_cols:
        plt.figure(figsize=(10, 6))
        top_cats = df[col].value_counts().head(10)
        sns.barplot(x=top_cats.index, y=top_cats.values)
        plt.title(f"Top 10 values in {col}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# If there are at least 2 numerical columns, create a correlation matrix
if len(num_cols) >= 2:
    print("\\nCorrelation Matrix:")
    corr = df[num_cols].corr()
    print(corr)
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
               square=True, linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

print("\\nAnalysis complete!")
"""
        return code
    
    def _generate_text_analysis(self, text: str, file_data: Dict[str, Any]) -> str:
        """
        Generate default analysis code for text data
        
        Args:
            text: Text content
            file_data: File metadata
            
        Returns:
            Python code string
        """
        # Get file info
        file_name = file_data.get("file_name", "text")
        
        # Generate analysis code
        code = f"""
# Analysis of {file_name}
import re
import collections
import matplotlib.pyplot as plt
import seaborn as sns

# Access the text content
text = file['text_content']

# Basic text statistics
print("Text Statistics:")
char_count = len(text)
word_count = len(text.split())
sentence_count = len(re.split(r'[.!?]+', text))
print(f"Character count: {char_count}")
print(f"Word count: {word_count}")
print(f"Sentence count: {sentence_count}")
print(f"Average word length: {char_count / max(1, word_count):.2f} characters")
print(f"Average sentence length: {word_count / max(1, sentence_count):.2f} words")

# Word frequency analysis
print("\\nWord Frequency Analysis:")
words = re.findall(r'\\b[\\w\\']+\\b', text.lower())
stop_words = set(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'at', 'by', 'an', 'this', 'that', 'it', 'from'])
filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
word_freq = collections.Counter(filtered_words)
print("Most common words:")
for word, count in word_freq.most_common(20):
    print(f"  {word}: {count}")

# Plot word frequency
plt.figure(figsize=(12, 6))
top_words = dict(word_freq.most_common(15))
sns.barplot(x=list(top_words.keys()), y=list(top_words.values()))
plt.title("Top 15 Words by Frequency")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Word length distribution
word_lengths = [len(word) for word in filtered_words]
plt.figure(figsize=(10, 6))
sns.histplot(word_lengths, bins=range(1, 20), kde=True)
plt.title("Word Length Distribution")
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\\nAnalysis complete!")
"""
        return code
