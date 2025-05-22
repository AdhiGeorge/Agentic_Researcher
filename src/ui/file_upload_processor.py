"""
File Upload Processor for Agentic Researcher

This module handles file uploads in the Streamlit UI, including:
- Processing various file types (CSV, Excel, JSON, PDF, etc.)
- Extracting and structuring data for analysis
- Providing data summaries and statistics
"""

import os
import io
import json
import logging
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class FileUploadProcessor:
    """
    Handles file uploads and processes them based on file type
    
    This class provides:
    1. File validation and type detection
    2. Data extraction based on file type
    3. Basic statistics and data summaries
    4. Structured data conversion for analysis
    """
    
    def __init__(self, upload_dir: Optional[str] = None):
        """
        Initialize the file upload processor
        
        Args:
            upload_dir: Directory to store uploaded files (temp dir if None)
        """
        self.upload_dir = upload_dir or os.path.join(tempfile.gettempdir(), "agentic_researcher_uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        logger.info(f"FileUploadProcessor initialized with upload dir: {self.upload_dir}")
        
    def process_upload(self, uploaded_file) -> Dict[str, Any]:
        """
        Process an uploaded file based on its type
        
        Args:
            uploaded_file: The Streamlit uploaded file object
            
        Returns:
            Dictionary with processing results including data and metadata
        """
        if uploaded_file is None:
            return {"error": "No file uploaded"}
        
        try:
            # Extract file metadata
            file_name = uploaded_file.name
            file_type = uploaded_file.type
            file_size = uploaded_file.size
            
            logger.info(f"Processing uploaded file: {file_name} ({file_type}, {file_size} bytes)")
            
            # Process based on file type
            if file_type == "text/csv" or file_name.endswith(".csv"):
                return self._process_csv(uploaded_file)
                
            elif "spreadsheet" in file_type or any(file_name.endswith(ext) for ext in [".xlsx", ".xls"]):
                return self._process_excel(uploaded_file)
                
            elif file_type == "application/json" or file_name.endswith(".json"):
                return self._process_json(uploaded_file)
                
            elif file_type == "application/pdf" or file_name.endswith(".pdf"):
                return self._process_pdf(uploaded_file)
                
            elif file_type == "text/plain" or file_name.endswith(".txt"):
                return self._process_text(uploaded_file)
                
            else:
                # Generic file handling
                return self._process_generic_file(uploaded_file)
                
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            return {
                "error": f"Error processing file: {str(e)}",
                "file_name": uploaded_file.name if uploaded_file else "Unknown",
                "timestamp": datetime.now().isoformat()
            }
    
    def _process_csv(self, uploaded_file) -> Dict[str, Any]:
        """Process CSV file and extract data"""
        try:
            # Read CSV into pandas DataFrame
            df = pd.read_csv(uploaded_file)
            
            # Save a copy of the file
            file_path = os.path.join(self.upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Get basic statistics and info
            stats = self._get_dataframe_stats(df)
            
            return {
                "success": True,
                "file_name": uploaded_file.name,
                "file_type": "csv",
                "file_path": file_path,
                "data_type": "tabular",
                "data": df,
                "shape": df.shape,
                "stats": stats,
                "preview_html": df.head(10).to_html(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            return {"error": f"Error processing CSV file: {str(e)}"}
    
    def _process_excel(self, uploaded_file) -> Dict[str, Any]:
        """Process Excel file and extract data"""
        try:
            # Read Excel file into pandas DataFrame
            df_dict = pd.read_excel(uploaded_file, sheet_name=None)
            
            # Save a copy of the file
            file_path = os.path.join(self.upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process each sheet
            sheets_data = {}
            for sheet_name, df in df_dict.items():
                sheets_data[sheet_name] = {
                    "data": df,
                    "shape": df.shape,
                    "stats": self._get_dataframe_stats(df),
                    "preview_html": df.head(10).to_html()
                }
            
            return {
                "success": True,
                "file_name": uploaded_file.name,
                "file_type": "excel",
                "file_path": file_path,
                "data_type": "multi_tabular",
                "sheets": list(df_dict.keys()),
                "sheets_data": sheets_data,
                "active_sheet": list(df_dict.keys())[0],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            return {"error": f"Error processing Excel file: {str(e)}"}
    
    def _process_json(self, uploaded_file) -> Dict[str, Any]:
        """Process JSON file and extract data"""
        try:
            # Read JSON file
            content = uploaded_file.read()
            string_data = content.decode('utf-8')
            json_data = json.loads(string_data)
            
            # Save a copy of the file
            file_path = os.path.join(self.upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Determine data structure
            data_type = "unknown"
            if isinstance(json_data, list):
                data_type = "list"
                # Try to convert to DataFrame if it's a list of objects
                if all(isinstance(item, dict) for item in json_data):
                    try:
                        df = pd.DataFrame(json_data)
                        return {
                            "success": True,
                            "file_name": uploaded_file.name,
                            "file_type": "json",
                            "file_path": file_path,
                            "data_type": "tabular",
                            "data": df,
                            "shape": df.shape,
                            "stats": self._get_dataframe_stats(df),
                            "preview_html": df.head(10).to_html(),
                            "raw_json": json_data,
                            "timestamp": datetime.now().isoformat()
                        }
                    except:
                        pass
            elif isinstance(json_data, dict):
                data_type = "dict"
            
            return {
                "success": True,
                "file_name": uploaded_file.name,
                "file_type": "json",
                "file_path": file_path,
                "data_type": data_type,
                "data": json_data,
                "structure_summary": self._summarize_json_structure(json_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing JSON file: {str(e)}")
            return {"error": f"Error processing JSON file: {str(e)}"}
    
    def _process_pdf(self, uploaded_file) -> Dict[str, Any]:
        """Process PDF file and extract text content"""
        try:
            # Save the PDF file first
            file_path = os.path.join(self.upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Try to import PDF extraction capabilities
            try:
                # Check for different PDF libraries
                extraction_method = "none"
                extracted_text = ""
                
                # Try PyMuPDF (fitz)
                try:
                    import fitz
                    extraction_method = "pymupdf"
                    pdf_document = fitz.open(file_path)
                    extracted_text = ""
                    for page_num in range(pdf_document.page_count):
                        page = pdf_document[page_num]
                        extracted_text += page.get_text()
                    pdf_document.close()
                except ImportError:
                    # Try pdfplumber
                    try:
                        import pdfplumber
                        extraction_method = "pdfplumber"
                        with pdfplumber.open(file_path) as pdf:
                            extracted_text = ""
                            for page in pdf.pages:
                                extracted_text += page.extract_text() or ""
                    except ImportError:
                        # Try pdfminer
                        try:
                            from pdfminer.high_level import extract_text as pm_extract_text
                            extraction_method = "pdfminer"
                            extracted_text = pm_extract_text(file_path)
                        except ImportError:
                            extraction_method = "none"
                            extracted_text = "PDF text extraction requires PyMuPDF, pdfplumber, or pdfminer libraries."
            
                return {
                    "success": True,
                    "file_name": uploaded_file.name,
                    "file_type": "pdf",
                    "file_path": file_path,
                    "data_type": "document",
                    "extraction_method": extraction_method,
                    "text_content": extracted_text,
                    "text_length": len(extracted_text),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as pdf_e:
                logger.error(f"Error extracting PDF text: {str(pdf_e)}")
                return {
                    "success": True,
                    "file_name": uploaded_file.name,
                    "file_type": "pdf",
                    "file_path": file_path,
                    "data_type": "document",
                    "extraction_error": str(pdf_e),
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            return {"error": f"Error processing PDF file: {str(e)}"}
    
    def _process_text(self, uploaded_file) -> Dict[str, Any]:
        """Process text file and extract content"""
        try:
            # Read text content
            content = uploaded_file.read()
            text_content = content.decode('utf-8')
            
            # Save a copy of the file
            file_path = os.path.join(self.upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Check if it might be a structured format
            is_csv = ',' in text_content and '\n' in text_content
            if is_csv:
                try:
                    # Try to parse as CSV
                    df = pd.read_csv(io.StringIO(text_content))
                    return {
                        "success": True,
                        "file_name": uploaded_file.name,
                        "file_type": "text/csv",
                        "file_path": file_path,
                        "data_type": "tabular",
                        "data": df,
                        "shape": df.shape,
                        "stats": self._get_dataframe_stats(df),
                        "preview_html": df.head(10).to_html(),
                        "timestamp": datetime.now().isoformat()
                    }
                except:
                    pass  # Not a valid CSV, continue as text
            
            return {
                "success": True,
                "file_name": uploaded_file.name,
                "file_type": "text",
                "file_path": file_path,
                "data_type": "text",
                "text_content": text_content,
                "text_length": len(text_content),
                "line_count": text_content.count('\n') + 1,
                "word_count": len(text_content.split()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            return {"error": f"Error processing text file: {str(e)}"}
    
    def _process_generic_file(self, uploaded_file) -> Dict[str, Any]:
        """Process generic file and save it"""
        try:
            # Save the file
            file_path = os.path.join(self.upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            return {
                "success": True,
                "file_name": uploaded_file.name,
                "file_type": "binary",
                "file_path": file_path,
                "data_type": "binary",
                "file_size_bytes": uploaded_file.size,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing generic file: {str(e)}")
            return {"error": f"Error processing file: {str(e)}"}
    
    def _get_dataframe_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics for a DataFrame"""
        try:
            # Get column types
            column_types = df.dtypes.astype(str).to_dict()
            
            # Get basic statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_stats = {}
            if numeric_columns:
                numeric_stats = df[numeric_columns].describe().to_dict()
            
            # Get nulls count
            null_counts = df.isnull().sum().to_dict()
            
            # Get unique counts for categorical columns
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
            unique_counts = {}
            for col in categorical_columns:
                unique_counts[col] = df[col].nunique()
            
            return {
                "column_types": column_types,
                "numeric_stats": numeric_stats,
                "null_counts": null_counts,
                "unique_counts": unique_counts,
                "row_count": len(df),
                "column_count": len(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Error getting DataFrame stats: {str(e)}")
            return {"error": f"Error calculating statistics: {str(e)}"}
    
    def _summarize_json_structure(self, json_data, max_depth=3, current_depth=0) -> Dict[str, Any]:
        """Create a summary of the JSON structure"""
        if current_depth >= max_depth:
            return {"type": type(json_data).__name__, "truncated": True}
        
        if isinstance(json_data, dict):
            result = {"type": "dict", "keys": {}}
            for key, value in list(json_data.items())[:10]:  # Limit to first 10 keys
                result["keys"][key] = self._summarize_json_structure(value, max_depth, current_depth + 1)
            if len(json_data) > 10:
                result["truncated"] = True
            return result
        
        elif isinstance(json_data, list):
            result = {"type": "list", "length": len(json_data)}
            if json_data and current_depth < max_depth - 1:
                # Sample first element
                result["sample"] = self._summarize_json_structure(json_data[0], max_depth, current_depth + 1)
            return result
        
        else:
            return {"type": type(json_data).__name__}


# Example usage
if __name__ == "__main__":
    processor = FileUploadProcessor()
    print(f"Initialized file upload processor with upload directory: {processor.upload_dir}")
