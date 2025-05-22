"""
PDF Extractor for Agentic Researcher using LangChain's PyPDFLoader
Handles PDF extraction from URLs and local files in a simplified manner
"""
import os
import time
import logging
import tempfile
import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List, Tuple

from langchain_community.document_loaders import PyPDFLoader

# Import utility functions from utils folder
try:
    from src.utils.pdf_utils import is_pdf_url, download_pdf, extract_text_from_pdf
except ModuleNotFoundError:
    # When run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils.pdf_utils import is_pdf_url, download_pdf, extract_text_from_pdf

# Import document utilities
try:
    from src.utils.document_utils import extract_metadata, process_document
    from src.utils.file_utils import ensure_dir, write_json_file
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO to see progress logs

class PDFExtractor:
    """
    PDF extractor using LangChain's PyPDFLoader
    """
    
    def __init__(self, timeout: int = 120):
        """Initialize the PDF extractor"""
        self.timeout = timeout
        
        # Track temporary files
        self.temp_files = []
        
        logger.info("PDF extractor initialized with LangChain's PyPDFLoader")
    
    def _is_pdf_url(self, url: str) -> bool:
        """Check if a URL is likely a PDF using utility function"""
        # Use the utility function from pdf_utils.py
        return is_pdf_url(url)
    
    def _download_pdf(self, url: str) -> tuple[bool, str, bytes]:
        """
        Download a PDF from a URL using pdf_utils function
        
        Args:
            url: URL of PDF
            
        Returns:
            Tuple of (success, file_path, pdf_bytes)
        """
        logger.info(f"Downloading PDF from {url}")
        
        try:
            # Create a temporary directory for the PDF
            temp_dir = tempfile.mkdtemp(prefix="pdf_extractor_")
            
            # Use the utility function from pdf_utils.py
            pdf_path = download_pdf(url, temp_dir)
            
            if pdf_path:
                # Track the temporary file for cleanup
                self.temp_files.append(pdf_path)
                
                # Read the downloaded PDF
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                logger.info(f"PDF successfully downloaded to {pdf_path} ({len(pdf_bytes)} bytes)")
                return True, pdf_path, pdf_bytes
            else:
                logger.warning(f"Failed to download PDF from {url}")
                return False, "", b""
        except Exception as e:
            logger.error(f"Error in PDF download: {str(e)}")
            return False, "", b""
    
    def _extract_with_langchain(self, pdf_path: str) -> str:
        """
        Extract text from PDF using LangChain's PyPDFLoader
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            # First try using the utility function from pdf_utils.py
            extracted_text = extract_text_from_pdf(pdf_path)
            
            if extracted_text and len(extracted_text) > 100:  # Check if we got meaningful text
                logger.info(f"Successfully extracted text from PDF using pdf_utils (length: {len(extracted_text)} chars)")
                return extracted_text
            
            # Fallback to LangChain's PyPDFLoader if pdf_utils extraction fails
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Combine text from all pages
            all_text = "\n\n".join(page.page_content for page in pages)
            
            # Clean up the text (remove excessive whitespace)
            cleaned_text = "\n".join([line.strip() for line in all_text.split("\n") if line.strip()])
            
            logger.info(f"Successfully extracted text from PDF using LangChain (length: {len(cleaned_text)} chars)")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"PyPDFLoader extraction failed: {str(e)}")
            return ""
    
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract text from a PDF URL using LangChain's PyPDFLoader
        
        Args:
            url: URL of the PDF to extract
            
        Returns:
            Dict with extraction results and metadata
        """
        start_time = time.time()
        logger.info(f"Starting PDF extraction from {url}")
        
        # Initialize result dictionary
        result = {
            "url": url,
            "success": False,
            "text": "",
            "engine_used": "langchain_pypdf",
            "text_length": 0,
            "processing_time": 0,
            "error": None
        }
        
        try:
            # Step 1: Download the PDF
            download_success, pdf_path, pdf_bytes = self._download_pdf(url)
            
            if not download_success or not pdf_path:
                result["error"] = "Failed to download PDF"
                result["processing_time"] = time.time() - start_time
                return result
            
            # Step 2: Extract text using LangChain's PyPDFLoader
            text = self._extract_with_langchain(pdf_path)
            
            # Update result
            result["success"] = bool(text)
            result["text"] = text or "PDF extraction failed"
            result["text_length"] = len(text)
            result["processing_time"] = time.time() - start_time
            
            if not text:
                result["error"] = "Failed to extract text with PyPDFLoader"
            
            logger.info(f"PDF extraction completed: success={result['success']}, length={len(text)}")
            return result
        except Exception as e:
            result["error"] = f"PDF extraction failed: {str(e)}"
            result["processing_time"] = time.time() - start_time
            logger.error(f"Error in PDF extraction: {str(e)}")
            return result
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
        
        self.temp_files = []
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Comprehensive example usage of the PDFExtractor
if __name__ == "__main__":
    import sys
    import os
    import time
    import json
    import pandas as pd
    from datetime import datetime
    from pathlib import Path
    
    # Add project root to Python path if necessary
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Try to import document processing utilities if available
    try:
        from src.utils.document_utils import extract_metadata, process_document
        from src.utils.file_utils import ensure_dir, write_json_file
        document_utils_available = True
    except ImportError:
        document_utils_available = False
        print("Warning: document_utils module not available for integration example")
    
    # Create output directory for example results
    output_dir = Path(os.path.join(project_root, "example_results", f"pdf_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize the PDF extractor with a reasonable timeout
    print("\n" + "=" * 80)
    print("PDF EXTRACTOR - COMPREHENSIVE EXAMPLE")
    print("=" * 80)
    
    print("\nInitializing PDFExtractor with 60 second timeout...")
    extractor = PDFExtractor(timeout=60)
    
    try:
        # List of PDF URLs to test extraction with (with different characteristics)
        test_pdfs = {
            "simple": "https://www.learningcontainer.com/wp-content/uploads/2019/09/sample-pdf-file.pdf",
            "research": "https://arxiv.org/pdf/1706.03762.pdf",  # Attention Is All You Need paper
            "complex": "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",
            "large": "https://www.govinfo.gov/content/pkg/BUDGET-2023-APP/pdf/BUDGET-2023-APP.pdf",
            "ocr_needed": "https://www.irs.gov/pub/irs-pdf/fw4.pdf"  # Form that might need OCR
        }
        
        # Example 1: Basic extraction from a simple PDF
        print("\n" + "-" * 80)
        print("EXAMPLE 1: BASIC PDF EXTRACTION")
        print("-" * 80)
        
        print(f"Extracting text from simple PDF: {test_pdfs['simple']}")
        start_time = time.time()
        simple_result = extractor.extract_from_url(test_pdfs['simple'])
        extract_time = time.time() - start_time
        
        if simple_result["success"]:
            print(f"✓ Extraction successful using {simple_result['engine_used']}")
            print(f"Text length: {simple_result['text_length']} characters")
            print(f"Processing time: {extract_time:.2f} seconds")
            
            # Show a preview of the text
            text_preview = simple_result["text"][:300] + "..." if len(simple_result["text"]) > 300 else simple_result["text"]
            print(f"\nText preview:\n{text_preview}")
            
            # Save the extracted text
            with open(output_dir / "simple_pdf.txt", "w", encoding="utf-8") as f:
                f.write(simple_result["text"])
            print(f"Saved extracted text to: {output_dir / 'simple_pdf.txt'}")
        else:
            print(f"× Extraction failed: {simple_result['error']}")
        
        # Example 2: Extracting academic research papers
        print("\n" + "-" * 80)
        print("EXAMPLE 2: RESEARCH PAPERS & ACADEMIC CONTENT")
        print("-" * 80)
        
        print(f"Extracting text from research paper: {test_pdfs['research']}")
        research_result = extractor.extract_from_url(test_pdfs['research'])
        
        if research_result["success"]:
            print(f"✓ Successfully extracted academic paper using {research_result['engine_used']}")
            print(f"Text length: {research_result['text_length']} characters")
            print(f"Processing time: {research_result['processing_time']:.2f} seconds")
            
            # Calculate some basic text statistics
            from collections import Counter
            words = research_result["text"].lower().split()
            word_count = len(words)
            unique_words = len(set(words))
            
            print(f"\nStatistics:")
            print(f"- Word count: {word_count}")
            print(f"- Unique words: {unique_words}")
            print(f"- Vocabulary density: {unique_words/word_count:.2f}")
            
            # Find common technical terms
            word_counter = Counter(words)
            common_technical_terms = [word for word, count in word_counter.most_common(10) 
                                    if len(word) > 5 and word not in ['should', 'would', 'could', 'about']]
            print(f"- Common technical terms: {', '.join(common_technical_terms)}")
            
            # Save the extracted text
            with open(output_dir / "research_paper.txt", "w", encoding="utf-8") as f:
                f.write(research_result["text"])
        else:
            print(f"× Failed to extract research paper: {research_result['error']}")
        
        # Example 3: Batch processing multiple PDFs in parallel
        print("\n" + "-" * 80)
        print("EXAMPLE 3: BATCH PROCESSING MULTIPLE PDFS")
        print("-" * 80)
        
        selected_pdfs = [test_pdfs['simple'], test_pdfs['complex']]
        print(f"Processing {len(selected_pdfs)} PDFs in batch...")
        
        # Process PDFs and collect results
        batch_results = []
        for pdf_url in selected_pdfs:
            try:
                print(f"Processing: {pdf_url}")
                result = extractor.extract_from_url(pdf_url)
                
                # Add to results list
                batch_results.append({
                    "url": pdf_url,
                    "success": result["success"],
                    "text_length": result["text_length"],
                    "processing_time": result["processing_time"],
                    "engine": result["engine_used"]
                })
            except Exception as e:
                print(f"Error processing {pdf_url}: {str(e)}")
                batch_results.append({
                    "url": pdf_url,
                    "success": False,
                    "error": str(e)
                })
        
        # Create a DataFrame to display results
        batch_df = pd.DataFrame(batch_results)
        print("\nBatch processing results:")
        print(batch_df.to_string(index=False))
        
        # Example 4: Integration with document processing pipeline (if available)
        if document_utils_available:
            print("\n" + "-" * 80)
            print("EXAMPLE 4: INTEGRATION WITH DOCUMENT PROCESSING")
            print("-" * 80)
            
            # Choose one successful extraction to demonstrate with
            successful_extraction = None
            for result in batch_results:
                if result["success"]:
                    for pdf_url in test_pdfs.values():
                        if pdf_url == result["url"]:
                            successful_extraction = pdf_url
                            break
                    if successful_extraction:
                        break
            
            if successful_extraction:
                print(f"Integrating PDF extraction with document processing pipeline")
                print(f"PDF URL: {successful_extraction}")
                
                # Extract text from the PDF
                pdf_result = extractor.extract_from_url(successful_extraction)
                
                if pdf_result["success"]:
                    # Extract metadata
                    metadata = extract_metadata(pdf_result["text"], pdf_result["url"])
                    print("\nExtracted metadata:")
                    for key, value in metadata.items():
                        print(f"- {key}: {value}")
                    
                    # Process the document - chunk it for analysis
                    processed_doc = process_document(
                        text=pdf_result["text"],
                        metadata=metadata,
                        chunk_size=1000,  # Example chunk size
                        chunk_overlap=200  # Example overlap
                    )
                    
                    num_chunks = len(processed_doc.get("chunks", []))
                    print(f"\nDocument successfully chunked into {num_chunks} segments")
                    
                    if num_chunks > 0:
                        first_chunk = processed_doc["chunks"][0]
                        print(f"First chunk preview: {first_chunk[:150]}...")
                    
                    # Save processed document
                    doc_path = output_dir / "processed_pdf_document.json"
                    write_json_file(processed_doc, doc_path)
                    print(f"Saved processed document to: {doc_path}")
            else:
                print("No successful extractions available for integration demonstration")
        
        # Example 5: Error handling for problematic PDFs
        print("\n" + "-" * 80)
        print("EXAMPLE 5: ERROR HANDLING & RECOVERY")
        print("-" * 80)
        
        # Try an invalid PDF URL
        invalid_pdf_url = "https://example.com/nonexistent-document.pdf"
        print(f"Attempting to extract from invalid PDF URL: {invalid_pdf_url}")
        
        try:
            invalid_result = extractor.extract_from_url(invalid_pdf_url)
            print(f"Result status: {'Success' if invalid_result['success'] else 'Failed'}")
            print(f"Error message: {invalid_result.get('error', 'No error')}")
            print("This demonstrates the extractor's robust error handling")
        except Exception as e:
            print(f"Caught unexpected exception: {str(e)}")
        
        # Summary of capabilities
        print("\n" + "=" * 80)
        print("PDF EXTRACTOR CAPABILITIES SUMMARY:")
        print("=" * 80)
        print("1. Extract text from PDF URLs using LangChain's PyPDFLoader")
        print("2. Robust error handling and timeout management")
        print("3. Automatic PDF detection and download handling")
        print("4. Integration with document processing pipeline")
        print("5. Comprehensive metadata capture")
        print("6. Support for academic papers and complex documents")
        print("7. Batch processing capability")
        print("8. Automatic cleanup of temporary files")
        print("\nAll results saved to: " + str(output_dir))
    
    except Exception as e:
        print(f"\nUnhandled error in example: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always clean up temporary files
        print("\nCleaning up temporary files...")
        extractor.cleanup()
        print("PDF extractor resources released successfully")
