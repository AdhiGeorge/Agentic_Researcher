"""
PDF utility functions for Agentic Researcher
Handles PDF downloading, extraction, and processing
"""
import os
import logging
import urllib.parse
import requests
from typing import Optional, Dict, Any
from pathlib import Path
import pdfplumber
from requests.exceptions import RequestException

# Configure logging
logger = logging.getLogger(__name__)

def is_pdf_url(url: str) -> bool:
    """
    Check if the URL is likely to be a PDF
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if URL appears to be a PDF, False otherwise
    """
    # Check if URL ends with .pdf
    if url.lower().endswith('.pdf'):
        return True
    
    # Check if URL contains pdf in the path or query parameters
    parsed_url = urllib.parse.urlparse(url)
    if 'pdf' in parsed_url.path.lower() or 'pdf' in parsed_url.query.lower():
        # Additional check for academic paper repositories that commonly serve PDFs
        academic_domains = [
            'arxiv.org', 'researchgate.net', 'academia.edu', 'ssrn.com',
            'citeseerx.ist.psu.edu', 'sciencedirect.com', 'springer.com',
            'ieee.org', 'acm.org', 'jstor.org', 'nature.com', 'science.org'
        ]
        
        domain = parsed_url.netloc.lower()
        for academic_domain in academic_domains:
            if academic_domain in domain:
                return True
    
    return False

def download_pdf(url: str, save_dir: str, filename: Optional[str] = None) -> Optional[str]:
    """
    Download a PDF from a URL and save it locally
    
    Args:
        url: URL of the PDF to download
        save_dir: Directory to save the PDF in
        filename: Optional filename to use (if None, derives from URL)
        
    Returns:
        str: Path to the downloaded PDF, or None if download failed
    """
    # Ensure PDF directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        # Get filename from URL or generate a unique name based on URL hash
        parsed_url = urllib.parse.urlparse(url)
        path_components = parsed_url.path.split('/')
        
        if path_components and path_components[-1]:
            # Use the last component of the URL path as the filename
            base_filename = path_components[-1]
            # Remove query parameters if present
            base_filename = base_filename.split('?')[0]
            # Ensure the extension is .pdf
            if not base_filename.lower().endswith('.pdf'):
                base_filename += '.pdf'
        else:
            # Create a filename based on the URL domain if path is empty
            domain = parsed_url.netloc.replace('.', '_')
            base_filename = f"{domain}_{hash(url) % 10000:04d}.pdf"
    else:
        base_filename = filename
        if not base_filename.lower().endswith('.pdf'):
            base_filename += '.pdf'
    
    # Create full path
    pdf_path = os.path.join(save_dir, base_filename)
    
    try:
        # Set up the request with headers to appear as a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Download the PDF
        logger.info(f"Downloading PDF from {url}")
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save the PDF to disk
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded PDF to {pdf_path}")
        return pdf_path
    
    except RequestException as e:
        logger.error(f"Failed to download PDF from {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error saving PDF from {url}: {str(e)}")
        return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text content from the PDF
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    text_content = ""
    
    try:
        # Open the PDF file
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from each page
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_content += page_text + "\n\n"
        
        logger.info(f"Successfully extracted {len(text_content)} characters from PDF")
        return text_content
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return f"Error extracting PDF text: {str(e)}"


def extract_text_with_pymupdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz) library
    Often better for academic papers and complex layouts
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        import fitz  # PyMuPDF
        text_content = ""
        
        with fitz.open(pdf_path) as doc:
            # Get metadata
            metadata = doc.metadata
            if metadata:
                logger.info(f"PDF Metadata: Title: {metadata.get('title', 'Unknown')}, Author: {metadata.get('author', 'Unknown')}")
            
            # Extract text from each page
            for page in doc:
                text_content += page.get_text() + "\n\n"
                
        return text_content
        
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed. Falling back to pdfplumber.")
        return extract_text_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error extracting text with PyMuPDF: {str(e)}")
        return f"Error with PyMuPDF extraction: {str(e)}"


def extract_text_with_pdfminer(pdf_path: str) -> str:
    """
    Extract text from PDF using pdfminer.six library
    Often better for preserving text layout and order
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path)
    except ImportError:
        logger.warning("pdfminer.six not installed. Falling back to pdfplumber.")
        return extract_text_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error extracting text with pdfminer: {str(e)}")
        return f"Error with pdfminer extraction: {str(e)}"


def extract_with_ocr(pdf_path: str) -> str:
    """
    Extract text from PDF using OCR with pytesseract
    Best for scanned documents or image-based PDFs
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        import fitz  # PyMuPDF for PDF to image conversion
        import pytesseract
        from PIL import Image
        import tempfile
        import os
        
        text_content = ""
        
        # Open the PDF
        with fitz.open(pdf_path) as doc:
            # Process each page
            for page_num, page in enumerate(doc):
                # Get page as an image
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                
                # Create a temporary image file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    pix.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                # Use OCR to extract text from the image
                img = Image.open(tmp_path)
                page_text = pytesseract.image_to_string(img)
                text_content += page_text + "\n\n"
                
                # Clean up the temporary file
                os.unlink(tmp_path)
                
        return text_content
        
    except ImportError as e:
        lib_name = str(e).split("'")[-2] if "'" in str(e) else "required libraries"
        logger.warning(f"{lib_name} not installed for OCR. Falling back to standard extraction.")
        return extract_text_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error extracting text with OCR: {str(e)}")
        return f"Error with OCR extraction: {str(e)}"


# Example usage with real API calls
if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    
    print("===== PDF Utilities Example Usage =====")
    
    # Create a temporary directory for our downloaded PDFs
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="pdf_utils_example_")
    print(f"Created temporary directory: {temp_dir}")
    
    # Example URLs to academic PDFs
    pdf_urls = [
        "https://arxiv.org/pdf/2303.08774.pdf",  # ArXiv paper
        "https://www.jmlr.org/papers/volume18/17-468/17-468.pdf",  # JMLR paper
    ]
    
    print("\nExample 1: Download and process a PDF from ArXiv")
    try:
        # Download the PDF
        print(f"Downloading PDF from {pdf_urls[0]}")
        pdf_path = download_pdf(pdf_urls[0], temp_dir)
        
        if pdf_path:
            print(f"Successfully downloaded PDF to {pdf_path}")
            
            # Extract text with pdfplumber (default method)
            start_time = time.time()
            text = extract_text_from_pdf(pdf_path)
            duration = time.time() - start_time
            
            print(f"\nExtracted {len(text)} characters in {duration:.2f} seconds")
            print("First 300 characters of extracted text:")
            # Fix f-string syntax error by moving the replacement outside the f-string
            preview_text = text[:300].replace('\n', ' ')
            print(f"\"{preview_text}...\"")


            
            # Example with different extraction methods
            print("\nExample 2: Comparing different PDF extraction engines")
            
            # Try extraction with PyMuPDF if available
            print("\n1. Extracting with PyMuPDF (fitz):")
            try:
                start_time = time.time()
                pymupdf_text = extract_text_with_pymupdf(pdf_path)
                duration = time.time() - start_time
                print(f"Extracted {len(pymupdf_text)} characters in {duration:.2f} seconds")
                # Fix f-string syntax error
                pymupdf_preview = pymupdf_text[:100].replace('\n', ' ')
                print(f"First 100 characters: \"{pymupdf_preview}...\"")
            except ImportError:
                print("PyMuPDF not available")
            
            # Try extraction with pdfminer if available
            print("\n2. Extracting with pdfminer.six:")
            try:
                start_time = time.time()
                pdfminer_text = extract_text_with_pdfminer(pdf_path)
                duration = time.time() - start_time
                print(f"Extracted {len(pdfminer_text)} characters in {duration:.2f} seconds")
                # Fix f-string syntax error
                pdfminer_preview = pdfminer_text[:100].replace('\n', ' ')
                print(f"First 100 characters: \"{pdfminer_preview}...\"")
            except ImportError:
                print("pdfminer.six not available")
            
            # If OCR libraries are installed, demonstrate OCR
            print("\n3. OCR extraction (for scanned documents):")
            try:
                import pytesseract
                # OCR is only useful for scanned documents
                # We'll only test if the library is available
                print("pytesseract available for OCR extraction")
                print("(Not demonstrating OCR on already-digital PDF)")
            except ImportError:
                print("pytesseract not available for OCR")
    
    except Exception as e:
        print(f"Error in Example 1: {str(e)}")
    
    # Example 3: Check multiple PDF URLs
    print("\nExample 3: Checking if URLs are likely PDFs")
    
    test_urls = [
        "https://arxiv.org/pdf/2303.08774.pdf",
        "https://arxiv.org/abs/2303.08774",  # Not a direct PDF, but a page that links to a PDF
        "https://www.nature.com/articles/s41586-020-2649-2",  # Academic article
        "https://example.com/sample.pdf",
        "https://example.com/article?format=pdf",
        "https://github.com/microsoft/playwright/blob/main/README.md"  # Not a PDF
    ]
    
    for url in test_urls:
        is_pdf = is_pdf_url(url)
        print(f"URL: {url}\nIs likely PDF? {is_pdf}\n")
    
    # Clean up - keep files if user wants to inspect them
    print(f"\nExample files can be found in: {temp_dir}")
    print("You can delete this directory when finished exploring.")
    print("\nPDF utility examples completed!")
