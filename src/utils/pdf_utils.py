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
