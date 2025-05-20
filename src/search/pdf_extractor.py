"""
Advanced PDF Extractor for Agentic Researcher
Handles any PDF extraction with multiple fallback mechanisms
"""
import os
import io
import re
import time
import logging
import tempfile
import requests
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional, Tuple, Union

# Import PDF libraries with fallbacks
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfparser import PDFParser
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    from pdf2image import convert_from_path, convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO to see progress logs

class AdvancedPDFExtractor:
    """
    Advanced PDF extractor that can handle any PDF format with multiple fallbacks
    """
    
    def __init__(self, timeout: int = 120):
        """Initialize the advanced PDF extractor with fallback mechanisms"""
        self.timeout = timeout
        
        # Track available engines
        self.engines = []
        if PYMUPDF_AVAILABLE:
            self.engines.append("pymupdf")
        if PDFPLUMBER_AVAILABLE:
            self.engines.append("pdfplumber")
        if PDFMINER_AVAILABLE:
            self.engines.append("pdfminer")
        if OCR_AVAILABLE:
            self.engines.append("ocr")
        
        # Track temporary files
        self.temp_files = []
        
        logger.info(f"Advanced PDF extractor initialized with engines: {', '.join(self.engines)}")
    
    def _is_pdf_url(self, url: str) -> bool:
        """Check if a URL is likely a PDF"""
        url_lower = url.lower()
        
        # Direct check for PDF extension
        if url_lower.endswith('.pdf'):
            return True
        
        # Check URL path and query parameters
        parsed_url = urlparse(url)
        
        # Check path component for PDF indicators
        if 'pdf' in parsed_url.path.lower():
            return True
            
        # Check query string for PDF indicators
        if 'pdf' in parsed_url.query.lower():
            return True
            
        # Check for academic and research domains that commonly serve PDFs
        domain = parsed_url.netloc.lower()
        pdf_domains = ['arxiv.org', 'researchgate.net', 'ieee.org', 'sciencedirect.com', 
                      'springer.com', 'jstor.org', 'ncbi.nlm.nih.gov', 'ssrn.com']
        
        if any(pd in domain for pd in pdf_domains):
            return True
            
        return False
    
    def _download_pdf(self, url: str) -> Tuple[bool, str, bytes]:
        """
        Download a PDF from a URL
        
        Args:
            url: URL of PDF
            
        Returns:
            Tuple of (success, file_path, pdf_bytes)
        """
        logger.info(f"Downloading PDF from {url}")
        
        try:
            # Advanced headers to avoid detection
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                'Accept': 'application/pdf,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': url,
                'DNT': '1',  # Do Not Track
                'Cache-Control': 'max-age=0',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Advanced download with retry and timeout handling
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Stream download for large files
                    response = requests.get(url, headers=headers, stream=True, timeout=30)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if retry == max_retries - 1:
                        logger.error(f"Failed to download PDF after {max_retries} retries: {str(e)}")
                        return False, "", b""
                    logger.warning(f"Retry {retry + 1}/{max_retries} downloading PDF: {str(e)}")
                    time.sleep(2 * (retry + 1))  # Exponential backoff
            
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if not any(pdf_type in content_type for pdf_type in ['pdf', 'octet-stream', 'application/']):
                # Try to determine if it's really a PDF by checking the first few bytes
                content_peek = next(response.iter_content(512), b"")
                if not content_peek.startswith(b'%PDF'):
                    logger.warning(f"Not a PDF file. Content type: {content_type}")
                    if b'<!DOCTYPE html>' in content_peek or b'<html' in content_peek:
                        logger.warning("Received HTML content instead of PDF")
                    return False, "", b""
            
            # Save to temp file
            fd, temp_path = tempfile.mkstemp(suffix='.pdf')
            self.temp_files.append(temp_path)
            
            # Read and save content
            pdf_bytes = b''
            with os.fdopen(fd, 'wb') as pdf_file:
                for chunk in response.iter_content(chunk_size=8192):
                    pdf_file.write(chunk)
                    pdf_bytes += chunk
            
            # Verify file size
            filesize = os.path.getsize(temp_path)
            if filesize < 100:  # Too small to be a valid PDF
                logger.warning(f"Downloaded file too small: {filesize} bytes")
                return False, "", b""
            
            logger.info(f"Successfully downloaded PDF ({filesize} bytes)")
            return True, temp_path, pdf_bytes
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            return False, "", b""
    
    def _extract_with_pymupdf(self, pdf_path: str, pdf_bytes: bytes = None) -> str:
        """Extract text using PyMuPDF (most reliable engine)"""
        try:
            logger.info("Extracting text with PyMuPDF")
            
            # Handle both file paths and byte content
            if pdf_bytes:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                doc = fitz.open(pdf_path)
            
            text_pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_pages.append(page.get_text())
            
            doc.close()
            
            # Process and clean the text
            combined_text = "\n\n".join(text_pages)
            cleaned_text = self._clean_text(combined_text)
            
            logger.info(f"Successfully extracted {len(cleaned_text)} characters with PyMuPDF")
            return cleaned_text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_path: str, pdf_bytes: bytes = None) -> str:
        """Extract text using pdfplumber"""
        try:
            logger.info("Extracting text with pdfplumber")
            
            # Handle both file paths and byte content
            if pdf_bytes:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    text_pages = [page.extract_text() or "" for page in pdf.pages]
            else:
                with pdfplumber.open(pdf_path) as pdf:
                    text_pages = [page.extract_text() or "" for page in pdf.pages]
            
            # Process and clean the text
            combined_text = "\n\n".join(text_pages)
            cleaned_text = self._clean_text(combined_text)
            
            logger.info(f"Successfully extracted {len(cleaned_text)} characters with pdfplumber")
            return cleaned_text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
            return ""
    
    def _extract_with_pdfminer(self, pdf_path: str, pdf_bytes: bytes = None) -> str:
        """Extract text using pdfminer"""
        try:
            logger.info("Extracting text with pdfminer")
            
            # Handle both file paths and byte content
            if pdf_bytes:
                extracted_text = pdfminer_extract_text(io.BytesIO(pdf_bytes))
            else:
                extracted_text = pdfminer_extract_text(pdf_path)
            
            # Clean the text
            cleaned_text = self._clean_text(extracted_text)
            
            logger.info(f"Successfully extracted {len(cleaned_text)} characters with pdfminer")
            return cleaned_text
        except Exception as e:
            logger.warning(f"pdfminer extraction failed: {str(e)}")
            return ""
    
    def _extract_with_ocr(self, pdf_path: str, pdf_bytes: bytes = None) -> str:
        """Extract text using OCR (for scanned PDFs)"""
        try:
            logger.info("Extracting text with OCR")
            
            # Convert PDF to images
            if pdf_bytes:
                images = convert_from_bytes(pdf_bytes)
            else:
                images = convert_from_path(pdf_path)
            
            # Extract text from each image using Tesseract OCR
            text_pages = []
            for img in images:
                text = pytesseract.image_to_string(img)
                text_pages.append(text)
            
            # Process and clean the text
            combined_text = "\n\n".join(text_pages)
            cleaned_text = self._clean_text(combined_text)
            
            logger.info(f"Successfully extracted {len(cleaned_text)} characters with OCR")
            return cleaned_text
        except Exception as e:
            logger.warning(f"OCR extraction failed: {str(e)}")
            return ""
    
    def _repair_pdf(self, pdf_path: str, pdf_bytes: bytes = None) -> Tuple[bool, str, bytes]:
        """
        Attempt to repair corrupted PDF files
        
        Args:
            pdf_path: Path to potentially corrupted PDF
            pdf_bytes: PDF content as bytes (optional)
            
        Returns:
            Tuple of (success, file_path, pdf_bytes)
        """
        try:
            logger.info("Attempting to repair PDF")
            
            # Read PDF bytes if not provided
            if not pdf_bytes and pdf_path:
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
            
            # Check if file starts with PDF header
            if not pdf_bytes.startswith(b'%PDF'):
                # Add PDF header if missing
                pdf_bytes = b'%PDF-1.4\n' + pdf_bytes
            
            # Check for EOF marker
            if not pdf_bytes.rstrip().endswith(b'%%EOF'):
                pdf_bytes = pdf_bytes + b'\n%%EOF'
            
            # Write repaired PDF to a new file
            fd, repaired_path = tempfile.mkstemp(suffix='.pdf')
            self.temp_files.append(repaired_path)
            with os.fdopen(fd, 'wb') as f:
                f.write(pdf_bytes)
            
            logger.info(f"Created repaired PDF at {repaired_path}")
            return True, repaired_path, pdf_bytes
            
        except Exception as e:
            logger.error(f"PDF repair failed: {str(e)}")
            return False, "", b""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text for better usability
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple newlines with a single one
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove form feed characters
        text = text.replace('\f', '\n\n')
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract text from a PDF URL with all fallbacks
        
        Args:
            url: URL of the PDF to extract
            
        Returns:
            Dict with extraction results and metadata
        """
        start_time = time.time()
        logger.info(f"Extracting PDF from URL: {url}")
        
        # Prepare result
        result = {
            "url": url,
            "success": False,
            "text": "",
            "error": None,
            "engine_used": None,
            "text_length": 0,
            "processing_time": 0,
            "local_path": None
        }
        
        # Skip if not a PDF URL
        if not self._is_pdf_url(url):
            result["error"] = "Not a PDF URL"
            return result
        
        # Step 1: Download the PDF
        download_success, pdf_path, pdf_bytes = self._download_pdf(url)
        if not download_success:
            result["error"] = "Failed to download PDF"
            return result
        
        result["local_path"] = pdf_path
        
        # Step 2: Extract text using multiple engines
        text = ""
        engine_used = None
        
        # Try each engine in order of reliability
        for engine in self.engines:
            try:
                if engine == "pymupdf" and PYMUPDF_AVAILABLE:
                    text = self._extract_with_pymupdf(pdf_path, pdf_bytes)
                elif engine == "pdfplumber" and PDFPLUMBER_AVAILABLE:
                    text = self._extract_with_pdfplumber(pdf_path, pdf_bytes)
                elif engine == "pdfminer" and PDFMINER_AVAILABLE:
                    text = self._extract_with_pdfminer(pdf_path, pdf_bytes)
                elif engine == "ocr" and OCR_AVAILABLE:
                    text = self._extract_with_ocr(pdf_path, pdf_bytes)
                
                # Check if we got usable text
                if text and len(text) > 100:  # Minimum usable text
                    engine_used = engine
                    break
            except Exception as e:
                logger.warning(f"Error with {engine}: {str(e)}")
        
        # Step 3: If all direct methods fail, try to repair the PDF first
        if not text:
            logger.warning("All direct extraction methods failed, attempting repair")
            repair_success, repaired_path, repaired_bytes = self._repair_pdf(pdf_path, pdf_bytes)
            
            if repair_success:
                # Try again with each engine on the repaired PDF
                for engine in self.engines:
                    try:
                        if engine == "pymupdf" and PYMUPDF_AVAILABLE:
                            text = self._extract_with_pymupdf(repaired_path, repaired_bytes)
                        elif engine == "pdfplumber" and PDFPLUMBER_AVAILABLE:
                            text = self._extract_with_pdfplumber(repaired_path, repaired_bytes)
                        elif engine == "pdfminer" and PDFMINER_AVAILABLE:
                            text = self._extract_with_pdfminer(repaired_path, repaired_bytes)
                        elif engine == "ocr" and OCR_AVAILABLE:
                            text = self._extract_with_ocr(repaired_path, repaired_bytes)
                        
                        # Check if we got usable text
                        if text and len(text) > 100:
                            engine_used = f"repaired-{engine}"
                            break
                    except Exception as e:
                        logger.warning(f"Error with {engine} after repair: {str(e)}")
        
        # Step 4: Final fallback - if PDF is corrupted beyond repair, at least extract any text fragments
        if not text:
            logger.warning("All extraction methods failed, extracting any text fragments")
            
            # Try to extract any readable text as a last resort
            extracted_fragments = []
            
            # Find text patterns in the binary content
            printable_chars = re.findall(b'[a-zA-Z0-9 \.,;:\-\'\"?!()\\n\\r]{4,}', pdf_bytes)
            fragments = [frag.decode('utf-8', errors='ignore') for frag in printable_chars]
            
            # Filter out very short fragments
            fragments = [frag for frag in fragments if len(frag) > 20]
            
            if fragments:
                text = "\n\n".join(fragments)
                engine_used = "binary-fragments"
        
        # Update result
        result["success"] = bool(text)
        result["text"] = text or "PDF extraction failed with all methods"
        result["engine_used"] = engine_used
        result["text_length"] = len(text)
        result["processing_time"] = time.time() - start_time
        
        if not text:
            result["error"] = "Failed to extract text with all available methods"
        
        logger.info(f"PDF extraction completed: success={result['success']}, engine={engine_used}, length={len(text)}")
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


# Example usage
if __name__ == "__main__":
    extractor = AdvancedPDFExtractor()
    
    # Example PDF URL - using a reliable sample PDF
    pdf_url = "https://www.learningcontainer.com/wp-content/uploads/2019/09/sample-pdf-file.pdf"
    
    try:
        print(f"Extracting text from: {pdf_url}")
        result = extractor.extract_from_url(pdf_url)
        
        if result["success"]:
            print(f"Extraction successful using {result['engine_used']}")
            print(f"Text length: {result['text_length']} characters")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            
            # Show a preview of the text
            text_preview = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
            print(f"\nText preview:\n{text_preview}")
        else:
            print(f"Extraction failed: {result['error']}")
    
    except Exception as e:
        print(f"Error in example: {str(e)}")
    
    finally:
        extractor.cleanup()
