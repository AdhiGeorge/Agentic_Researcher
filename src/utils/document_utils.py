"""
Document Utilities for Agentic Researcher

This module contains consolidated document handling utilities including:
- File processing functions for various file types (txt, docx, pdf, etc.)
- PDF parsing and extraction capabilities with metadata handling
- General file system operations for the research workflow
"""

import os
import re
import json
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, BinaryIO, Union, Any

# Enhanced PDF handling
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available; PDF handling will be limited")

# Document processing
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx not available; DOCX handling will be limited")

# Configure logging
logger = logging.getLogger(__name__)

#============================================================================
# File Detection and Type Handling
#============================================================================

def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        str: Lowercase file extension without the dot
    """
    return os.path.splitext(filename)[1].lower().lstrip('.')

def get_mime_type(file_path: str) -> str:
    """
    Get the MIME type of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: MIME type of the file
    """
    import mimetypes
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"

def is_text_file(file_path: str) -> bool:
    """
    Check if a file is a text file based on its MIME type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is a text file
    """
    mime_type = get_mime_type(file_path)
    return mime_type and mime_type.startswith('text/')

def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is a binary file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is binary
    """
    return not is_text_file(file_path)

def is_pdf_file(file_path: str) -> bool:
    """
    Check if a file is a PDF file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is a PDF
    """
    return get_file_extension(file_path) == 'pdf'

def get_supported_document_extensions() -> List[str]:
    """
    Get a list of supported document extensions.
    
    Returns:
        List[str]: List of supported document extensions
    """
    extensions = ['txt', 'md', 'html', 'htm', 'json', 'csv', 'tsv']
    
    # Add PDF if PyMuPDF is available
    if PYMUPDF_AVAILABLE:
        extensions.append('pdf')
    
    # Add DOCX if python-docx is available
    if DOCX_AVAILABLE:
        extensions.append('docx')
    
    return extensions

#============================================================================
# File Reading and Writing
#============================================================================

def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Read a text file and return its contents.
    
    Args:
        file_path: Path to the text file
        encoding: Text encoding to use
        
    Returns:
        str: Contents of the text file
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            return ""

def write_text_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    Write content to a text file.
    
    Args:
        file_path: Path to the text file
        content: Content to write
        encoding: Text encoding to use
        
    Returns:
        bool: True if successful
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing text file {file_path}: {str(e)}")
        return False

def append_to_text_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    Append content to a text file.
    
    Args:
        file_path: Path to the text file
        content: Content to append
        encoding: Text encoding to use
        
    Returns:
        bool: True if successful
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'a', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error appending to text file {file_path}: {str(e)}")
        return False

def read_json_file(file_path: str) -> Dict:
    """
    Read a JSON file and return its contents.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict: Contents of the JSON file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {str(e)}")
        return {}

def write_json_file(file_path: str, data: Dict, indent: int = 2) -> bool:
    """
    Write data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to write
        indent: Indentation level
        
    Returns:
        bool: True if successful
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {str(e)}")
        return False

def save_binary_file(file_path: str, data: bytes) -> bool:
    """
    Save binary data to a file.
    
    Args:
        file_path: Path to the file
        data: Binary data to save
        
    Returns:
        bool: True if successful
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        logger.error(f"Error saving binary file {file_path}: {str(e)}")
        return False

def read_binary_file(file_path: str) -> bytes:
    """
    Read binary data from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bytes: Binary data from the file
    """
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading binary file {file_path}: {str(e)}")
        return b''

#============================================================================
# Document Content Extraction
#============================================================================

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a file based on its type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Extracted text
    """
    # Get the file extension
    extension = get_file_extension(file_path)
    
    # Process based on extension
    if extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif extension == 'docx':
        return extract_text_from_docx(file_path)
    elif extension == 'txt' or is_text_file(file_path):
        return read_text_file(file_path)
    elif extension == 'json':
        data = read_json_file(file_path)
        return json.dumps(data, indent=2)
    else:
        logger.warning(f"Unsupported file type: {extension} for file {file_path}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        str: Extracted text
    """
    if not DOCX_AVAILABLE:
        logger.error("python-docx not available; can't extract text from DOCX")
        return ""
    
    try:
        doc = docx.Document(file_path)
        text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available; can't extract text from PDF")
        return ""
    
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        return ""

def extract_pages_from_pdf(file_path: str, start_page: int = 0, end_page: Optional[int] = None) -> str:
    """
    Extract text from specific pages of a PDF file.
    
    Args:
        file_path: Path to the PDF file
        start_page: Starting page (0-indexed)
        end_page: Ending page (inclusive, None for all pages)
        
    Returns:
        str: Extracted text
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available; can't extract pages from PDF")
        return ""
    
    try:
        text = ""
        with fitz.open(file_path) as doc:
            max_page = len(doc) - 1
            end_page = min(end_page, max_page) if end_page is not None else max_page
            start_page = max(0, min(start_page, max_page))
            
            for page_num in range(start_page, end_page + 1):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting pages from PDF {file_path}: {str(e)}")
        return ""

def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dict[str, Any]: Metadata dictionary
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available; can't get PDF metadata")
        return {}
    
    try:
        with fitz.open(file_path) as doc:
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "page_count": len(doc),
                "file_size": os.path.getsize(file_path)
            }
            return metadata
    except Exception as e:
        logger.error(f"Error getting PDF metadata for {file_path}: {str(e)}")
        return {}

def extract_pdf_table_of_contents(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract the table of contents (outline) from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List[Dict[str, Any]]: Table of contents entries
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available; can't extract PDF table of contents")
        return []
    
    try:
        toc = []
        with fitz.open(file_path) as doc:
            for item in doc.get_toc():
                toc.append({
                    "level": item[0],
                    "title": item[1],
                    "page": item[2]
                })
        return toc
    except Exception as e:
        logger.error(f"Error extracting PDF table of contents for {file_path}: {str(e)}")
        return []

def extract_pdf_images(file_path: str, output_dir: str) -> List[str]:
    """
    Extract images from a PDF file and save them to a directory.
    
    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save images to
        
    Returns:
        List[str]: Paths to the extracted images
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available; can't extract PDF images")
        return []
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image_ext = base_image["ext"]
                    image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_paths.append(image_path)
        
        return image_paths
    except Exception as e:
        logger.error(f"Error extracting PDF images from {file_path}: {str(e)}")
        return []

#============================================================================
# Document Processing for Research
#============================================================================

def create_file_hash(file_path: str) -> str:
    """
    Create a hash of a file's contents for identification.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: MD5 hash of the file
    """
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read in 1MB chunks to handle large files
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error creating file hash for {file_path}: {str(e)}")
        return ""

def get_file_stats(file_path: str) -> Dict[str, Any]:
    """
    Get statistics about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict[str, Any]: File statistics
    """
    try:
        stats = os.stat(file_path)
        return {
            "size_bytes": stats.st_size,
            "size_kb": stats.st_size / 1024,
            "size_mb": stats.st_size / (1024 * 1024),
            "created": stats.st_ctime,
            "modified": stats.st_mtime,
            "accessed": stats.st_atime,
        }
    except Exception as e:
        logger.error(f"Error getting file stats for {file_path}: {str(e)}")
        return {}

def process_document_for_research(file_path: str) -> Dict[str, Any]:
    """
    Process a document for research, extracting text, metadata, and other information.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Dict[str, Any]: Processed document information
    """
    result = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_extension": get_file_extension(file_path),
        "file_stats": get_file_stats(file_path),
        "file_hash": create_file_hash(file_path)
    }
    
    # Extract text based on file type
    result["text"] = extract_text_from_file(file_path)
    result["text_length"] = len(result["text"])
    
    # Get PDF-specific metadata if applicable
    if is_pdf_file(file_path) and PYMUPDF_AVAILABLE:
        result["metadata"] = get_pdf_metadata(file_path)
        result["table_of_contents"] = extract_pdf_table_of_contents(file_path)
    
    return result

def count_words_in_document(file_path: str) -> int:
    """
    Count the number of words in a document.
    
    Args:
        file_path: Path to the document
        
    Returns:
        int: Word count
    """
    text = extract_text_from_file(file_path)
    words = re.findall(r'\w+', text)
    return len(words)

def scan_directory_for_documents(directory: str, recursive: bool = True) -> List[Dict[str, Any]]:
    """
    Scan a directory for documents and return their information.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        List[Dict[str, Any]]: List of document information dictionaries
    """
    supported_extensions = get_supported_document_extensions()
    documents = []
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_extension = get_file_extension(file)
                if file_extension in supported_extensions:
                    file_path = os.path.join(root, file)
                    doc_info = {
                        "file_path": file_path,
                        "file_name": file,
                        "file_extension": file_extension,
                        "file_size": os.path.getsize(file_path),
                        "relative_path": os.path.relpath(file_path, directory)
                    }
                    documents.append(doc_info)
            
            if not recursive:
                break
    
    except Exception as e:
        logger.error(f"Error scanning directory {directory}: {str(e)}")
    
    return documents

def merge_pdf_files(input_files: List[str], output_file: str) -> bool:
    """
    Merge multiple PDF files into one.
    
    Args:
        input_files: List of input PDF file paths
        output_file: Output PDF file path
        
    Returns:
        bool: True if successful
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available; can't merge PDF files")
        return False
    
    try:
        result = fitz.open()
        
        for input_file in input_files:
            with fitz.open(input_file) as pdf:
                result.insert_pdf(pdf)
        
        result.save(output_file)
        result.close()
        return True
    except Exception as e:
        logger.error(f"Error merging PDF files: {str(e)}")
        return False

# Example usage when run directly
if __name__ == "__main__":
    import sys
    import urllib.request
    from dotenv import load_dotenv
    
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n===== Document Utilities - Real-World Examples =====\n")
    
    # Set up a working directory for our examples
    working_dir = os.path.join(tempfile.gettempdir(), "agentic_researcher_examples")
    os.makedirs(working_dir, exist_ok=True)
    print(f"Working directory: {working_dir}\n")
    
    # Example 1: List supported document types
    print("EXAMPLE 1: SUPPORTED DOCUMENT TYPES")
    print("-" * 50)
    
    supported_extensions = get_supported_document_extensions()
    print(f"The following document formats are supported for processing:")
    print(f"  {', '.join(supported_extensions)}")
    print()
    
    for ext in supported_extensions:
        if ext == "pdf" and not PYMUPDF_AVAILABLE:
            print(f"  {ext}: Limited support (PyMuPDF not available)")
        elif ext == "docx" and not DOCX_AVAILABLE:
            print(f"  {ext}: Limited support (python-docx not available)")
        else:
            print(f"  {ext}: Full support")
            
    # Example 2: Working with a real research paper
    print("\nEXAMPLE 2: DOWNLOADING AND PROCESSING A REAL RESEARCH PAPER")
    print("-" * 50)
    
    print("Downloading a quantum computing research paper...")
    
    # Define a real PDF to download (IBM Quantum Computing paper)
    pdf_url = "https://arxiv.org/pdf/1801.00862.pdf"
    pdf_filename = "quantum_computing_research.pdf"
    pdf_path = os.path.join(working_dir, pdf_filename)
    
    try:
        # Download the PDF file
        if not os.path.exists(pdf_path):
            urllib.request.urlretrieve(pdf_url, pdf_path)
            print(f"Downloaded research paper to {pdf_path}")
        else:
            print(f"Using existing research paper at {pdf_path}")
        
        # Get file information
        file_stats = get_file_stats(pdf_path)
        file_hash = create_file_hash(pdf_path)
        
        print(f"\nFile information:")
        print(f"  Size: {file_stats['size_mb']:.2f} MB")
        print(f"  Created: {datetime.fromtimestamp(file_stats['created']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  MD5 Hash: {file_hash}")
        
        # Extract and display PDF metadata if PyMuPDF is available
        if PYMUPDF_AVAILABLE:
            # Get metadata
            metadata = get_pdf_metadata(pdf_path)
            toc = extract_pdf_table_of_contents(pdf_path)
            
            print(f"\nPDF Metadata:")
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  Author(s): {metadata.get('author', 'N/A')}")
            print(f"  Pages: {metadata.get('page_count', 0)}")
            
            if toc:
                print(f"\nTable of Contents (First 5 entries):")
                for i, entry in enumerate(toc[:5], 1):
                    print(f"  {i}. {entry['title']} (Page {entry['page']})")
                if len(toc) > 5:
                    print(f"  ... and {len(toc) - 5} more entries")
            
            # Extract specific pages
            print(f"\nExtracting content from the first page...")
            first_page = extract_pages_from_pdf(pdf_path, 0, 0)
            
            # Display a preview of the first page
            preview_length = 300
            first_page_preview = first_page.replace('\n', ' ')
            first_page_preview = re.sub(r'\s+', ' ', first_page_preview).strip()
            
            print(f"\nFirst page preview:")
            print(f"  \"{first_page_preview[:preview_length]}...\"")
            print(f"  Content length: {len(first_page)} characters")
            
            # Extract images if any (limit to first 3 pages for the example)
            print(f"\nExtracting images from the first 3 pages...")
            images_dir = os.path.join(working_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            try:
                with fitz.open(pdf_path) as doc:
                    # Only process first 3 pages for the example
                    max_pages = min(3, len(doc))
                    images_found = 0
                    
                    for page_num in range(max_pages):
                        images_on_page = 0
                        page = doc[page_num]
                        
                        for img_index, img in enumerate(page.get_images(full=True)):
                            images_found += 1
                            images_on_page += 1
                            xref = img[0]
                            
                            # Only extract first 2 images per page for the example
                            if images_on_page <= 2:
                                try:
                                    base_image = doc.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    image_ext = base_image["ext"]
                                    image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                                    image_path = os.path.join(images_dir, image_filename)
                                    
                                    with open(image_path, "wb") as img_file:
                                        img_file.write(image_bytes)
                                except Exception as e:
                                    print(f"    Error extracting image: {e}")
                    
                    print(f"  Found {images_found} images in the first {max_pages} pages")
                    print(f"  Images saved to {images_dir}")
            except Exception as e:
                print(f"Error processing images: {e}")
                
        else:
            print("\nPyMuPDF is not available - install it with 'pip install pymupdf' for full PDF processing")
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
    
    # Example 3: Creating a research document corpus
    print("\nEXAMPLE 3: CREATING A RESEARCH DOCUMENT CORPUS")
    print("-" * 50)
    
    corpus_dir = os.path.join(working_dir, "research_corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    
    print(f"Created research corpus directory: {corpus_dir}")
    
    # Create a few sample documents
    docs = [
        {
            "filename": "quantum_intro.txt",
            "content": """Quantum Computing: An Introduction
            
            Quantum computing is an emerging field that uses quantum-mechanical phenomena such as superposition and entanglement to perform computations. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously.
            
            Key concepts in quantum computing include:
            1. Qubits - The basic unit of quantum information
            2. Superposition - The ability of qubits to exist in multiple states simultaneously
            3. Entanglement - A quantum phenomenon where qubits become correlated
            4. Quantum gates - Operations that manipulate qubits
            
            Quantum computers have the potential to solve certain problems exponentially faster than classical computers."""
        },
        {
            "filename": "quantum_applications.txt",
            "content": """Applications of Quantum Computing
            
            Quantum computers show promise in several key areas:
            
            1. Cryptography: Shor's algorithm can efficiently factor large numbers, potentially breaking current encryption methods.
            
            2. Drug Discovery: Simulating molecular interactions at the quantum level could accelerate pharmaceutical research.
            
            3. Optimization Problems: Quantum algorithms may find optimal solutions for complex logistical challenges like supply chain management.
            
            4. Machine Learning: Quantum machine learning algorithms may process certain data structures exponentially faster.
            
            5. Materials Science: Quantum simulations could help design new materials with specific properties."""
        },
        {
            "filename": "research_notes.json",
            "content": json.dumps({
                "project": "Quantum Computing Research",
                "researcher": "Dr. Quantum Researcher",
                "date": datetime.now().isoformat(),
                "topics": [
                    "quantum gates",
                    "error correction",
                    "quantum supremacy"
                ],
                "references": [
                    {
                        "title": "Quantum Computing for Computer Scientists",
                        "authors": "Yanofsky, N. S., Mannucci, M. A.",
                        "year": 2008
                    },
                    {
                        "title": "Quantum Computation and Quantum Information",
                        "authors": "Nielsen, M. A., Chuang, I. L.",
                        "year": 2010
                    }
                ]
            }, indent=4)
        }
    ]
    
    # Create the sample documents
    for doc in docs:
        doc_path = os.path.join(corpus_dir, doc["filename"])
        if doc["filename"].endswith(".json"):
            write_json_file(doc_path, json.loads(doc["content"]))
        else:
            write_text_file(doc_path, doc["content"])
    
    print(f"Created {len(docs)} research documents in the corpus")
    
    # Scan and process the corpus
    print("\nScanning and processing the research corpus...")
    corpus_files = scan_directory_for_documents(corpus_dir)
    
    print(f"Found {len(corpus_files)} documents in the corpus:")
    for i, doc_info in enumerate(corpus_files, 1):
        print(f"  {i}. {doc_info['file_name']} ({doc_info['file_extension']}) - {doc_info['file_size']/1024:.1f} KB")
    
    # Process the documents in the corpus
    print("\nProcessing document content:")
    for doc_info in corpus_files:
        file_path = doc_info["file_path"]
        
        # Extract text based on file type
        text = extract_text_from_file(file_path)
        word_count = count_words_in_document(file_path)
        
        print(f"\n  Document: {os.path.basename(file_path)}")
        print(f"  Word Count: {word_count}")
        print(f"  Content Preview: \"{text[:100].replace('\n', ' ')}...\"")
    
    # Example 4: Document Processing for Research Pipeline
    print("\nEXAMPLE 4: DOCUMENT PROCESSING FOR RESEARCH PIPELINE")
    print("-" * 50)
    
    print("Simulating a complete document processing pipeline for research")
    print("1. Document intake and identification")
    print("2. Text extraction and cleaning")
    print("3. Metadata enrichment")
    print("4. Storage preparation")
    
    # Select a document to process
    sample_doc_path = os.path.join(corpus_dir, "quantum_applications.txt")
    
    print(f"\nProcessing document: {os.path.basename(sample_doc_path)}")
    
    # Step 1: Document information and identification
    doc_hash = create_file_hash(sample_doc_path)
    doc_stats = get_file_stats(sample_doc_path)
    
    print("\nDocument Identification:")
    print(f"  File: {os.path.basename(sample_doc_path)}")
    print(f"  Hash: {doc_hash[:10]}...{doc_hash[-10:]}")
    print(f"  Size: {doc_stats['size_kb']:.2f} KB")
    print(f"  Modified: {datetime.fromtimestamp(doc_stats['modified']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 2: Text extraction
    doc_text = extract_text_from_file(sample_doc_path)
    cleaned_text = re.sub(r'\s+', ' ', doc_text).strip()
    
    print("\nContent Analysis:")
    print(f"  Raw Length: {len(doc_text)} characters")
    print(f"  Cleaned Length: {len(cleaned_text)} characters")
    print(f"  Word Count: {len(cleaned_text.split())} words")
    
    # Step 3: Metadata enrichment
    # Extract entities (simplified version for the example)
    entities = []
    keywords = ["Shor's algorithm", "cryptography", "drug discovery", "optimization", 
               "machine learning", "materials science"]
    
    for keyword in keywords:
        if keyword.lower() in cleaned_text.lower():
            entities.append({
                "text": keyword,
                "type": "CONCEPT",
                "positions": [cleaned_text.lower().find(keyword.lower())]
            })
    
    print("\nExtracted Entities:")
    for entity in entities:
        print(f"  {entity['text']} (Type: {entity['type']})")
    
    # Step 4: Create processed document record
    processed_doc = {
        "id": doc_hash,
        "filename": os.path.basename(sample_doc_path),
        "path": sample_doc_path,
        "stats": doc_stats,
        "content": {
            "raw": doc_text,
            "cleaned": cleaned_text,
            "word_count": len(cleaned_text.split())
        },
        "metadata": {
            "entities": entities,
            "keywords": keywords,
            "processed_at": datetime.now().isoformat()
        }
    }
    
    # Export the processed document
    processed_doc_path = os.path.join(working_dir, "processed_document.json")
    write_json_file(processed_doc_path, processed_doc)
    
    print(f"\nProcessed document saved to: {processed_doc_path}")
    
    print("\n" + "=" * 80 + "\n")
    print("All document processing examples completed successfully!")
    print("These utilities are essential for handling various document types")
    print("in research pipelines and knowledge management systems.")
    print("=" * 80)
    
    # Note about cleanup
    print(f"\nNOTE: Example files remain in {working_dir} for your inspection.")
    print("You can delete this directory when you're done exploring the examples.")
    print(f"To remove: 'rm -rf {working_dir}'")

