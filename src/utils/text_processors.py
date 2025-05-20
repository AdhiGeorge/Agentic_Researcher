"""
Text Processing Utilities for Agentic Researcher

This module contains consolidated text processing utilities including:
- TextSplitter: Splits text into manageable chunks based on characters
- SemanticChunker: Creates semantically meaningful chunks with entity awareness
- Tokenizer: Handles tokenization and token counting for LLM context management
"""

import re
import uuid
import hashlib
import tiktoken
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

# Try to import NLP libraries for enhanced functionality
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available for advanced sentence tokenization")

# Configure logging
logger = logging.getLogger(__name__)

#============================================================================
# TextSplitter Class
#============================================================================

class TextSplitter:
    """TextSplitter for chunking text into manageable pieces
    
    Used for:
    1. Preparing text for embedding and vector storage
    2. Breaking large documents into smaller chunks for processing
    3. Maintaining context by controlling chunk size and overlap
    """
    
    def __init__(self, default_chunk_size: int = 1000, default_overlap: int = 200):
        """
        Initialize the TextSplitter
        
        Args:
            default_chunk_size: Default size for text chunks in characters
            default_overlap: Default overlap between chunks in characters
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
    
    def split_text(self, text: str, chunk_size: Optional[int] = None, 
                   chunk_overlap: Optional[int] = None) -> List[str]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List[str]: List of text chunks
        """
        # Use default values if not provided
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_overlap
        
        # Clean the text
        text = self._clean_text(text)
        
        # If text is shorter than chunk size, return it as a single chunk
        if len(text) <= chunk_size:
            return [text]
        
        # Split text into chunks
        chunks = []
        start = 0
        
        while start < len(text):
            # Get the chunk
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to find a good split point - prefer sentence boundaries
            if end < len(text):
                # Look for the last sentence boundary in the chunk
                last_period = chunk.rfind('. ')
                last_newline = chunk.rfind('\n')
                
                # Find the best split point (prefer sentence boundary, then paragraph)
                split_point = max(last_period + 2, last_newline + 1)
                
                # If no good split point found, just use the chunk size
                if split_point < 0.5 * chunk_size:
                    split_point = chunk_size
                
                # Adjust the chunk
                chunk = text[start:start + split_point]
                end = start + split_point
            
            # Add the chunk to the list
            chunks.append(chunk)
            
            # Move the start position for the next chunk, considering overlap
            start = end - chunk_overlap
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace, etc.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text

#============================================================================
# SemanticChunker Class
#============================================================================

class SemanticChunker:
    """
    Semantic Chunker for creating optimal text chunks with semantic awareness.
    
    This class provides:
    1. Text chunking with controlled size and overlap
    2. Sentence-aware boundary handling
    3. Entity preservation across chunks
    4. Deduplication
    5. Metadata enhancement
    
    Attributes:
        logger (logging.Logger): Logger for the chunker
        chunk_size (int): Target size for text chunks in characters
        chunk_overlap (int): Overlap between chunks in characters
        min_chunk_size (int): Minimum size for a valid chunk
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """Initialize the SemanticChunker.
        
        Args:
            chunk_size (int, optional): Target size for chunks in characters. Defaults to 1000.
            chunk_overlap (int, optional): Overlap between chunks in characters. Defaults to 200.
            min_chunk_size (int, optional): Minimum size for a valid chunk. Defaults to 100.
        """
        self.logger = logging.getLogger("utils.semantic_chunker")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self.logger.info(f"SemanticChunker initialized with size={chunk_size}, overlap={chunk_overlap}")
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """Create semantic chunks from text.
        
        Args:
            text (str): Text to chunk
            metadata (Dict, optional): Metadata to attach to chunks. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        if not text or len(text) < self.min_chunk_size:
            self.logger.warning(f"Text too short to chunk: {len(text) if text else 0} chars")
            return []
        
        metadata = metadata or {}
        
        # Generate content ID if not provided
        content_id = metadata.get("content_id", hashlib.md5(text[:1000].encode()).hexdigest())
        
        # Tokenize text into sentences
        sentences = self._tokenize_sentences(text)
        
        if not sentences:
            self.logger.warning("No sentences found in text")
            return []
        
        # Create chunks
        chunks = self._create_chunks_from_sentences(sentences, content_id, metadata)
        
        # Deduplicate
        unique_chunks = self._deduplicate_chunks(chunks)
        
        self.logger.info(f"Created {len(unique_chunks)} chunks from text of length {len(text)}")
        return unique_chunks
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of sentences
        """
        # Clean text
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        
        # Use NLTK if available
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        else:
            # Simple regex-based sentence tokenization
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunks_from_sentences(
        self, 
        sentences: List[str], 
        content_id: str,
        metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Create chunks from sentences respecting semantic boundaries.
        
        Args:
            sentences (List[str]): List of sentences
            content_id (str): Content identifier
            metadata (Dict): Metadata to attach to chunks
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size and we already have content,
            # then finalize the current chunk and start a new one
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk with metadata
                chunk_text = " ".join(current_chunk)
                chunk_id = f"{content_id}-{len(chunks)}"
                
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "content_id": content_id,
                    "chunk_index": len(chunks),
                    "chunk_size": len(chunk_text),
                    "created_at": datetime.now().isoformat()
                })
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                
                # Start new chunk with overlap
                # Take the last few sentences to maintain context
                overlap_size = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s) + 1  # +1 for space
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        
        # Don't forget the last chunk if it's long enough
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{content_id}-{len(chunks)}"
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "content_id": content_id,
                "chunk_index": len(chunks),
                "chunk_size": len(chunk_text),
                "created_at": datetime.now().isoformat()
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate chunks based on content hash.
        
        Args:
            chunks (List[Dict[str, Any]]): Chunks to deduplicate
            
        Returns:
            List[Dict[str, Any]]: Deduplicated chunks
        """
        seen_hashes = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Create a hash of the chunk's text
            chunk_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
            
            # Only include this chunk if we haven't seen its hash before
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                
                # Add deduplication info to metadata
                chunk["metadata"]["hash"] = chunk_hash
                
                # Add to unique chunks
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def chunk_with_entities(self, text: str, entities: List[Dict], metadata: Dict = None) -> List[Dict[str, Any]]:
        """Create chunks with awareness of named entities.
        
        This method tries to keep entity mentions within the same chunk when possible.
        
        Args:
            text (str): Text to chunk
            entities (List[Dict]): List of entities to preserve in chunks
            metadata (Dict, optional): Metadata to attach to chunks. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        # Start with basic chunking
        chunks = self.create_chunks(text, metadata)
        
        # If no entities, return regular chunks
        if not entities:
            return chunks
        
        # Extract entity mentions to preserve
        entity_mentions = {}
        for entity in entities:
            entity_text = entity.get("text", "").lower()
            if entity_text:
                entity_mentions[entity_text] = entity.get("type", "ENTITY")
        
        # Enhance chunks with entity information
        for chunk in chunks:
            chunk_text = chunk.get("text", "").lower()
            found_entities = []
            
            for entity_text, entity_type in entity_mentions.items():
                if entity_text in chunk_text:
                    found_entities.append({"text": entity_text, "type": entity_type})
            
            # Add entity information to metadata
            if found_entities:
                if "entities" not in chunk["metadata"]:
                    chunk["metadata"]["entities"] = []
                chunk["metadata"]["entities"].extend(found_entities)
        
        return chunks

#============================================================================
# Tokenizer Class
#============================================================================

class Tokenizer:
    """
    Tokenizer utility for text tokenization and token counting.
    Uses tiktoken for OpenAI GPT tokenization.
    """
    # Singleton instance
    _instance = None

    # Supported models and their respective encodings
    MODELS_ENCODINGS = {
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-35-turbo": "cl100k_base",  # Azure naming
        "gpt-4": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base"
    }

    # Default fallback encoding
    DEFAULT_ENCODING = "cl100k_base"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tokenizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Load encodings
        self._encodings = {}
        for _, encoding_name in self.MODELS_ENCODINGS.items():
            if encoding_name not in self._encodings:
                self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)
        
        # Default encoding
        self._default_encoding = self._encodings[self.DEFAULT_ENCODING]
        self._initialized = True

    def get_encoding_for_model(self, model_name: str):
        """
        Get the appropriate tiktoken encoding for a model
        
        Args:
            model_name: Name of the model to get encoding for
            
        Returns:
            tiktoken.Encoding: The encoding for the specified model
        """
        encoding_name = self.MODELS_ENCODINGS.get(model_name, self.DEFAULT_ENCODING)
        return self._encodings[encoding_name]

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """
        Count the number of tokens in a text string
        
        Args:
            text: Text to count tokens for
            model_name: Optional model name to determine encoding
            
        Returns:
            int: Number of tokens
        """
        if text is None or text == "":
            return 0
            
        if model_name:
            encoding = self.get_encoding_for_model(model_name)
        else:
            encoding = self._default_encoding
            
        return len(encoding.encode(text))

    def split_text_into_chunks(
        self, 
        text: str, 
        chunk_size: int = 400, 
        chunk_overlap: int = 50,
        model_name: Optional[str] = None
    ) -> List[str]:
        """
        Split text into chunks of specified token size with overlap
        Tries to split at sentence boundaries for better context
        
        Args:
            text: Text to split
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            model_name: Optional model name to determine encoding
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
            
        if model_name:
            encoding = self.get_encoding_for_model(model_name)
        else:
            encoding = self._default_encoding
            
        # Get all tokens
        tokens = encoding.encode(text)
        
        # Initialize chunks
        chunks = []
        i = 0
        
        while i < len(tokens):
            # Get chunk end position
            end = min(i + chunk_size, len(tokens))
            
            # Try to find a good split point if this isn't the last chunk
            if end < len(tokens):
                # Decode the current chunk
                chunk_text = encoding.decode(tokens[i:end])
                
                # Try to find sentence boundaries
                sentence_pattern = r'[.!?]\s+'
                matches = list(re.finditer(sentence_pattern, chunk_text))
                
                if matches:
                    # Get the last sentence boundary
                    last_match = matches[-1]
                    end_pos = last_match.end()
                    adjusted_text = chunk_text[:end_pos]
                    adjusted_end = i + self.count_tokens(adjusted_text, model_name)
                    
                    # Only use the sentence boundary if it's not too early in the chunk
                    if adjusted_end > i + (chunk_size * 0.7):  # At least 70% of max chunk size
                        end = adjusted_end
                    # Otherwise use the full chunk_size
            
            # Add the chunk
            chunk_text = encoding.decode(tokens[i:end])
            chunks.append(chunk_text)
            
            # Move to next chunk, with overlap
            i = end - chunk_overlap
            if i >= len(tokens):
                break
        
        return chunks

# Example usage when run directly
if __name__ == "__main__":
    import os
    import sys
    import time
    from dotenv import load_dotenv
    
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("===== Text Processing Utilities Examples =====\n")
    
    # Example text for processing - using a real research article excerpt
    sample_text = """
    Quantum computing is an emerging field that promises to revolutionize computation by utilizing quantum 
    mechanical phenomena. Unlike classical computers that use bits (0 or 1), quantum computers use quantum 
    bits or qubits, which can exist in superposition states. This fundamental difference enables quantum 
    computers to perform certain calculations exponentially faster than classical computers.
    
    Recent advancements in quantum computing include:
    
    1. Error Correction: Researchers have developed better quantum error correction codes, essential for 
    creating fault-tolerant quantum computers. Google's Quantum AI team demonstrated a distance-5 surface 
    code that significantly reduces logical error rates.
    
    2. Quantum Supremacy: In 2019, Google claimed to achieve quantum supremacy with its 53-qubit Sycamore 
    processor, performing a specific calculation in 200 seconds that would take a classical supercomputer 
    approximately 10,000 years.
    
    3. Quantum Machine Learning: Quantum neural networks and variational quantum algorithms are being 
    developed to enhance machine learning capabilities, potentially offering speedups for training complex models.
    
    4. Quantum Internet: Researchers are working on quantum networks that allow for secure communication 
    through quantum key distribution, with China's quantum satellite Micius demonstrating intercontinental 
    quantum-secured communication.
    
    Despite these advances, significant challenges remain in scaling quantum systems, reducing error rates, 
    and developing practical quantum algorithms for real-world problems.
    """
    
    print("Using a real research text about quantum computing advancements\n")
    print("-" * 80)
    
    # EXAMPLE 1: Basic TextSplitter with real text
    print("\n1. BASIC TEXT SPLITTING")
    print("-" * 50)
    
    # Create TextSplitter with specific chunk parameters
    text_splitter = TextSplitter(default_chunk_size=300, default_overlap=50)
    basic_chunks = text_splitter.split_text(sample_text)
    
    print(f"Split into {len(basic_chunks)} chunks based on character count")
    print(f"Total text length: {len(sample_text)} characters")
    
    for i, chunk in enumerate(basic_chunks, 1):
        print(f"\nChunk {i}: {len(chunk)} characters, starts with:")
        print(f"'{chunk[:100]}...'")
    
    print("\nNote how the splitting tries to respect sentence boundaries")
    
    # EXAMPLE 2: SemanticChunker with real text
    print("\n\n2. SEMANTIC CHUNKING WITH METADATA")
    print("-" * 50)
    
    # Initialize semantic chunker with specific parameters
    semantic_chunker = SemanticChunker(chunk_size=400, chunk_overlap=100)
    
    # Add custom metadata to track the source
    metadata = {
        "source": "quantum_computing_review",
        "date": datetime.now().isoformat(),
        "category": "technology",
        "author": "Quantum Research Team"
    }
    
    # Create semantic chunks with metadata
    semantic_chunks = semantic_chunker.create_chunks(sample_text, metadata)
    
    print(f"Created {len(semantic_chunks)} semantic chunks with metadata")
    
    # Show each chunk with its metadata
    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"\nChunk {i}: {len(chunk['text'])} characters")
        print(f"Preview: '{chunk['text'][:100]}...'")
        print(f"Metadata fields: {list(chunk['metadata'].keys())}")
        print(f"Content ID: {chunk['metadata']['content_id']}")
        print(f"Chunk ID: {chunk['metadata']['chunk_id']}")
    
    # EXAMPLE 3: Entity-aware chunking with real entities
    print("\n\n3. ENTITY-AWARE CHUNKING")
    print("-" * 50)
    
    # Define actual entities from the text
    entities = [
        {"text": "quantum computing", "type": "TECHNOLOGY"},
        {"text": "qubits", "type": "QUANTUM_CONCEPT"},
        {"text": "Google", "type": "COMPANY"},
        {"text": "Sycamore", "type": "QUANTUM_PROCESSOR"},
        {"text": "quantum supremacy", "type": "MILESTONE"},
        {"text": "Micius", "type": "SATELLITE"}
    ]
    
    # Create chunks with entity awareness
    entity_chunks = semantic_chunker.chunk_with_entities(
        sample_text, 
        entities, 
        {"source": "entity_aware_example", "domain": "quantum_computing"}
    )
    
    print(f"Created {len(entity_chunks)} entity-aware chunks")
    
    # Show entities found in each chunk
    for i, chunk in enumerate(entity_chunks, 1):
        print(f"\nChunk {i}: {len(chunk['text'])} characters")
        if "entities" in chunk["metadata"]:
            found_entities = [e['text'] for e in chunk['metadata']['entities']]
            print(f"Entities found: {found_entities}")
            print(f"Entity count: {len(found_entities)}")
        else:
            print("No entities found in this chunk")
    
    # EXAMPLE 4: Tokenizer with real text and accurate token counts
    print("\n\n4. TOKENIZATION FOR LLM CONTEXT MANAGEMENT")
    print("-" * 50)
    
    # Initialize the tokenizer
    tokenizer = Tokenizer()
    
    # Count tokens using different models
    gpt35_tokens = tokenizer.count_tokens(sample_text, model_name="gpt-35-turbo")
    gpt4_tokens = tokenizer.count_tokens(sample_text, model_name="gpt-4")
    embedding_tokens = tokenizer.count_tokens(sample_text, model_name="text-embedding-3-small")
    
    print(f"Quantum computing text statistics:")
    print(f"Character count: {len(sample_text)}")
    print(f"GPT-3.5 Turbo tokens: {gpt35_tokens}")
    print(f"GPT-4 tokens: {gpt4_tokens}")
    print(f"Text embedding tokens: {embedding_tokens}")
    
    # Split text into token-sized chunks for context window management
    token_chunks = tokenizer.split_text_into_chunks(
        sample_text, 
        chunk_size=150,  # Small size for demonstration
        chunk_overlap=20,
        model_name="gpt-4"
    )
    
    print(f"\nSplit into {len(token_chunks)} token-sized chunks for GPT-4:")
    
    # Display token counts for each chunk
    for i, chunk in enumerate(token_chunks, 1):
        chunk_tokens = tokenizer.count_tokens(chunk, model_name="gpt-4")
        print(f"Chunk {i}: {chunk_tokens} tokens")
    
    # EXAMPLE 5: Practical application - Processing a long document for a chatbot
    print("\n\n5. PRACTICAL APPLICATION - PROCESSING FOR CHATBOT")
    print("-" * 50)
    
    print("Scenario: Preparing quantum computing text for a chatbot with 4K context window")
    
    # Step 1: Count tokens to check if we need splitting
    total_tokens = tokenizer.count_tokens(sample_text, model_name="gpt-4")
    context_limit = 4000  # Example context window size
    
    print(f"Document tokens: {total_tokens} / {context_limit} context limit")
    
    if total_tokens > context_limit:
        print("Document exceeds context window, will split into chunks")
        # Calculate optimal chunk size (leaving room for prompts and responses)
        optimal_chunk_size = int(context_limit * 0.8)  # 80% of context window
        chunks = tokenizer.split_text_into_chunks(
            sample_text,
            chunk_size=optimal_chunk_size,
            chunk_overlap=50,
            model_name="gpt-4"
        )
        print(f"Split into {len(chunks)} chunks with optimal token sizes")
    else:
        print("Document fits within context window, no splitting needed")
        chunks = [sample_text]
    
    print(f"\nNow each chunk can be processed by the LLM while staying within context limits")
    
    print("\n" + "=" * 80 + "\n")
    print("All examples completed successfully!")
    print("These text processing utilities are essential for handling documents")
    print("and preparing them for LLM processing and vector storage.")
    print("=" * 80)
