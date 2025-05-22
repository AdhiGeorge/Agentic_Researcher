"""
Large Output Generator for Agentic Researcher

This module implements a large output generation system that can produce
comprehensive text outputs well beyond standard token limits (50,000+ characters)
by using a chunking and continuation approach.
"""

import os
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import utilities
from src.utils.config import config
from src.utils.openai_client import get_chat_client

# Configure logging
logger = logging.getLogger(__name__)

class LargeOutputGenerator:
    """
    Handles generation of large text outputs beyond standard token limits
    
    This class manages:
    1. Chunked generation of long-form content
    2. Context management between chunks
    3. Seamless continuation of text
    4. Token usage optimization
    
    Attributes:
        config: Configuration settings
        max_tokens_per_chunk: Maximum tokens per generation chunk
        overlap_tokens: Token overlap between chunks for continuity
    """
    
    def __init__(self, max_tokens_per_chunk: int = 4000, overlap_tokens: int = 300):
        """
        Initialize the large output generator.
        
        Args:
            max_tokens_per_chunk: Maximum tokens per generation chunk
            overlap_tokens: Token overlap between chunks for continuity
        """
        self.config = config
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.openai_client = get_chat_client()
        
        logger.info(f"LargeOutputGenerator initialized with max_tokens_per_chunk={max_tokens_per_chunk}")
    
    async def generate_large_output(
        self, 
        prompt: str, 
        system_message: str, 
        target_length: int = 50000,
        format_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a large text output by chunking the generation process.
        
        Args:
            prompt: The input prompt
            system_message: The system message guiding the generation
            target_length: Target character length (approx)
            format_instructions: Optional formatting instructions
            
        Returns:
            Dictionary containing the combined large output and metadata
        """
        start_time = time.time()
        logger.info(f"Starting large output generation with target length: {target_length} characters")
        
        # Estimate number of chunks needed (rough estimate: 1 token ≈ 4 characters)
        estimated_tokens = target_length // 4
        num_chunks = max(1, estimated_tokens // (self.max_tokens_per_chunk - self.overlap_tokens))
        
        logger.info(f"Estimated {num_chunks} chunks for generation")
        
        combined_output = ""
        all_chunks = []
        chunk_metadata = []
        current_chunk_index = 0
        total_tokens_used = 0
        
        # Initial prompt
        current_prompt = prompt
        
        # Add format instructions if provided
        if format_instructions:
            current_prompt = f"{current_prompt}\n\n{format_instructions}"
        
        while len(combined_output) < target_length and current_chunk_index < num_chunks + 5:  # Add safety limit
            is_first_chunk = current_chunk_index == 0
            
            # Modify system message based on chunk position
            if is_first_chunk:
                chunk_system = f"{system_message}\n\nYou are generating the beginning of a long response. Focus on introduction and initial content."
            else:
                chunk_system = f"{system_message}\n\nYou are continuing a response. Continue from the previous content without repeating or introducing new topics that don't naturally follow from the previous content."
                
                # Add continuity instruction
                current_prompt = f"Continue the following text. Pick up exactly where it left off and maintain the same style, tone, and format. Do not repeat what's already written, and do not start with transitional phrases like 'continuing from' or 'to continue'.\n\n--- Previous Content ---\n{combined_output[-self.overlap_tokens*4:]}\n\n--- Continue here ---"
            
            # Generate this chunk
            logger.info(f"Generating chunk {current_chunk_index+1}")
            
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.azure_openai_deployment,
                    messages=[
                        {"role": "system", "content": chunk_system},
                        {"role": "user", "content": current_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=self.max_tokens_per_chunk
                )
                
                # Extract and process the chunk
                chunk_text = response.choices[0].message.content.strip()
                chunk_tokens = response.usage.total_tokens
                total_tokens_used += chunk_tokens
                
                # Store chunk info
                chunk_info = {
                    "index": current_chunk_index,
                    "length": len(chunk_text),
                    "tokens": chunk_tokens
                }
                chunk_metadata.append(chunk_info)
                all_chunks.append(chunk_text)
                
                # Add to combined output (for first chunk) or stitch with overlap handling
                if is_first_chunk:
                    combined_output = chunk_text
                else:
                    # For simplicity, just append - in a production system you'd do smarter joining
                    combined_output += " " + chunk_text
                
                logger.info(f"Chunk {current_chunk_index+1} generated: {len(chunk_text)} chars, {chunk_tokens} tokens")
                
                # Increment chunk index
                current_chunk_index += 1
                
                # Check if we've reached target length
                if len(combined_output) >= target_length:
                    logger.info(f"Reached target length after {current_chunk_index} chunks")
                    break
                    
            except Exception as e:
                logger.error(f"Error generating chunk {current_chunk_index+1}: {str(e)}")
                break
        
        generation_time = time.time() - start_time
        
        logger.info(f"Large output generation completed in {generation_time:.2f}s")
        logger.info(f"Generated {len(combined_output)} characters in {current_chunk_index} chunks")
        logger.info(f"Total tokens used: {total_tokens_used}")
        
        return {
            "text": combined_output,
            "length": len(combined_output),
            "chunks": current_chunk_index,
            "tokens_used": total_tokens_used,
            "generation_time": generation_time,
            "chunks_info": chunk_metadata,
            "timestamp": datetime.now().isoformat()
        }
    
    async def generate_structured_output(
        self,
        prompt: str,
        system_message: str,
        sections: List[Dict[str, str]],
        format_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a large output with predefined sections.
        
        Args:
            prompt: The input prompt
            system_message: The system message
            sections: List of sections with titles and section-specific prompts
            format_instructions: Optional formatting instructions
            
        Returns:
            Dictionary containing the combined structured output and metadata
        """
        start_time = time.time()
        logger.info(f"Starting structured output generation with {len(sections)} sections")
        
        combined_output = ""
        section_outputs = {}
        total_tokens_used = 0
        
        for i, section in enumerate(sections):
            section_title = section.get("title", f"Section {i+1}")
            section_prompt = section.get("prompt", "")
            
            # Create the section-specific prompt
            if i == 0:
                # First section includes the main prompt
                full_prompt = f"{prompt}\n\n{section_prompt}"
            else:
                # Later sections reference the original prompt
                full_prompt = f"For the original query: \"{prompt}\"\n\n{section_prompt}"
            
            # Add format instructions if provided
            if format_instructions:
                full_prompt = f"{full_prompt}\n\n{format_instructions}"
            
            # Adjust system message for the section
            section_system = f"{system_message}\n\nYou are generating the {section_title} section of a comprehensive document."
            
            try:
                logger.info(f"Generating section: {section_title}")
                
                response = self.openai_client.chat.completions.create(
                    model=self.config.azure_openai_deployment,
                    messages=[
                        {"role": "system", "content": section_system},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=self.max_tokens_per_chunk
                )
                
                # Extract and process the section
                section_text = response.choices[0].message.content.strip()
                section_tokens = response.usage.total_tokens
                total_tokens_used += section_tokens
                
                # Store section info
                section_outputs[section_title] = {
                    "text": section_text,
                    "length": len(section_text),
                    "tokens": section_tokens
                }
                
                # Add formatted section to combined output
                combined_output += f"\n\n## {section_title}\n\n{section_text}"
                
                logger.info(f"Section '{section_title}' generated: {len(section_text)} chars, {section_tokens} tokens")
                
            except Exception as e:
                logger.error(f"Error generating section '{section_title}': {str(e)}")
                section_outputs[section_title] = {
                    "text": f"Error generating this section: {str(e)}",
                    "length": 0,
                    "tokens": 0,
                    "error": str(e)
                }
        
        generation_time = time.time() - start_time
        
        logger.info(f"Structured output generation completed in {generation_time:.2f}s")
        logger.info(f"Generated {len(combined_output)} characters across {len(sections)} sections")
        logger.info(f"Total tokens used: {total_tokens_used}")
        
        return {
            "text": combined_output.strip(),
            "length": len(combined_output),
            "sections": section_outputs,
            "tokens_used": total_tokens_used,
            "generation_time": generation_time,
            "timestamp": datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Simple test
    async def test_large_output():
        generator = LargeOutputGenerator()
        
        system_msg = "You are a helpful research assistant that provides comprehensive, detailed information."
        prompt = "Write a detailed explanation of quantum computing, including history, current state, and future prospects."
        
        result = await generator.generate_large_output(
            prompt=prompt,
            system_message=system_msg,
            target_length=10000  # Target 10K characters for testing
        )
        
        print(f"Generated {result['length']} characters")
        print(f"Number of chunks: {result['chunks']}")
        print(f"Tokens used: {result['tokens_used']}")
        print("\nSample of output:")
        print(result['text'][:500] + "...")
    
    # Run the test
    import asyncio
    asyncio.run(test_large_output())
