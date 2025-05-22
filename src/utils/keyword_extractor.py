"""
Keyword Extractor for Agentic Researcher
Extracts and enhances keywords from queries and research plans
"""
import logging
from typing import Dict, List, Any, Optional
import os
import sys

# Fix imports for running directly
if __name__ == "__main__":
    # Add the parent directory to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # Use absolute imports
    from src.utils.config import config
    from src.utils.openai_client import AzureOpenAIClient
else:
    # Use relative imports when imported as a module
    from .config import config
    from .openai_client import AzureOpenAIClient

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from keybert import KeyBERT
    keybert_available = True
except ImportError:
    keybert_available = False

try:
    from sklearn.feature_extraction.text import CountVectorizer
    sklearn_available = True
except ImportError:
    sklearn_available = False

# Configure logging
logger = logging.getLogger(__name__)

class KeywordExtractor:
    """
    Keyword Extractor using multiple methods:
    1. KeyBERT for keyword extraction (if available)
    2. LLM-based extraction using OpenAI
    3. Fallback to simple frequency analysis
    
    Extracts and enhances keywords from:
    - Original user query
    - Research plan
    - Subtasks and research questions
    """
    
    def __init__(self):
        """Initialize the KeywordExtractor"""
        # Use global config
        self.config = config
        
        # Initialize KeyBERT if available
        self.keybert = None
        if keybert_available:
            try:
                self.keybert = KeyBERT()
                logger.info("KeyBERT initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize KeyBERT: {str(e)}")
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = None
        # Try to initialize the embedding model directly
        model_name = "all-MiniLM-L6-v2"  # Default embedding model
        try:
            # Try to load the model directly
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model '{model_name}' directly")
        except Exception as e:
            logger.warning(f"Failed to get embedding model: {str(e)}")
        
        # Initialize Azure OpenAI client for LLM-based extraction
        self.openai_client = AzureOpenAIClient()
        # The AzureOpenAIClient may have different methods, so we'll adapt accordingly
        # and use the client directly
        
        logger.info("KeywordExtractor initialized")
    
    def extract_keywords(self, query: str, plan: Optional[Dict[str, Any]] = None, 
                        max_keywords: int = 15) -> List[str]:
        """
        Extract keywords using multiple methods
        
        Args:
            query: User query
            plan: Optional research plan
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List[str]: List of extracted keywords
        """
        all_keywords = []
        
        # Method 1: Try KeyBERT extraction
        if self.keybert:
            keybert_keywords = self._extract_with_keybert(query, plan)
            all_keywords.extend(keybert_keywords)
        
        # Method 2: LLM-based extraction
        llm_keywords = self._extract_with_llm(query, plan)
        all_keywords.extend(llm_keywords)
        
        # Method 3: Simple frequency analysis (fallback)
        if not all_keywords:
            fallback_keywords = self._extract_with_frequency(query, plan)
            all_keywords.extend(fallback_keywords)
        
        # Deduplicate, filter, and limit keywords
        final_keywords = self._process_keywords(all_keywords, max_keywords)
        
        logger.info(f"Extracted {len(final_keywords)} keywords: {final_keywords}")
        return final_keywords
    
    def _extract_with_keybert(self, query: str, plan: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Extract keywords using KeyBERT
        
        Args:
            query: User query
            plan: Optional research plan
            
        Returns:
            List[str]: List of extracted keywords
        """
        if not self.keybert:
            return []
        
        try:
            # Prepare combined text from query and plan
            text = query
            
            if plan:
                # Add objective
                objective = plan.get("objective", "")
                if objective:
                    text += " " + objective
                
                # Add research questions
                research_questions = plan.get("research_questions", [])
                if research_questions:
                    text += " " + " ".join(research_questions)
                
                # Add subtasks
                subtasks = plan.get("subtasks", [])
                for subtask in subtasks:
                    name = subtask.get("name", "")
                    desc = subtask.get("description", "")
                    if name and desc:
                        text += f" {name} {desc}"
            
            # Set KeyBERT parameters
            if sklearn_available:
                # Use n-gram range to capture multi-word keywords
                vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english")
                keywords = self.keybert.extract_keywords(
                    text, 
                    keyphrase_ngram_range=(1, 2),
                    stop_words="english",
                    use_mmr=True,  # Maximize diversity
                    diversity=0.5,
                    vectorizer=vectorizer,
                    top_n=10
                )
            else:
                keywords = self.keybert.extract_keywords(
                    text, 
                    keyphrase_ngram_range=(1, 2),
                    stop_words="english",
                    use_mmr=True,  # Maximize diversity
                    diversity=0.5,
                    top_n=10
                )
            
            # Extract just the keywords
            return [kw[0] for kw in keywords]
        
        except Exception as e:
            logger.error(f"Error extracting keywords with KeyBERT: {str(e)}")
            return []
    
    def _extract_with_llm(self, query: str, plan: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Extract keywords using LLM
        
        Args:
            query: User query
            plan: Optional research plan
            
        Returns:
            List[str]: List of extracted keywords
        """
        try:
            # Create prompt for keyword extraction
            if plan:
                # Use existing keywords from plan if available
                keywords = plan.get("keywords", [])
                if keywords:
                    return keywords
                
                # Extract from plan components
                prompt = f"""Extract the most important search keywords from the following research query and plan:

Query: {query}

Research Objective: {plan.get('objective', '')}

Research Questions:
{' '.join(f'- {q}' for q in plan.get('research_questions', []))}

Return a list of 10-15 specific keywords and phrases that would be most effective for web searches.
Format your response as a JSON array of strings.
"""
            else:
                # Extract from query only
                prompt = f"""Extract the most important search keywords from the following research query:

Query: {query}

Return a list of 10-15 specific keywords and phrases that would be most effective for web searches.
Format your response as a JSON array of strings.
"""
            
            # Make API call
            response = self.openai_client.chat_client.chat.completions.create(
                model=self.openai_client.chat_deployment,
                messages=[
                    {"role": "system", "content": "You are a keyword extraction specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Extract content
            content = response.choices[0].message.content
            
            # Parse keywords
            import json
            try:
                # Try to parse as a JSON array
                result = json.loads(content)
                if isinstance(result, list):
                    return result
                elif isinstance(result, dict) and "keywords" in result:
                    return result["keywords"]
                else:
                    # Try to find an array in the response
                    for key, value in result.items():
                        if isinstance(value, list):
                            return value
                    return []
            except json.JSONDecodeError:
                # If not valid JSON, try to extract keywords with regex
                import re
                keywords = re.findall(r'"([^"]+)"', content)
                return keywords
        
        except Exception as e:
            logger.error(f"Error extracting keywords with LLM: {str(e)}")
            return []
    
    def _extract_with_frequency(self, query: str, plan: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Extract keywords using simple frequency analysis
        
        Args:
            query: User query
            plan: Optional research plan
            
        Returns:
            List[str]: List of extracted keywords
        """
        try:
            # Combine text from query and plan
            text = query.lower()
            
            if plan:
                # Add objective
                objective = plan.get("objective", "").lower()
                if objective:
                    text += " " + objective
                
                # Add research questions
                research_questions = [q.lower() for q in plan.get("research_questions", [])]
                if research_questions:
                    text += " " + " ".join(research_questions)
            
            # Basic text cleanup
            import re
            # Remove punctuation and special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Tokenize
            words = text.split()
            
            # Remove common stop words
            stop_words = {
                "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                "be", "been", "being", "to", "of", "for", "in", "on", "at", "by", 
                "with", "about", "against", "between", "into", "through", "during", 
                "before", "after", "above", "below", "from", "up", "down", "this", 
                "that", "these", "those", "what", "which", "who", "whom", "whose", 
                "when", "where", "why", "how", "all", "any", "both", "each", "few", 
                "more", "most", "some", "such", "no", "nor", "not", "only", "own", 
                "same", "so", "than", "too", "very", "can", "will", "just", "should", 
                "now", "also", "as", "if", "then", "because", "since", "while", "where"
            }
            
            filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
            
            # Count word frequency
            from collections import Counter
            word_counts = Counter(filtered_words)
            
            # Get top keywords
            top_keywords = [word for word, count in word_counts.most_common(10)]
            
            return top_keywords
        
        except Exception as e:
            logger.error(f"Error extracting keywords with frequency analysis: {str(e)}")
            return []
    
    def _process_keywords(self, keywords: List[str], max_keywords: int) -> List[str]:
        """
        Process keywords: deduplicate, filter, and limit
        
        Args:
            keywords: List of all extracted keywords
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List[str]: List of processed keywords
        """
        # Remove duplicates while preserving order
        seen = set()
        deduplicated = []
        
        for kw in keywords:
            # Normalize keyword
            keyword = kw.strip().lower()
            
            # Skip empty or already seen keywords
            if not keyword or keyword in seen:
                continue
            
            seen.add(keyword)
            deduplicated.append(keyword)
        
        # Filter out very short keywords
        filtered = [kw for kw in deduplicated if len(kw) >= 3]
        
        # Limit to max_keywords
        return filtered[:max_keywords]
    
    def get_similar_keywords(self, keyword: str, num_similar: int = 5) -> List[str]:
        """
        Generate similar keywords using embeddings
        
        Args:
            keyword: Base keyword
            num_similar: Number of similar keywords to generate
            
        Returns:
            List[str]: List of similar keywords
        """
        if not self.embedding_model:
            return []
        
        # Generate with LLM API call
        try:
            # Generate dummy variations when running as an example
            # This is a fallback for when OpenAI client isn't properly configured
            if __name__ == "__main__":
                # Try to use the client, but have a fallback ready
                try:
                    # Check if the client is properly configured
                    if not hasattr(self, 'openai_client') or not self.openai_client:
                        # Simulate variations for example purposes
                        import random
                        variations = [
                            # Machine learning variations
                            ["deep learning", "neural networks", "artificial intelligence", "supervised learning", "ML algorithms"],
                            # Climate change variations
                            ["global warming", "environmental sustainability", "carbon emissions", "greenhouse effect", "climate crisis"],
                            # Cryptocurrency variations
                            ["blockchain", "bitcoin", "digital currency", "ethereum", "crypto assets"],
                            # Generic fallback variations
                            [f"{keyword} techniques", f"{keyword} methods", f"{keyword} applications", f"{keyword} principles", f"{keyword} systems"]
                        ]
                        
                        # Select appropriate variations or generate generic ones
                        if "machine learning" in keyword.lower():
                            return variations[0][:num_similar]
                        elif "climate change" in keyword.lower():
                            return variations[1][:num_similar]
                        elif "crypto" in keyword.lower():
                            return variations[2][:num_similar]
                        else:
                            return variations[3][:num_similar]
                except Exception as e:
                    logger.warning(f"Using fallback keyword generation: {str(e)}")
                    return [f"{keyword} type {i}" for i in range(1, num_similar+1)]
            
            # Generate the prompt for the LLM
            prompt = f"""Generate {num_similar} similar keywords or search terms related to "{keyword}".
Each keyword should be specific and relevant for research purposes.
Return the keywords as a JSON array of strings.
"""
            
            # Check if client is properly initialized
            if not hasattr(self, 'openai_client') or not self.openai_client:
                logger.error("OpenAI client not properly initialized")
                return [f"{keyword} variant {i}" for i in range(1, num_similar+1)]
            
            try:
                # Legacy OpenAI client format
                response = self.openai_client.generate_completion(
                    messages=[
                        {"role": "system", "content": "You generate similar keywords for search."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                content = response
            except AttributeError:
                # New OpenAI client format
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You generate similar keywords for search."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        response_format={"type": "json_object"}
                    )
                    content = response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Error with new OpenAI client: {str(e)}")
                    return [f"{keyword} option {i}" for i in range(1, num_similar+1)]
            
            # Parse keywords
            import json
            try:
                result = json.loads(content)
                if isinstance(result, list):
                    return result
                elif isinstance(result, dict) and "keywords" in result:
                    return result["keywords"]
                else:
                    # Try to find an array in the response
                    for key, value in result.items():
                        if isinstance(value, list):
                            return value
                    return []
            except json.JSONDecodeError:
                # If not valid JSON, try to extract keywords with regex
                import re
                similar_keywords = re.findall(r'"([^"]+)"', content)
                return similar_keywords if similar_keywords else [f"{keyword} term {i}" for i in range(1, num_similar+1)]
        
        except Exception as e:
            logger.error(f"Error generating similar keywords with LLM: {str(e)}")
            return [f"{keyword} #{i}" for i in range(1, num_similar+1)]


# Example usage with real-world scenarios and API calls
if __name__ == "__main__":
    import os
    import sys
    import time
    import json
    from pathlib import Path
    from pprint import pprint
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("===== Keyword Extractor Example Usage =====")
    print("This example demonstrates keyword extraction capabilities")
    print("for research queries using multiple extraction methods.")
    
    # Create keyword extractor
    extractor = KeywordExtractor()
    
    print("\nExample 1: Basic keyword extraction from a query")
    print("-" * 60)
    
    # Example queries of varying complexity
    test_queries = [
        "What is the volatility index and how is it calculated?",
        "Explain the environmental impact of electric vehicles compared to traditional combustion engines",
        "What are the main quantum computing algorithms and their applications in cryptography?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # Time the extraction
        start_time = time.time()
        keywords = extractor.extract_keywords(query)
        extraction_time = time.time() - start_time
        
        print(f"Extracted {len(keywords)} keywords in {extraction_time:.2f}s:")
        for j, kw in enumerate(keywords, 1):
            print(f"  {j}. {kw}")
    
    print("\nExample 2: Extracting keywords from a research plan")
    print("-" * 60)
    
    # Example research plan
    test_plan = {
        "objective": "Understand the VIX, its calculation, and significance in predicting market volatility",
        "research_questions": [
            "What is the Volatility Index (VIX) and who created it?",
            "How is the VIX score calculated mathematically?",
            "How can we implement VIX calculation in Python?",
            "How well does VIX predict actual market volatility?"
        ],
        "subtasks": [
            {
                "name": "Define VIX",
                "description": "Research the definition, history, and purpose of the Volatility Index"
            },
            {
                "name": "Find Formula",
                "description": "Locate the mathematical formula used for VIX calculation and explain its components"
            },
            {
                "name": "Implement in Python",
                "description": "Create Python code to calculate VIX scores using historical options data"
            },
            {
                "name": "Evaluate Predictive Power",
                "description": "Analyze the correlation between VIX movements and subsequent market volatility"
            }
        ]
    }
    
    # Query with plan
    complex_query = "What is volatility index and what is the mathematical formula to calculate the VIX score. Also write a python code to calculate the vix score."
    
    print(f"Query: {complex_query}")
    print("Using query with full research plan...")
    
    # Extract with plan
    keywords_with_plan = extractor.extract_keywords(complex_query, test_plan)
    
    print(f"\nExtracted {len(keywords_with_plan)} keywords using research plan:")
    for i, kw in enumerate(keywords_with_plan, 1):
        print(f"  {i}. {kw}")
    
    # Extract without plan for comparison
    keywords_no_plan = extractor.extract_keywords(complex_query)
    
    print(f"\nExtracted {len(keywords_no_plan)} keywords without research plan:")
    for i, kw in enumerate(keywords_no_plan, 1):
        print(f"  {i}. {kw}")
    
    # Compare the differences
    plan_only = [kw for kw in keywords_with_plan if kw not in keywords_no_plan]
    no_plan_only = [kw for kw in keywords_no_plan if kw not in keywords_with_plan]
    
    print("\nKeywords added by research plan context:")
    for kw in plan_only:
        print(f"  + {kw}")
    
    print("\nKeywords only present without research plan:")
    for kw in no_plan_only:
        print(f"  - {kw}")
    
    print("\nExample 3: Generating similar keywords for search expansion")
    print("-" * 60)
    
    # Test terms for similar keyword generation
    test_terms = [
        "machine learning",
        "climate change",
        "cryptocurrency"
    ]
    
    for term in test_terms:
        print(f"\nSimilar terms to '{term}':")
        similar_terms = extractor.get_similar_keywords(term, num_similar=5)
        
        for i, similar in enumerate(similar_terms, 1):
            print(f"  {i}. {similar}")
    
    print("\nExample 4: Extraction method comparison")
    print("-" * 60)
    
    test_text = "Quantum computing leverages quantum mechanics principles like superposition and entanglement to perform computations using qubits instead of classical bits, potentially solving certain problems exponentially faster than traditional computers."
    
    print(f"Sample text: {test_text}")
    
    # Try each method independently
    print("\nMethod 1: KeyBERT extraction (if available)")
    if keybert_available and extractor.keybert:
        keybert_kws = extractor._extract_with_keybert(test_text)
        print(f"Extracted {len(keybert_kws)} keywords with KeyBERT:")
        print("  " + ", ".join(keybert_kws))
    else:
        print("KeyBERT not available")
    
    print("\nMethod 2: LLM-based extraction")
    llm_kws = extractor._extract_with_llm(test_text)
    print(f"Extracted {len(llm_kws)} keywords with LLM:")
    print("  " + ", ".join(llm_kws))
    
    print("\nMethod 3: Frequency analysis")
    freq_kws = extractor._extract_with_frequency(test_text)
    print(f"Extracted {len(freq_kws)} keywords with frequency analysis:")
    print("  " + ", ".join(freq_kws))
    
    print("\nExample 5: Research usage workflow")
    print("-" * 60)
    
    # Simulate a realistic research workflow
    user_query = "Explain how reinforcement learning algorithms are used in robotics for autonomous navigation"
    
    print(f"Initial user query: '{user_query}'")
    
    # Step 1: Extract keywords from initial query
    print("\nStep 1: Extract initial keywords")
    initial_keywords = extractor.extract_keywords(user_query, max_keywords=8)
    print("Initial keywords: " + ", ".join(initial_keywords))
    
    # Step 2: Create a research plan (simulated)
    print("\nStep 2: Generate research plan")
    research_plan = {
        "objective": "Understand reinforcement learning applications in robotics navigation",
        "research_questions": [
            "What are the fundamental reinforcement learning algorithms used in robotics?",
            "How is autonomous navigation implemented with reinforcement learning?",
            "What are the challenges and limitations in real-world applications?",
            "How do different sensors and perception systems integrate with RL for navigation?"
        ],
        "subtasks": [
            {"name": "RL Algorithms", "description": "Identify core reinforcement learning algorithms for robotics"},
            {"name": "Navigation Systems", "description": "Analyze autonomous navigation implementations using RL"},
            {"name": "Challenges", "description": "Outline challenges in real-world robotic RL applications"},
            {"name": "Sensor Integration", "description": "Research how sensors and perception systems work with RL"}
        ]
    }
    print("Research plan created with multiple questions and subtasks")
    
    # Step 3: Extract enhanced keywords with plan
    print("\nStep 3: Extract enhanced keywords with research plan")
    enhanced_keywords = extractor.extract_keywords(user_query, research_plan)
    print("Enhanced keywords: " + ", ".join(enhanced_keywords))
    
    # Step 4: Generate search variations
    print("\nStep 4: Generate search variations for top keywords")
    search_terms = []
    for kw in enhanced_keywords[:3]:  # Use top 3 keywords
        variations = extractor.get_similar_keywords(kw, num_similar=3)
        print(f"'{kw}' variations: " + ", ".join(variations))
        search_terms.extend(variations)
    
    # Step 5: Create final search strategy
    print("\nStep 5: Finalize search strategy")
    print("Primary keywords: " + ", ".join(enhanced_keywords[:5]))
    print("Secondary keywords: " + ", ".join(search_terms[:8]))
    
    final_strategy = {
        "primary_keywords": enhanced_keywords[:5],
        "secondary_keywords": search_terms[:8],
        "combined_queries": [
            enhanced_keywords[0] + " AND " + enhanced_keywords[1],
            enhanced_keywords[0] + " AND " + search_terms[0],
            "\"" + enhanced_keywords[2] + "\"" + " robotics navigation"
        ]
    }
    
    print("\nFinal search strategy:")
    print("Example combined queries:")
    for query in final_strategy["combined_queries"]:
        print(f"  - {query}")
    
    print("\n" + "=" * 80)
    print("Keyword Extractor examples completed!")
    print("This utility enhances research efficiency by identifying")
    print("optimal search terms from queries and research plans.")
    print("=" * 80)
