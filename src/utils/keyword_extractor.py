"""
Keyword Extractor for Agentic Researcher
Extracts and enhances keywords from queries and research plans
"""
import logging
from typing import Dict, List, Any, Optional

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

from .config import config
from .openai_client import AzureOpenAIClient

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
            response = self.client.chat.completions.create(
                model=self.model,
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
            prompt = f"""Generate {num_similar} similar keywords or search terms related to "{keyword}".
Each keyword should be specific and relevant for research purposes.
Return the keywords as a JSON array of strings.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You generate similar keywords for search."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Extract content
            content = response.choices[0].message.content
            
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
                return similar_keywords
        
        except Exception as e:
            logger.error(f"Error generating similar keywords with LLM: {str(e)}")
            return []


# Example usage when run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create keyword extractor
    extractor = KeywordExtractor()
    
    # Example query and plan
    test_query = "What is volatility index and what is the mathematical formula to calculate the VIX score. Also write a python code to calculate the vix score."
    test_plan = {
        "objective": "Understand the VIX, its calculation, and significance.",
        "research_questions": [
            "What is the Volatility Index (VIX)?",
            "How is the VIX score calculated mathematically?",
            "How can we implement VIX calculation in Python?"
        ],
        "subtasks": [
            {
                "name": "Define VIX",
                "description": "Research the definition and purpose of the Volatility Index"
            },
            {
                "name": "Find Formula",
                "description": "Locate the mathematical formula used for VIX calculation"
            },
            {
                "name": "Implement in Python",
                "description": "Create Python code to calculate VIX scores"
            }
        ]
    }
    
    # Extract keywords
    keywords = extractor.extract_keywords(test_query, test_plan)
    
    print("Extracted Keywords:")
    for i, kw in enumerate(keywords, 1):
        print(f"{i}. {kw}")
    
    if keywords:
        # Generate similar keywords for the first keyword
        print("\nSimilar Keywords:")
        similar = extractor.get_similar_keywords(keywords[0])
        for i, kw in enumerate(similar, 1):
            print(f"{i}. {kw}")
