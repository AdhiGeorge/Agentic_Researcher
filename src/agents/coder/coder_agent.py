"""Coder Agent for Agentic Researcher
Generates code based on research results and requirements
"""
import json
import os
import sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import re
from typing import Dict, List, Any, Optional, Tuple

from src.utils.config import Config
from src.db.sqlite_manager import SQLiteManager
from src.db.qdrant_manager import QdrantManager
from src.utils.openai_client import AzureOpenAIClient
from src.utils.response_adapter import extract_openai_response_content, safe_parse_json

class CoderAgent:
    """
    Coder Agent that generates code based on research findings
    Uses Azure OpenAI to write, analyze, and explain code
    """
    
    def __init__(self, use_db=True):
        # Load configuration
        self.config = Config()
        
        # Initialize databases if use_db is True (otherwise mock them for standalone execution)
        if use_db:
            # SQL Logger for tracking state - can cause errors with semantic_hash column
            try:
                self.logger = SQLiteManager()
            except Exception as e:
                print(f"Warning: Failed to initialize SQLiteManager: {e}")
                # Create a mock logger with a save_agent_state method
                class MockLogger:
                    def save_agent_state(self, **kwargs):
                        print(f"Mock logging: {kwargs}")
                self.logger = MockLogger()
            
            # Vector database for retrieving research data
            try:
                self.vector_db = QdrantManager(collection_name="preprocessed_chunks")
            except Exception as e:
                print(f"Warning: Failed to initialize QdrantManager: {e}")
                # Create a mock vector_db with a search method
                class MockVectorDB:
                    def search(self, **kwargs):
                        print(f"Mock vector search with: {kwargs}")
                        return []
                self.vector_db = MockVectorDB()
        else:
            # Create mock objects for standalone execution
            class MockLogger:
                def save_agent_state(self, **kwargs):
                    print(f"Mock logging: {kwargs}")
            self.logger = MockLogger()
            
            class MockVectorDB:
                def search(self, **kwargs):
                    print(f"Mock vector search with: {kwargs}")
                    return []
            self.vector_db = MockVectorDB()
        
        # Get the Azure OpenAI client
        self.openai_client = AzureOpenAIClient()
        # No longer need to access client directly - will use generate_completion method
        
        # Azure OpenAI model for coding
        self.model = self.config.azure_openai_deployment
    
    def create_prompt(self, query: str, plan: Dict[str, Any], 
                     relevant_chunks: List[Dict[str, Any]], 
                     requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Create prompt for the coder agent
        
        Args:
            query: Original user query
            plan: Research plan from planner agent
            relevant_chunks: Relevant text chunks from research
            requirements: Optional coding requirements
            
        Returns:
            str: Full prompt for the coder
        """
        # Extract relevant information from the plan
        objective = plan.get("objective", "")
        
        # Extract content from chunks
        chunk_texts = []
        for chunk in relevant_chunks:
            content = chunk.get("content", "")
            source = chunk.get("metadata", {}).get("url", "Unknown source")
            chunk_texts.append(f"SOURCE: {source}\n\n{content}\n")
        
        chunk_content = "\n---\n".join(chunk_texts)
        
        # Extract coding requirements if provided
        language = "Python"  # Default
        libraries = []
        functionality = []
        
        if requirements:
            language = requirements.get("language", language)
            libraries = requirements.get("libraries", [])
            functionality = requirements.get("functionality", [])
        
        libraries_str = "\n".join([f"- {lib}" for lib in libraries])
        functionality_str = "\n".join([f"- {func}" for func in functionality])
        
        # Base prompt template
        prompt = f"""You are an expert {language} developer. Your task is to write code based on research findings and requirements.

ORIGINAL QUERY: {query}

OBJECTIVE: {objective}

RESEARCH FINDINGS:
{chunk_content}

CODING REQUIREMENTS:
- Language: {language}
- Libraries/Frameworks:
{libraries_str if libraries else "- Use appropriate libraries based on requirements"}
- Required Functionality:
{functionality_str if functionality else "- Implement functionality to meet the objective"}

Your task:
1. Analyze the research findings to understand the requirements
2. Write clean, well-structured, and well-documented {language} code
3. Include appropriate error handling and edge cases
4. Explain your implementation choices in comments
5. Provide instructions on how to use the code

Format your response as a JSON object:
{{
    "explanation": "Brief explanation of the code's purpose and approach",
    "files": [
        {{
            "file_name": "filename.{language.lower()}",
            "content": "Full source code including docstrings and comments"
        }},
        ...
    ],
    "usage_instructions": "How to use the code, including any setup or dependencies"
}}
"""
        
        return prompt
    
    def execute(self, query: str, plan: Dict[str, Any], chunked_results: Dict[str, Any], 
                project_id: int, requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the coder agent to generate code
        
        Args:
            query: Original user query
            plan: Research plan from planner agent
            chunked_results: Results from chunker agent
            project_id: Project ID
            requirements: Optional coding requirements
            
        Returns:
            Dict: Generated code and explanations
        """
        # Log the agent state - starting code generation
        self.logger.save_agent_state(
            project_id=project_id,
            agent_type="coder",
            state_data={"status": "generating"}
        )
        
        try:
            # Get relevant chunks from the chunked results
            chunks = chunked_results.get("chunks", [])
            
            if not chunks:
                # If no chunks in results, get them directly from vector DB
                chunks = self._get_relevant_chunks(query, project_id)
            
            # Create the prompt
            prompt = self.create_prompt(query, plan, chunks, requirements)
            
            # Make API call to Azure OpenAI using the client's generate_completion method
            messages = [
                {"role": "system", "content": "You are an expert software developer."},
                {"role": "user", "content": prompt}
            ]
            
            # Use the generate_completion method which handles the API call
            response = self.openai_client.generate_completion(
                messages=messages,
                model=self.model,
                temperature=0.2,
                max_tokens=3000,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Process the response - generate_completion already returns content as string
            content = response
            
            try:
                code_result = safe_parse_json(content)
            except json.JSONDecodeError:
                # If direct parsing fails, extract code files manually
                code_result = self._extract_code_files(content)
            
            # Log the agent state - completed code generation
            self.logger.save_agent_state(
                project_id=project_id,
                agent_type="coder",
                state_data={"status": "completed", 
                           "files_generated": len(code_result.get("files", []))}
            )
            
            return code_result
        
        except Exception as e:
            # Log the agent state - error
            self.logger.save_agent_state(
                project_id=project_id,
                agent_type="coder",
                state_data={"status": "error", "error": str(e)}
            )
            
            raise Exception(f"Code generation failed: {str(e)}")
    
    def _get_relevant_chunks(self, query: str, project_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get relevant chunks for the query
        
        Args:
            query: User query
            project_id: Project ID
            limit: Maximum number of chunks to retrieve
            
        Returns:
            List[Dict]: List of relevant chunks
        """
        try:
            # Retrieve relevant chunks from vector DB
            results = self.vector_db.search(
                query=query,
                filter_dict={"project_id": project_id},
                limit=limit
            )
            return results
        except Exception as e:
            print(f"Error retrieving chunks: {str(e)}")
            return []
    
    def _extract_code_files(self, content: str):
        """
        Extract code files from non-JSON response
        
        Args:
            content: Response content
            
        Returns:
            Dict: Structured code results
        """
        # Initialize result structure
        result = {
            "explanation": "",
            "files": [],
            "usage_instructions": ""
        }
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:(?P<lang>\w+)\n)?(?P<code>.*?)```', content, re.DOTALL)
        
        # Extract filenames and content
        files_info = []
        current_file = None
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Look for file name patterns
            file_match = re.match(r'^(?:file|filename):\s*["\']*([\w\.-]+)["\']*\s*$', line, re.IGNORECASE)
            if file_match:
                current_file = file_match.group(1)
                continue
                
            # Alternative file pattern: heading with .py or similar extension
            if line.startswith('#') and ('.' in line) and not current_file:
                file_match = re.search(r'["\']*([\w\.-]+\.[a-z]{1,5})["\']*', line)
                if file_match:
                    current_file = file_match.group(1)
                    continue
        
        # Process code blocks
        for lang, code in code_blocks:
            # If we've identified a filename for this code block, use it
            if current_file:
                file_name = current_file
                current_file = None  # Reset for next block
            else:
                # If no filename found, create a generic one based on language
                if lang and lang.strip():
                    extension = lang.strip().lower()
                    # Map common language names to file extensions
                    ext_map = {"python": "py", "javascript": "js", "typescript": "ts", "html": "html", "css": "css"}
                    ext = ext_map.get(extension, extension)
                    file_name = f"main.{ext}"
                else:
                    # Default to Python if no language specified
                    file_name = "main.py"
            
            # Clean up code and add to files list
            clean_code = code.strip()
            files_info.append({"file_name": file_name, "content": clean_code})
        
        # Extract explanation (typically at the beginning)
        explanation_match = re.search(r'^(.*?)(?=```|#|import|from|class|def)', content, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""    
        
        # If no explanation found but there are paragraphs, use the first one
        if not explanation and '\n\n' in content:
            first_para = content.split('\n\n')[0].strip()
            if first_para and not first_para.startswith('```'):
                explanation = first_para
        
        # Extract usage instructions (typically at the end)
        usage_match = re.search(r'(?:usage|instructions|how to use)[:\s]+(.*?)(?:\n\n|\n#|\Z)', content, re.IGNORECASE | re.DOTALL)
        usage = usage_match.group(1).strip() if usage_match else "See code comments for usage instructions."
        
        return {
            "explanation": explanation,
            "files": files_info,
            "usage_instructions": usage
        }
    
    def refine_code(self, code_result: Dict[str, Any], feedback: str, project_id: int) -> Dict[str, Any]:
        """
        Refine code based on feedback
        
        Args:
            code_result: Original code generated
            feedback: User feedback
            project_id: Project ID
            
        Returns:
            Dict: Refined code
        """
        # Log the agent state - starting refinement
        self.logger.save_agent_state(
            project_id=project_id,
            agent_type="coder",
            state_data={"status": "refining", "feedback": feedback}
        )
        
        try:
            # Convert code files to text for the prompt
            code_files_text = ""
            for file in code_result.get("files", []):
                name = file.get("file_name", "unknown")
                content = file.get("content", "")
                code_files_text += f"FILE: {name}\n```\n{content}\n```\n\n"
            
            # Create refinement prompt
            refine_prompt = f"""You previously generated the following code:

{code_files_text}

EXPLANATION: {code_result.get('explanation', '')}

USAGE INSTRUCTIONS: {code_result.get('usage_instructions', '')}

The user has provided the following feedback:

{feedback}

Please refine the code based on this feedback. Maintain the same JSON structure in your response:
{{
    "explanation": "...",
    "files": [
        {{
            "file_name": "...",
            "content": "..."
        }},
        ...
    ],
    "usage_instructions": "..."
}}

Make specific changes to address the feedback while preserving the overall structure.
"""
            
            # Make API call to Azure OpenAI using the client's generate_completion method
            messages = [
                {"role": "system", "content": "You are an expert software developer."},
                {"role": "user", "content": refine_prompt}
            ]
            
            # Use the generate_completion method which handles the API call
            response = self.openai_client.generate_completion(
                messages=messages,
                model=self.model,
                temperature=0.2,
                max_tokens=3000,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Process the response - generate_completion already returns content as string
            content = response
            
            try:
                refined_code = json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from markdown
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    refined_code = json.loads(json_match.group(1))
                else:
                    # Extract code files manually if JSON parsing fails
                    refined_code = self._extract_code_files(content)
            
            # Log the agent state - completed refinement
            self.logger.save_agent_state(
                project_id=project_id,
                agent_type="coder",
                state_data={"status": "refinement_completed", 
                           "files_generated": len(refined_code.get("files", []))}
            )
            
            return refined_code
        
        except Exception as e:
            # Log the agent state - error
            self.logger.save_agent_state(
                project_id=project_id,
                agent_type="coder",
                state_data={"status": "refinement_error", "error": str(e)}
            )
            
            raise Exception(f"Code refinement failed: {str(e)}")


# Example usage
if __name__ == "__main__":
    print("\n===== CoderAgent Example =====\n")
    print("Initializing CoderAgent...")
    
    # Initialize CoderAgent with use_db=False to bypass database initialization
    coder = CoderAgent(use_db=False)
    
    # Example research data for generating code
    example_plan = {
        "objective": "Calculate volatility index (VIX) using standard formula",
        "steps": [
            "Understand the VIX formula",
            "Implement the calculation in Python",
            "Add error handling and documentation"
        ]
    }
    
    # Example chunks that would come from research
    example_chunks = {
        "chunks": [
            {
                "content": "The VIX formula involves calculating the weighted average of S&P 500 option prices. "
                          "It uses the formula: VIX = 100 * sqrt(T * sum((delta K_i / K_i^2) * e^(rT) * Q(K_i)))",
                "metadata": {"url": "https://example.com/vix-formula"}
            },
            {
                "content": "To calculate VIX in Python, you will need to: \n"
                          "1. Gather option prices for S&P 500\n"
                          "2. Calculate the time to expiration (T)\n"
                          "3. Use the risk-free interest rate (r)\n"
                          "4. Find the intervals between strike prices (delta K_i)\n"
                          "5. Compute the weighted contribution of each option\n"
                          "6. Apply the final formula",
                "metadata": {"url": "https://example.com/vix-implementation"}
            }
        ]
    }
    
    print("\nExecuting CoderAgent...")
    print("Query: Write Python code to calculate VIX score")
    print(f"Plan: {example_plan['objective']}")
    print(f"Chunks: {len(example_chunks['chunks'])} research sources")
    
    # Instead of making actual API calls, use pre-defined example results
    print("\nGenerating code based on research...")
    
    # Print the explanation
    print("\nEXPLANATION:")
    explanation = ("This code implements the VIX calculation formula based on the provided research. "
                  "It calculates the Volatility Index (VIX) using S&P 500 option prices, applying "
                  "the standard formula: VIX = 100 * sqrt(T * sum((delta K_i / K_i^2) * e^(rT) * Q(K_i))). "
                  "The implementation handles option data processing, interval calculations, and weighted averages.")
    print(explanation)
    
# Real-world example usage
if __name__ == "__main__":
    import time
    import os
    import logging
    from datetime import datetime
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("\n" + "=" * 80)
    print("CoderAgent - Real Example Usage")
    print("=" * 80)
    
    # Initialize the CoderAgent (no DB connection for standalone example)
    print("\nInitializing CoderAgent...")
    coder_agent = CoderAgent(use_db=False)
    
    # Define a research query that we want to generate code for
    query = "How can I create a sentiment analysis tool for social media posts?"
    print(f"\nResearch query: '{query}'")
    
    # This would normally come from a planner agent
    research_plan = {
        "objective": "Create a Python sentiment analysis tool for social media posts",
        "research_questions": [
            "What NLP libraries can be used for sentiment analysis?",
            "How to preprocess social media text data?",
            "What machine learning approaches work best for sentiment analysis?"
        ],
        "expected_outputs": [
            "Python script for sentiment analysis",
            "Documentation on how to use it",
            "Example with sample data"
        ]
    }
    print("\nResearch plan created with objective:", research_plan["objective"])
    
    # Create simulated research results that would normally come from a search/research agent
    # These are the chunks of information from different sources
    research_chunks = [
        {
            "content": "The VADER (Valence Aware Dictionary and sEntiment Reasoner) library is specifically designed for social media sentiment analysis. It's part of the NLTK package and works well with text that includes emojis, slang, and informal language common in social media posts.",
            "metadata": {"url": "https://github.com/cjhutto/vaderSentiment", "title": "VADER Sentiment Analysis"}
        },
        {
            "content": "Text preprocessing for social media typically involves: removing URLs, converting to lowercase, removing stopwords, handling emojis (either removing or converting to text), and tokenization. For social media specifically, you might want to handle hashtags and mentions specially.",
            "metadata": {"url": "https://towardsdatascience.com/preprocessing-text-data-for-nlp-5bf8688b1988", "title": "Text Preprocessing for NLP"}
        },
        {
            "content": "TextBlob is another simple library that provides API for common NLP tasks including sentiment analysis. It returns polarity and subjectivity scores for text, where polarity ranges from -1 (negative) to 1 (positive).",
            "metadata": {"url": "https://textblob.readthedocs.io/en/dev/", "title": "TextBlob Documentation"}
        },
        {
            "content": "A more advanced approach to sentiment analysis involves fine-tuning transformer models like BERT, RoBERTa or DistilBERT using HuggingFace's transformers library. This requires more computational resources but provides state-of-the-art accuracy.",
            "metadata": {"url": "https://huggingface.co/transformers/", "title": "HuggingFace Transformers"}
        }
    ]
    print(f"\nCollected {len(research_chunks)} relevant research chunks")
    
    # Define the coding requirements
    requirements = {
        "language": "Python",
        "libraries_allowed": ["nltk", "textblob", "pandas", "numpy", "matplotlib", "seaborn"],
        "output_format": [
            {"file_name": "sentiment_analyzer.py", "description": "Main implementation of the sentiment analyzer"},
            {"file_name": "example_usage.py", "description": "Example showing how to use the analyzer"}
        ],
        "code_style": "Include comprehensive docstrings, type hints, and follow PEP 8",
        "complexity_level": "intermediate"
    }
    print("\nDefined code requirements including allowed libraries and output files")
    
    # Use an arbitrary project ID for standalone example
    project_id = 12345
    
    print("\nGenerating code with CoderAgent...")
    print("-" * 50)
    start_time = time.time()
    
    # Execute the coder agent
    # In a mock run, create example results to demonstrate the output
    if not hasattr(coder_agent, 'execute_implemented') or not coder_agent.execute_implemented:
        print("Note: Using mock data for demonstration purposes")
        code_result = {
            "status": "success",
            "files": [
                {
                    "name": "sentiment_analyzer.py",
                    "content": '''"""Sentiment Analysis Tool for Social Media Posts

This module provides tools for analyzing sentiment in social media text data,
using both rule-based and ML approaches.
"""
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# Download necessary NLTK data (uncomment first time)
# nltk.download('punkt')
# nltk.download('stopwords')

class SentimentAnalyzer:
    """A tool for analyzing sentiment in social media posts."""
    
    def __init__(self, use_vader=True):
        """Initialize the sentiment analyzer.
        
        Args:
            use_vader (bool): Whether to use VADER for analysis (if False, uses TextBlob)
        """
        self.use_vader = use_vader
        
        # Initialize VADER if requested
        if self.use_vader:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self.vader = SentimentIntensityAnalyzer()
            except ImportError:
                print("VADER not available. Install with: nltk.download('vader_lexicon')")
                self.use_vader = False
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Stopwords not available. Using empty set.")
            self.stop_words = set()
    
    def preprocess_text(self, text):
        """Preprocess text for sentiment analysis.
        
        Args:
            text (str): The text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove user mentions (Twitter-style @user)
        text = re.sub(r'@\w+', '', text)
        
        # Handle hashtags - keep the text without #
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze(self, text):
        """Analyze sentiment of the provided text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results including:
                - compound: Overall sentiment (-1 to 1, only if VADER is used)
                - polarity: Sentiment polarity (-1 to 1)
                - subjectivity: Subjectivity score (0 to 1, 0 being objective)
                - sentiment: Textual representation (negative, neutral, positive)
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Initialize results dictionary
        results = {}
        
        # Use VADER if available
        if self.use_vader:
            vader_scores = self.vader.polarity_scores(processed_text)
            results['compound'] = vader_scores['compound']
            
            # Determine sentiment category from compound score
            if vader_scores['compound'] >= 0.05:
                results['sentiment'] = 'positive'
            elif vader_scores['compound'] <= -0.05:
                results['sentiment'] = 'negative'
            else:
                results['sentiment'] = 'neutral'
        
        # Always include TextBlob analysis 
        blob = TextBlob(processed_text)
        results['polarity'] = blob.sentiment.polarity
        results['subjectivity'] = blob.sentiment.subjectivity
        
        # Set sentiment based on TextBlob if VADER not used
        if 'sentiment' not in results:
            if blob.sentiment.polarity > 0.05:
                results['sentiment'] = 'positive'
            elif blob.sentiment.polarity < -0.05:
                results['sentiment'] = 'negative'
            else:
                results['sentiment'] = 'neutral'
        
        return results
    
    def analyze_batch(self, texts):
        """Analyze sentiment for a list of texts.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of sentiment analysis results for each text
        """
        return [self.analyze(text) for text in texts]
    
    def analyze_df(self, df, text_column):
        """Analyze sentiment for texts in a DataFrame column.
        
        Args:
            df (DataFrame): Pandas DataFrame with text data
            text_column (str): Name of the column containing text to analyze
            
        Returns:
            DataFrame: Original DataFrame with added sentiment columns
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Get all sentiment results
        all_results = self.analyze_batch(result_df[text_column].tolist())
        
        # Add each sentiment measure as a new column
        if self.use_vader:
            result_df['sentiment_compound'] = [r.get('compound') for r in all_results]
        
        result_df['sentiment_polarity'] = [r.get('polarity') for r in all_results]
        result_df['sentiment_subjectivity'] = [r.get('subjectivity') for r in all_results]
        result_df['sentiment'] = [r.get('sentiment') for r in all_results]
        
        return result_df
'''
                },
                {
                    "name": "example_usage.py",
                    "content": '''"""Example usage of the Sentiment Analyzer

This script demonstrates how to use the SentimentAnalyzer class
for analyzing social media posts.
"""
import pandas as pd
from sentiment_analyzer import SentimentAnalyzer

def main():
    # Create a sentiment analyzer
    analyzer = SentimentAnalyzer(use_vader=True)
    
    # Example single text analysis
    print("\nAnalyzing a single post:")
    sample_post = "I absolutely love this new phone! The camera is amazing! #technology"
    result = analyzer.analyze(sample_post)
    print(f"Post: {sample_post}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Polarity: {result['polarity']:.2f}")
    print(f"Subjectivity: {result['subjectivity']:.2f}")
    if 'compound' in result:
        print(f"Compound (VADER): {result['compound']:.2f}")
    
    # Example batch analysis
    print("\nAnalyzing multiple posts:")
    sample_posts = [
        "This product is terrible, I'm returning it immediately! #disappointed",
        "The weather is cloudy today, might rain later.",
        "Just got promoted at work! So excited for this new opportunity! #blessed"
    ]
    
    results = analyzer.analyze_batch(sample_posts)
    for i, (post, result) in enumerate(zip(sample_posts, results)):
        print(f"\nPost {i+1}: {post}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Polarity: {result['polarity']:.2f}")
    
    # Example with DataFrame
    print("\nAnalyzing posts in a DataFrame:")
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'username': ['user1', 'user2', 'user3', 'user1'],
        'timestamp': ['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03'],
        'post': [
            "I can't believe how bad the service was at this restaurant.",
            "The new software update is amazing, so many great features!",
            "Just a regular day, nothing special happening.",
            "Really excited about the upcoming concert! #music"
        ]
    })
    
    # Analyze the posts
    result_df = analyzer.analyze_df(df, 'post')
    
    # Display results
    print(result_df[['username', 'post', 'sentiment', 'sentiment_polarity']].to_string())

if __name__ == "__main__":
    main()
'''
                }
            ],
            "explanation": """The generated code provides a robust sentiment analysis tool for social media posts with the following features:

1. **Main Sentiment Analyzer Class:**
   - Supports both VADER (from NLTK) and TextBlob for sentiment analysis
   - Includes comprehensive text preprocessing for social media content
   - Provides methods for analyzing individual posts, batches, and DataFrames

2. **Text Preprocessing:**
   - Handles common social media elements like URLs, hashtags, and @mentions
   - Removes special characters and normalizes text

3. **Sentiment Metrics:**
   - Provides compound scores (VADER), polarity and subjectivity (TextBlob)
   - Classifies sentiment as positive, negative, or neutral

4. **Example Usage:**
   - Demonstrates analyzing single posts, multiple posts, and DataFrame integration
   - Shows how to access and interpret different sentiment metrics

The code follows PEP 8 style guidelines and includes comprehensive docstrings. It's designed to be easy to use while providing detailed sentiment analysis capabilities specifically tuned for social media content."""
        }
    else:
        code_result = coder_agent.execute(
            query=query,
            plan=research_plan,
            chunked_results={"query": query, "chunks": research_chunks},
            project_id=project_id,
            requirements=requirements
        )
    
    duration = time.time() - start_time
    print(f"\nCode generation completed in {duration:.2f} seconds")
    
    # Display the results
    print("\n" + "=" * 80)
    print("GENERATED CODE FILES:")
    print("=" * 80)
    
    if code_result.get("status") == "success":
        files = code_result.get("files", [])
        print(f"\nGenerated {len(files)} files:")
        
        # Create a directory to save the generated files
        output_dir = os.path.join(os.getcwd(), "generated_code_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving files to: {output_dir}")
        
        # Print and save each generated file
        for file in files:
            file_name = file.get("name", "unknown.py")
            content = file.get("content", "")
            
            # Display file in console
            print(f"\nFILE: {file_name}")
            print("-" * 80)
            print(content[:1000] + "..." if len(content) > 1000 else content)  # Truncate long files
            
            # Save file to disk
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        # Display explanation if available
        if "explanation" in code_result:
            print("\n" + "=" * 80)
            print("CODE EXPLANATION:")
            print("=" * 80)
            print(code_result["explanation"])
    else:
        print(f"Error generating code: {code_result.get('error', 'Unknown error')}")
    
    # Example of refinement based on feedback
    print("\n" + "=" * 80)
    print("EXAMPLE OF CODE REFINEMENT:")
    print("=" * 80)
    
    # Simulate user feedback
    feedback = "The code looks good, but could you add more visualization options for the sentiment results? Also, it would be nice to have a function to analyze sentiment trends over time for a collection of posts."
    print(f"\nUser feedback: '{feedback}'")
    
    print("\nRefining code based on feedback...")
    refined_result = coder_agent.refine_code(
        code_result=code_result,
        feedback=feedback,
        project_id=project_id
    )
    
    # For demonstration, create mock refined code result
    if not hasattr(coder_agent, 'refine_code_implemented') or not coder_agent.refine_code_implemented:
        refined_result = {
            "status": "success",
            "files": [
                {
                    "name": "sentiment_analyzer.py",
                    "content": '''"""Enhanced Sentiment Analysis Tool for Social Media Posts

This module provides tools for analyzing sentiment in social media text data,
with visualization capabilities and temporal trend analysis.
"""
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Download necessary NLTK data (uncomment first time)
# nltk.download('punkt')
# nltk.download('stopwords')

class SentimentAnalyzer:
    """A tool for analyzing sentiment in social media posts with visualization."""
    
    def __init__(self, use_vader=True):
        """Initialize the sentiment analyzer.
        
        Args:
            use_vader (bool): Whether to use VADER for analysis (if False, uses TextBlob)
        """
        self.use_vader = use_vader
        
        # Initialize VADER if requested
        if self.use_vader:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self.vader = SentimentIntensityAnalyzer()
            except ImportError:
                print("VADER not available. Install with: nltk.download('vader_lexicon')")
                self.use_vader = False
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Stopwords not available. Using empty set.")
            self.stop_words = set()
    
    def preprocess_text(self, text):
        """Preprocess text for sentiment analysis.
        
        Args:
            text (str): The text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove user mentions (Twitter-style @user)
        text = re.sub(r'@\w+', '', text)
        
        # Handle hashtags - keep the text without #
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
      
    # ... [Additional methods would be shown here] ...
    
    def visualize_sentiment_distribution(self, results, title="Sentiment Distribution"):
        """Create a visualization of sentiment distribution.
        
        Args:
            results (list): List of sentiment analysis results
            title (str): Title for the visualization
        """
        # Extract sentiment categories
        sentiments = [r.get('sentiment') for r in results]
        
        # Count occurrences
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        
        # Add title and labels
        plt.title(title)
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        
        # Add count labels on bars
        for i, count in enumerate(sentiment_counts.values):
            plt.text(i, count + 0.1, str(count), ha='center')
        
        plt.tight_layout()
        return plt.gcf()
    
    def analyze_sentiment_over_time(self, df, text_column, time_column, freq='D'):
        """Analyze sentiment trends over time.
        
        Args:
            df (DataFrame): DataFrame containing the posts
            text_column (str): Name of column containing text
            time_column (str): Name of column containing timestamps
            freq (str): Frequency for resampling (D=daily, W=weekly, etc.)
            
        Returns:
            DataFrame: Aggregated sentiment data by time period
        """
        # Ensure the time column is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
            
        # Analyze sentiment if not already done
        if 'sentiment_polarity' not in df.columns:
            df = self.analyze_df(df, text_column)
        
        # Group by time and aggregate
        grouped = df.set_index(time_column)
        
        # Resample by the specified frequency
        sentiment_over_time = grouped.resample(freq)['sentiment_polarity'].agg(['mean', 'count'])
        sentiment_over_time = sentiment_over_time.rename(columns={'mean': 'avg_polarity', 'count': 'post_count'})
        
        return sentiment_over_time'''    
                }
            ]
        }
    else:
        refined_result = coder_agent.refine_code(
            code_result=code_result, 
            feedback=feedback,
            project_id=project_id
        )
    
    # Display refined code
    if refined_result.get("status") == "success":
        # Just show the first file as an example
        if refined_result.get("files"):
            first_file = refined_result["files"][0]
            print(f"\nRefined file: {first_file.get('name')}")
            print("-" * 80)
            content = first_file.get("content", "")
            print(content[:1000] + "..." if len(content) > 1000 else content)  # Truncate long files
    else:
        print(f"Error refining code: {refined_result.get('error', 'Unknown error')}")
    
    # Create a safe output directory variable
    output_dir = "No files generated" 
    if code_result.get("status") == "success" and code_result.get("files"):
        # Only set this if files were actually created
        output_dir = os.path.join(os.getcwd(), "generated_code_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    print("\n" + "=" * 80)
    print("CoderAgent example completed")
    if output_dir != "No files generated":
        print(f"Generated files saved to: {output_dir}")
    print("=" * 80)
