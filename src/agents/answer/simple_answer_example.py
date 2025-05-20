"""Simple Answer Agent Example
Demonstrates how to use the AnswerAgent functionality without database dependencies
"""
import os
import sys

# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import config
from src.utils.openai_client import AzureOpenAIClient

class SimpleAnswerAgent:
    def __init__(self):
        print("Creating simplified AnswerAgent for testing (bypassing database initialization)")
        self.openai_client = AzureOpenAIClient()
        self.name = "answer"
        
        # Load prompt templates
        self.system_prompt = (
            "You are a research assistant tasked with synthesizing information from "
            "multiple sources into a coherent, well-structured answer. Your response "
            "should be comprehensive yet concise, focusing on the most relevant information. "
            "Organize the answer with clear sections, use bullet points where appropriate, "
            "and ensure all claims are supported by the research sources. Cite sources when "
            "presenting specific facts or quotes."
        )
        
        # For logging
        self.config = config
        
    def generate_answer(self, query, research_data):
        # Prepare the research data for the prompt
        formatted_research = ""
        for idx, source in enumerate(research_data):
            source_text = source.get("text", "").strip()
            source_title = source.get("title", f"Source {idx+1}")
            source_url = source.get("url", "Unknown source")
            
            if source_text:
                formatted_research += f"Source {idx+1}: {source_title} ({source_url})\n"
                formatted_research += f"{source_text}\n\n"
        
        # Prepare the prompt
        user_prompt = (
            f"Research Query: {query}\n\n"
            f"Research Data:\n{formatted_research}\n\n"
            "Based on the research data provided, synthesize a comprehensive "
            "answer to the query. Be factual, concise, and well-structured."
        )
        
        # Call OpenAI with the correct parameters format
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.openai_client.generate_completion(
            messages=messages,
            model=self.config.azure_openai_deployment,
            temperature=0.3,
            max_tokens=1500
        )
        
        # Extract and return answer - response is already the content string
        answer = response.strip()
        print(f"Generated answer (first 100 chars): {answer[:100]}...")
            
        return answer
        
    def process(self, input_data):
        query = input_data.get("query", "")
        research_data = input_data.get("research_data", [])
        
        answer = self.generate_answer(query, research_data)
        
        return {
            "answer": answer,
            "status": "success"
        }

# Example usage
if __name__ == "__main__":
    print("\n===== SimpleAnswerAgent Example =====\n")
    
    # Use the simplified AnswerAgent
    answer_agent = SimpleAnswerAgent()
    
    # Real example with research data
    query = "What is transformer architecture in machine learning and how does it work?"
    
    # This simulates data that would come from the researcher agent
    research_data = [
        {
            "title": "Attention Is All You Need",
            "url": "https://arxiv.org/abs/1706.03762",
            "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
        },
        {
            "title": "Understanding Transformer Architecture",
            "url": "https://towardsdatascience.com/transformer-architecture-explained",
            "text": "The transformer architecture consists of an encoder and decoder, each made up of multiple layers. Each layer contains two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The attention mechanism allows the model to weigh the importance of different words in the input sequence when making predictions, regardless of their position in the sequence. This is done through the query, key, and value vectors, which are used to compute attention scores between words. The multi-head attention allows the model to attend to information from different representation subspaces at different positions."
        },
        {
            "title": "How Self-Attention Works in Transformers",
            "url": "https://machinelearningmastery.com/self-attention-in-nlp",
            "text": "Self-attention, also known as intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. The self-attention mechanism calculates attention scores between each pair of words in the input sequence. These scores determine how much focus to place on other parts of the input sequence when encoding a specific word. The attention scores are computed using the dot product between the query and key vectors, scaled by the square root of their dimension, and then normalized using a softmax function to get a probability distribution."
        }
    ]
    
    print(f"\nProcessing query: '{query}'")
    print(f"Research data contains {len(research_data)} sources")
    
    # Process the query and get an answer
    result = answer_agent.process({
        "query": query,
        "research_data": research_data
    })
    
    # Display the result
    print("\n" + "-"*50)
    print("GENERATED ANSWER:")
    print("-"*50)
    print(result["answer"])
    print("-"*50)
    print(f"Status: {result['status']}")
