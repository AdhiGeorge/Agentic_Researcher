"""
Response Adapter for OpenAI API v1.78.0
Handles response extraction and processing for the OpenAI API
Provides utilities for safely extracting content from API responses
"""
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def ensure_context_variables(context_variables: Optional[Any] = None) -> Dict[str, Any]:
    """
    Ensure context_variables is a proper dictionary
    
    Args:
        context_variables: Input context variables or None
        
    Returns:
        Dict: Validated context variables
    """
    if context_variables is None:
        return {}
    
    # Handle src.swarm.types.Response objects
    if hasattr(context_variables, '__class__'):
        class_name = context_variables.__class__.__name__
        
        # Special handling for Response objects from swarm framework
        if class_name == 'Response':
            # Create a safe copy of attributes
            safe_dict = {}
            
            # Extract content if available
            if hasattr(context_variables, 'content') and context_variables.content:
                safe_dict["content"] = context_variables.content
                
            # Extract messages if available
            if hasattr(context_variables, 'messages') and context_variables.messages:
                safe_dict["messages"] = context_variables.messages
                
            # Extract role if available
            if hasattr(context_variables, 'role') and context_variables.role:
                safe_dict["role"] = context_variables.role
                
            # Extract choices if available
            if hasattr(context_variables, 'choices') and context_variables.choices:
                safe_dict["choices"] = context_variables.choices
            
            # If we have any data, return it
            if safe_dict:
                return safe_dict
                
            # As a last resort, try to convert all attributes
            try:
                return {k: v for k, v in vars(context_variables).items() if not k.startswith('_')}
            except Exception as e:
                logger.warning(f"Cannot convert Response object to dict: {e}, using empty dict")
                return {}
    
    # Handle dict-like objects
    if hasattr(context_variables, '__getitem__') and hasattr(context_variables, 'keys'):
        try:
            return dict(context_variables)
        except Exception as e:
            logger.warning(f"Failed to convert dict-like object: {e}")
            
    # Handle direct dictionary case
    if isinstance(context_variables, dict):
        return context_variables
        
    # Convert string to dict if it looks like JSON
    if isinstance(context_variables, str):
        try:
            if context_variables.strip().startswith('{') and context_variables.strip().endswith('}'): 
                return safe_parse_json(context_variables)
        except Exception as e:
            logger.warning(f"Failed to parse string as JSON: {e}")
    
    # Handle all other cases with warning
    logger.warning(f"Invalid context_variables type: {type(context_variables)}, using empty dict")
    return {}

def safe_parse_json(json_str: str) -> Dict[str, Any]:
    """
    Safely parse JSON string with error handling
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Dict: Parsed JSON or empty dict on failure
    """
    if not json_str or not isinstance(json_str, str):
        return {}
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return {}

def extract_openai_response_content(response: Any) -> str:
    """
    Extract content from OpenAI response safely across API versions
    
    Args:
        response: OpenAI API response object
        
    Returns:
        str: The extracted content or error message
    """
    if response is None:
        return ""
    
    try:
        # Log what we're processing for debugging
        if hasattr(response, '__class__'):
            logger.debug(f"Processing response of type {response.__class__.__name__}")
        
        # Handle swarm Response type first
        if hasattr(response, '__class__') and response.__class__.__name__ == 'Response':
            # Direct content attribute is highest priority
            if hasattr(response, 'content') and response.content:
                if isinstance(response.content, str):
                    return response.content
                else:
                    # Try to convert non-string content to string
                    try:
                        return str(response.content)
                    except:
                        pass
            
            # Try to get content from messages list
            if hasattr(response, 'messages') and response.messages:
                try:
                    # Try to extract content from messages
                    if isinstance(response.messages, list) and len(response.messages) > 0:
                        last_msg = response.messages[-1]
                        if isinstance(last_msg, dict) and 'content' in last_msg:
                            return last_msg['content']
                        elif hasattr(last_msg, 'content'):
                            return last_msg.content
                except Exception as e:
                    logger.debug(f"Failed to extract from messages: {e}")
            
            # If we have a choices attribute, try that next
            if hasattr(response, 'choices') and response.choices:
                try:
                    return extract_from_choices(response.choices)
                except Exception as e:
                    logger.debug(f"Failed to extract from choices: {e}")
            
            # Log that we're falling back to string conversion
            logger.info(f"Using fallback string conversion for Response object")
            return str(response)
        
        # Handle direct content attribute (any object type)
        if hasattr(response, 'content') and response.content is not None:
            if isinstance(response.content, str):
                return response.content
            else:
                return str(response.content)
        
        # Handle OpenAI API v1.x standard format
        if hasattr(response, 'choices') and response.choices:
            result = extract_from_choices(response.choices)
            if result:
                return result
        
        # Last resort - convert to string
        return str(response)
    
    except Exception as e:
        logger.error(f"Error extracting content from response: {e}")
        return ""


def extract_from_choices(choices) -> str:
    """
    Extract content from choices array in OpenAI responses
    
    Args:
        choices: The choices array from an OpenAI response
        
    Returns:
        str: The extracted content or empty string
    """
    # Empty check
    if not choices or len(choices) == 0:
        return ""
    
    # Get first choice
    choice = choices[0]
    
    # Check for message object (API v1.x chat completions)
    if hasattr(choice, 'message'):
        # Object with message attribute
        if hasattr(choice.message, 'content'):
            return choice.message.content or ""
        # Handle message as dict
        elif isinstance(choice.message, dict) and 'content' in choice.message:
            return choice.message['content'] or ""
    
    # Check for message dict (API v1.x chat completions as dict)
    if isinstance(choice, dict) and 'message' in choice:
        if isinstance(choice['message'], dict) and 'content' in choice['message']:
            return choice['message']['content'] or ""
    
    # Check for delta object (streaming API v1.x)
    if hasattr(choice, 'delta'):
        if hasattr(choice.delta, 'content'):
            return choice.delta.content or ""
        elif isinstance(choice.delta, dict) and 'content' in choice.delta:
            return choice.delta['content'] or ""
    
    # Check for delta dict (streaming API v1.x as dict)
    if isinstance(choice, dict) and 'delta' in choice:
        if isinstance(choice['delta'], dict) and 'content' in choice['delta']:
            return choice['delta']['content'] or ""
    
    # Check for text attribute (API v0.x completions)
    if hasattr(choice, 'text'):
        return choice.text or ""
    
    # Check for text key (API v0.x completions as dict)
    if isinstance(choice, dict) and 'text' in choice:
        return choice['text'] or ""
    
    # Handle finish_reason
    if hasattr(choice, 'finish_reason') or (isinstance(choice, dict) and 'finish_reason' in choice):
        # This could be a valid empty completion
        return ""
    
    # If we can't extract in any known way, convert the choice to string
    return str(choice)


def extract_response_content(response: Any) -> str:
    """
    Extract content from response objects (legacy function for compatibility)
    This function is needed by older components that expect this exact function name
    
    Args:
        response: API response object
        
    Returns:
        str: The extracted content or error message
    """
    return extract_openai_response_content(response)


def adapt_openai_response(response: Any, default_value: str = "") -> Dict[str, Any]:
    """
    Adapt OpenAI response to a consistent format
    
    Args:
        response: OpenAI API response object
        default_value: Default value to use if extraction fails
        
    Returns:
        Dict: Standardized response dictionary
    """
    if response is None:
        return {"content": default_value, "original_response": None, "messages": []}
    
    try:
        # Extract main content
        content = extract_openai_response_content(response)
        
        # Initialize result with content
        result = {
            "content": content or default_value,
            "original_response": response,
            "messages": []
        }
        
        # Check if this is a swarm Response object
        is_swarm_response = hasattr(response, '__class__') and response.__class__.__name__ == 'Response'
        
        # Extract messages
        messages = []
        
        # Path 1: Swarm Response object with messages attribute
        if is_swarm_response and hasattr(response, 'messages') and response.messages:
            if isinstance(response.messages, list):
                messages = response.messages
            else:
                # If messages is not a list, try to add it as a single message
                try:
                    messages = [response.messages]
                except:
                    logger.debug("Could not convert messages to list")
        
        # Path 2: OpenAI API response with choices
        elif hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                # Handle object-style choices
                if hasattr(choice, 'message'):
                    messages.append(choice.message)
                # Handle dict-style choices
                elif isinstance(choice, dict) and 'message' in choice:
                    messages.append(choice['message'])
                    
                # For completions API, create message from text
                elif hasattr(choice, 'text') and choice.text:
                    messages.append({"role": "assistant", "content": choice.text})
                elif isinstance(choice, dict) and 'text' in choice and choice['text']:
                    messages.append({"role": "assistant", "content": choice['text']})
        
        # Add synthetic message if we have content but no messages
        if content and not messages:
            messages.append({"role": "assistant", "content": content})
            
        # Add messages to result
        result["messages"] = messages
        
        # Add response_type for debugging
        if hasattr(response, '__class__'):
            result["response_type"] = response.__class__.__name__
        else:
            result["response_type"] = type(response).__name__
                
        return result
    except Exception as e:
        logger.error(f"Error adapting response: {e}")
        return {"content": default_value, "original_response": str(response), "messages": []}


# Example usage with real-world scenarios
if __name__ == "__main__":
    import os
    import sys
    import json
    from pprint import pprint
    
    # Configure more verbose logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("===== Response Adapter Example Usage =====")
    print("This example demonstrates various response")
    print("extraction and processing techniques for the OpenAI API")
    print("across different versions and response formats.")
    
    print("\nExample 1: Context Variable Processing")
    print("-" * 60)
    
    # Different context variable formats
    context_examples = [
        # Valid dictionary
        {"query": "what is quantum computing?", "results": ["result1", "result2"]},
        
        # None value
        None,
        
        # JSON string
        '{"query": "string as json", "count": 42}',
        
        # Invalid JSON string
        '{query: invalid json syntax}',
        
        # Custom object (simulating a Response object)
        type('Response', (), {
            '__class__': type('DummyClass', (), {'__name__': 'Response'}),
            'content': 'Response content',
            'messages': [{"role": "assistant", "content": "Object message"}],
            'role': 'assistant',
            'choices': None
        }),
    ]
    
    # Process each example
    for i, ctx in enumerate(context_examples, 1):
        print(f"\nContext Variable Example {i}:")
        print(f"Input type: {type(ctx)}")
        print(f"Input value: {ctx}")
        
        # Process the context variables
        result = ensure_context_variables(ctx)
        print("Processed result:")
        pprint(result)
    
    print("\nExample 2: JSON Parsing and Error Handling")
    print("-" * 60)
    
    # Different JSON examples
    json_examples = [
        # Valid simple JSON
        '{"name": "Quantum Research Paper", "citations": 42}',
        
        # Valid nested JSON
        '''{"research": {
            "title": "Advances in NLP",
            "keywords": ["transformers", "attention", "language models"],
            "metrics": {"accuracy": 0.92, "f1": 0.89}
        }}''',
        
        # Invalid JSON - syntax error
        '{name: "Missing quotes", count: 123}',
        
        # Invalid JSON - unclosed brackets
        '{"data": [1, 2, 3',
        
        # Empty string
        "",
        
        # Non-string
        42
    ]
    
    # Process each JSON example
    for i, json_str in enumerate(json_examples, 1):
        print(f"\nJSON Example {i}:")
        print(f"Input: {json_str}")
        
        # Parse the JSON
        result = safe_parse_json(json_str)
        print("Parsed result:")
        pprint(result)
    
    print("\nExample 3: OpenAI API Response Extraction")
    print("-" * 60)
    
    # Mock various OpenAI API response formats
    
    # Define mock classes to simulate response objects
    class MockMessage:
        def __init__(self, content, role="assistant"):
            self.content = content
            self.role = role
    
    class MockChoice:
        def __init__(self, message=None, text=None, finish_reason="stop", index=0):
            self.message = message
            self.text = text
            self.finish_reason = finish_reason
            self.index = index
    
    class MockUsage:
        def __init__(self, prompt_tokens=10, completion_tokens=20, total_tokens=30):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens
    
    # Different response formats to test
    
    # 1. OpenAI API v1.x Chat Completion Response (object-style)
    class ChatCompletionV1Response:
        def __init__(self, content=""):
            self.id = "chatcmpl-123456789"
            self.object = "chat.completion"
            self.created = 1677858242
            self.model = "gpt-4"
            self.choices = [MockChoice(message=MockMessage(content=content))]
            self.usage = MockUsage()
    
    # 2. OpenAI API v1.x Chat Completion Response (dict-style)
    chat_completion_dict = {
        "id": "chatcmpl-987654321",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a dictionary-style response from the chat API."
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    # 3. OpenAI API v0.28.0 Completion Response
    class CompletionV028Response:
        def __init__(self, text=""):
            self.id = "cmpl-123abc456def"
            self.object = "text_completion"
            self.created = 1677858242
            self.model = "text-davinci-003"
            self.choices = [MockChoice(text=text)]
            self.usage = MockUsage()
    
    # 4. Function calling response
    function_call_response = {
        "id": "chatcmpl-123function456",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
                    }
                },
                "finish_reason": "function_call",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 15, 
            "completion_tokens": 25,
            "total_tokens": 40
        }
    }
    
    # Create examples of different response objects
    response_examples = [
        ("OpenAI v1.x Chat Completion (object)", 
         ChatCompletionV1Response("This is an object-style response from the chat API.")),
        
        ("OpenAI v1.x Chat Completion (dict)", 
         chat_completion_dict),
        
        ("OpenAI v0.28.0 Completion", 
         CompletionV028Response("This is an old-style completion API response.")),
        
        ("Function Call Response", 
         function_call_response),
        
        ("Error Case - None", 
         None),
        
        ("Error Case - String", 
         "Not a real response object")
    ]
    
    # Test extraction with each response example
    for desc, response in response_examples:
        print(f"\n{desc}:")
        print(f"Input type: {type(response)}")
        
        # Extract content
        content = extract_openai_response_content(response)
        print(f"Extracted content: '{content}'")
        
        # Full adaptation
        adapted = adapt_openai_response(response)
        print("Adapted response:")
        # Don't print full details for cleaner output
        if "content" in adapted:
            print(f"  Content: '{adapted['content']}'")
        if "messages" in adapted:
            print(f"  Message count: {len(adapted['messages'])}")
            if adapted['messages']:
                print(f"  First message: {adapted['messages'][0]}")
        print(f"  Response type: {adapted.get('response_type', 'unknown')}")
    
    print("\nExample 4: Real-world Integration Scenarios")
    print("-" * 60)
    
    print("1. Response Processing in Research Pipeline")
    
    # Simulate a research pipeline that processes responses
    def simulate_research_pipeline(query, mock_api_response):
        print(f"\nProcessing query: '{query}'")
        
        # Step 1: Extract content from API response
        content = extract_openai_response_content(mock_api_response)
        print(f"Extracted content: '{content[:50]}...'" if len(content) > 50 else f"Extracted content: '{content}'")
        
        # Step 2: Parse any structured data in the response
        try:
            # Sometimes responses contain JSON data
            structured_data = safe_parse_json(content)
            if structured_data:
                print("Detected structured data:")
                if len(str(structured_data)) > 100:
                    print(f"  {str(structured_data)[:100]}...")
                else:
                    pprint(structured_data)
            else:
                print("No structured data detected in response")
        except Exception as e:
            print(f"Error parsing structured data: {e}")
        
        # Step 3: Adapt the response for downstream processing
        adapted = adapt_openai_response(mock_api_response)
        if adapted:
            print(f"Successfully adapted response for downstream processing")
            print(f"Response contains {len(adapted.get('messages', []))} messages")
        
        # Simulate downstream processing
        # In a real application, this would feed into other components
        return {
            "query": query,
            "extracted_content": content,
            "message_count": len(adapted.get('messages', [])),
            "success": bool(content),
        }
    
    # Test the pipeline with different responses
    test_queries = [
        ("What is quantum computing?", 
         ChatCompletionV1Response("Quantum computing uses quantum mechanics principles like superposition and entanglement to perform computations.")),
        
        ("Extract key entities from this text about climate change", 
         ChatCompletionV1Response(json.dumps({
             "entities": [
                 {"text": "climate change", "type": "TOPIC"},
                 {"text": "global warming", "type": "RELATED_TOPIC"},
                 {"text": "carbon emissions", "type": "CAUSE"}
             ]
         })))
    ]
    
    # Run the pipeline for each test case
    for query, response in test_queries:
        result = simulate_research_pipeline(query, response)
        print("Pipeline result summary:")
        print(f"  Query: '{result['query']}'")
        print(f"  Success: {result['success']}")
        print(f"  Message count: {result['message_count']}")
    
    print("\n2. Error Handling and Fallbacks")
    
    # Demonstrate error handling with problematic responses
    problematic_responses = [
        ("Empty response", None),
        ("Malformed response", {"broken": "structure", "no": "choices"}),
        ("String instead of object", "API Error: Rate limit exceeded")
    ]
    
    for desc, response in problematic_responses:
        print(f"\nHandling {desc}:")
        
        # Try to extract content with error handling
        try:
            content = extract_openai_response_content(response)
            print(f"Extracted content: '{content}'")
        except Exception as e:
            print(f"Extraction error (should not happen due to safe handling): {e}")
        
        # Use adapted response with default value
        adapted = adapt_openai_response(response, default_value="[FALLBACK CONTENT]")
        print(f"Adapted with fallback: '{adapted.get('content')}'")
    
    print("\n" + "=" * 80)
    print("Response Adapter examples completed!")
    print("This utility ensures consistent handling of API responses")
    print("across different OpenAI API versions and formats.")
    print("=" * 80)
