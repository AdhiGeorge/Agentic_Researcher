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


# Example usage
if __name__ == "__main__":
    # Example 1: Ensuring context variables
    print("Example 1: Ensuring context variables")
    
    # Valid dictionary
    ctx1 = {"query": "test query", "results": [1, 2, 3]}
    print(f"Valid input: {ctx1}")
    print(f"Output: {ensure_context_variables(ctx1)}")
    
    # None input
    print(f"None input")
    print(f"Output: {ensure_context_variables(None)}")
    
    # Invalid input
    print(f"Invalid input (string)")
    print(f"Output: {ensure_context_variables('not a dict')}")
    
    # Example 2: Safe JSON parsing
    print("\nExample 2: Safe JSON parsing")
    
    # Valid JSON
    valid_json = '{"name": "test", "value": 123}'
    print(f"Valid JSON: {valid_json}")
    print(f"Parsed: {safe_parse_json(valid_json)}")
    
    # Invalid JSON
    invalid_json = '{name: test, broken json'
    print(f"Invalid JSON: {invalid_json}")
    print(f"Parsed: {safe_parse_json(invalid_json)}")
    
    # Example 3: Mock response extraction
    print("\nExample 3: Response extraction (mocked responses)")
    
    # Create a mock response structure similar to v1.x
    class MockMessage:
        def __init__(self, content):
            self.content = content
    
    class MockChoice:
        def __init__(self, message):
            self.message = message
    
    class MockResponseV1:
        def __init__(self, content):
            self.choices = [MockChoice(MockMessage(content))]
    
    # Create a mock response structure similar to v0.28.0
    class MockResponseV028:
        def __init__(self, text):
            self.choices = [{"text": text}]
    
    # Test with v1.x format
    v1_response = MockResponseV1("Hello from v1.x API")
    print(f"v1.x extraction: {extract_openai_response_content(v1_response)}")
    
    # Test with v0.28.0 format
    v028_response = MockResponseV028("Hello from v0.28.0 API")
    print(f"v0.28.0 extraction: {extract_openai_response_content(v028_response)}")
