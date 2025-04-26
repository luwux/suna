"""
LLM API interface for making calls to various language models.

This module provides a unified interface for making API calls to different LLM providers
(OpenAI, Anthropic, Groq, etc.) using LiteLLM. It includes support for:
- Streaming responses
- Tool calls and function calling
- Retry logic with exponential backoff
- Model-specific configurations
- Comprehensive error handling and logging
"""

from typing import Union, Dict, Any, Optional, AsyncGenerator, List
import os
from dotenv import load_dotenv
load_dotenv()
import json
import asyncio
from openai import OpenAIError
import litellm
from utils.logger import logger
from utils.config import config
from datetime import datetime
import traceback

# litellm.set_verbose=True
litellm.modify_params=True

# Constants
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 30
RETRY_DELAY = 5

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class LLMRetryError(LLMError):
    """Exception raised when retries are exhausted."""
    pass

def setup_api_keys() -> None:
    """Set up API keys from environment variables."""
    providers = ["OPENAI", "ANTHROPIC", "GROQ", "OPENROUTER", "GEMINI"]
    for provider in providers:
        key = getattr(config, f'{provider}_API_KEY')
        if key:
            # Set the environment variable for LiteLLM to use
            os.environ[f"{provider}_API_KEY"] = key
            logger.debug(f"API key set for provider: {provider}")
        else:
            logger.warning(f"No API key found for provider: {provider}")
    
    # Set up OpenRouter API base if not already set
    if config.OPENROUTER_API_KEY and config.OPENROUTER_API_BASE:
        os.environ['OPENROUTER_API_BASE'] = config.OPENROUTER_API_BASE
        logger.debug(f"Set OPENROUTER_API_BASE to {config.OPENROUTER_API_BASE}")
    
    # Set up AWS Bedrock credentials
    aws_access_key = config.AWS_ACCESS_KEY_ID
    aws_secret_key = config.AWS_SECRET_ACCESS_KEY
    aws_region = config.AWS_REGION_NAME
    
    if aws_access_key and aws_secret_key and aws_region:
        logger.debug(f"AWS credentials set for Bedrock in region: {aws_region}")
        # Configure LiteLLM to use AWS credentials
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        os.environ['AWS_REGION_NAME'] = aws_region
    else:
        logger.warning(f"Missing AWS credentials for Bedrock integration - access_key: {bool(aws_access_key)}, secret_key: {bool(aws_secret_key)}, region: {aws_region}")

async def handle_error(error: Exception, attempt: int, max_attempts: int) -> None:
    """Handle API errors with appropriate delays and logging."""
    delay = RATE_LIMIT_DELAY if isinstance(error, litellm.exceptions.RateLimitError) else RETRY_DELAY
    logger.warning(f"Error on attempt {attempt + 1}/{max_attempts}: {str(error)}")
    logger.debug(f"Waiting {delay} seconds before retry...")
    await asyncio.sleep(delay)

def prepare_params(
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Dict[str, Any]:
    """Prepare parameters for the API call."""
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
        "top_p": top_p,
        "stream": stream,
    }

    if api_key:
        params["api_key"] = api_key
    if api_base:
        params["api_base"] = api_base
    if model_id:
        params["model_id"] = model_id

    # Handle token limits
    if max_tokens is not None:
        # For Claude 3.7 in Bedrock, do not set max_tokens or max_tokens_to_sample
        # as it causes errors with inference profiles
        if model_name.startswith("bedrock/") and "claude-3-7" in model_name:
            logger.debug(f"Skipping max_tokens for Claude 3.7 model: {model_name}")
            # Do not add any max_tokens parameter for Claude 3.7
        else:
            param_name = "max_completion_tokens" if 'o1' in model_name else "max_tokens"
            params[param_name] = max_tokens

    # Add tools if provided
    if tools:
        params.update({
            "tools": tools,
            "tool_choice": tool_choice
        })
        logger.debug(f"Added {len(tools)} tools to API parameters")

    # # Add Claude-specific headers
    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        params["extra_headers"] = {
            # "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"
            "anthropic-beta": "output-128k-2025-02-19"
        }
        logger.debug("Added Claude-specific headers")
    
    # Add OpenRouter-specific parameters
    if model_name.startswith("openrouter/"):
        logger.debug(f"Preparing OpenRouter parameters for model: {model_name}")
        
        # Add optional site URL and app name from config
        site_url = config.OR_SITE_URL
        app_name = config.OR_APP_NAME
        if site_url or app_name:
            extra_headers = params.get("extra_headers", {})
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if app_name:
                extra_headers["X-Title"] = app_name
            params["extra_headers"] = extra_headers
            logger.debug(f"Added OpenRouter site URL and app name to headers")
    
    # Add Bedrock-specific parameters
    if model_name.startswith("bedrock/"):
        logger.debug(f"Preparing AWS Bedrock parameters for model: {model_name}")
        
        if not model_id and "anthropic.claude-3-7-sonnet" in model_name:
            params["model_id"] = (
                "arn:aws:bedrock:us-west-2:935064898258:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            )
            logger.debug(
                f'Auto-set model_id for Claude 3.7 Sonnet: {params["model_id"]}'
            )

    # Apply context caching for Gemini models
    if "gemini" in model_name.lower():
        messages = params["messages"]  # Direct reference, modification affects params

        # Ensure messages is a list
        if not isinstance(messages, list):
            return params  # Return early if messages format is unexpected

        # 只对第一条系统消息应用缓存控制
        cache_control_applied = 0
        if messages and messages[0].get("role") == "system":
            message = messages[0]
            content = message.get("content")

            if isinstance(content, str):
                # Convert string content to structured content with cache_control
                message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                cache_control_applied = 1
            elif isinstance(content, list):
                # If content is already a list, add cache_control to text items
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        if "cache_control" not in item:
                            item["cache_control"] = {"type": "ephemeral"}
                            cache_control_applied += 1

        logger.info(
            f"Gemini context caching applied to {cache_control_applied} message parts (only first system message)"
        )

    # Apply Anthropic prompt caching (minimal implementation)
    # Check model name *after* potential modifications (like adding bedrock/ prefix)
    effective_model_name = params.get("model", model_name) # Use model from params if set, else original
    if "claude" in effective_model_name.lower() or "anthropic" in effective_model_name.lower():
        messages = params["messages"] # Direct reference, modification affects params

        # 只对第一条系统消息应用缓存控制
        cache_control_applied = 0
        if messages and messages[0].get("role") == "system":
            content = messages[0].get("content")
            if isinstance(content, str):
                # Wrap the string content in the required list structure
                messages[0]["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                cache_control_applied = 1
            elif isinstance(content, list):
                # If content is already a list, check if text blocks need cache_control
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        if "cache_control" not in item:
                            item["cache_control"] = {"type": "ephemeral"}
                            cache_control_applied += 1

        if cache_control_applied > 0:
            logger.info(f"Claude context caching applied to system message")

    # Add reasoning_effort for Anthropic models if enabled
    use_thinking = enable_thinking if enable_thinking is not None else False
    is_anthropic = (
        "anthropic" in effective_model_name.lower()
        or "claude" in effective_model_name.lower()
    )
    is_gemini = "gemini" in effective_model_name.lower()

    if is_anthropic and use_thinking:
        effort_level = reasoning_effort if reasoning_effort else 'low'
        params["reasoning_effort"] = effort_level
        params["temperature"] = (
            1.0  # Required by Anthropic when reasoning_effort is used
        )
        logger.info(
            f"Anthropic thinking enabled with reasoning_effort='{effort_level}'"
        )
    elif is_gemini and use_thinking:
        # Map reasoning_effort to Gemini's thinking parameter
        effort_level = reasoning_effort if reasoning_effort else "low"
        budget_tokens = 1024  # default for 'low'

        if effort_level == "medium":
            budget_tokens = 2048
        elif effort_level == "high":
            budget_tokens = 4096

        # Add thinking parameter to Gemini request
        params["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        logger.info(
            f"Gemini thinking enabled with budget_tokens={budget_tokens} (from effort='{effort_level}')"
        )

    return params

async def make_llm_api_call(
    messages: List[Dict[str, Any]],
    model_name: str,
    response_format: Optional[Any] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = True,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Union[Dict[str, Any], AsyncGenerator]:
    """
    Make an API call to a language model using LiteLLM.
    
    Args:
        messages: List of message dictionaries for the conversation
        model_name: Name of the model to use (e.g., "gpt-4", "claude-3", "openrouter/openai/gpt-4", "bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
        response_format: Desired format for the response
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in the response
        tools: List of tool definitions for function calling
        tool_choice: How to select tools ("auto" or "none")
        api_key: Override default API key
        api_base: Override default API base URL
        stream: Whether to stream the response
        top_p: Top-p sampling parameter
        model_id: Optional ARN for Bedrock inference profiles
        enable_thinking: Whether to enable thinking
        reasoning_effort: Level of reasoning effort
        
    Returns:
        Union[Dict[str, Any], AsyncGenerator]: API response or stream
        
    Raises:
        LLMRetryError: If API call fails after retries
        LLMError: For other API-related errors
    """
    # debug <timestamp>.json messages
    effective_model = config.MODEL_TO_USE if model_name is None else model_name
    logger.info(
        f"Making LLM API call to model: {effective_model} (Thinking: {enable_thinking}, Effort: {reasoning_effort})"
    )
    params = prepare_params(
        messages=messages,
        model_name=effective_model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        api_key=api_key,  # Allow API key override
        api_base=api_base,
        stream=stream,
        top_p=top_p,
        model_id=model_id,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES}")
            # logger.debug(f"API request parameters: {json.dumps(params, indent=2)}")
            
            response = await litellm.acompletion(**params)
            logger.debug(f"Successfully received API response from {model_name}")
            logger.debug(f"Response: {response}")
            return response
            
        except (litellm.exceptions.RateLimitError, OpenAIError, json.JSONDecodeError) as e:
            last_error = e
            await handle_error(e, attempt, MAX_RETRIES)
            
        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}", exc_info=True)
            raise LLMError(f"API call failed: {str(e)}")
    
    error_msg = f"Failed to make API call after {MAX_RETRIES} attempts"
    if last_error:
        error_msg += f". Last error: {str(last_error)}"
    logger.error(error_msg, exc_info=True)
    raise LLMRetryError(error_msg)

# Initialize API keys on module import
setup_api_keys()

# Test code for OpenRouter integration
async def test_openrouter():
    """Test the OpenRouter integration with a simple query."""
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]
    
    try:
        # Test with standard OpenRouter model
        print("\n--- Testing standard OpenRouter model ---")
        response = await make_llm_api_call(
            model_name="openrouter/openai/gpt-4o-mini",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        
        # Test with deepseek model
        print("\n--- Testing deepseek model ---")
        response = await make_llm_api_call(
            model_name="openrouter/deepseek/deepseek-r1-distill-llama-70b",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        
        # Test with Mistral model
        print("\n--- Testing Mistral model ---")
        response = await make_llm_api_call(
            model_name="openrouter/mistralai/mixtral-8x7b-instruct",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        
        return True
    except Exception as e:
        print(f"Error testing OpenRouter: {str(e)}")
        return False

async def test_bedrock():
    """Test the AWS Bedrock integration with a simple query."""
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]
    
    try:    
        response = await make_llm_api_call(
            model_name="bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
            model_id="arn:aws:bedrock:us-west-2:935064898258:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=test_messages,
            temperature=0.7,
            # Claude 3.7 has issues with max_tokens, so omit it
            # max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        
        return True
    except Exception as e:
        print(f"Error testing Bedrock: {str(e)}")
        return False


async def test_gemini():
    """Test the Gemini integration with context caching."""
    # Create a large system message to satisfy the caching requirement (min 4096 tokens)
    large_context = (
        "You are a helpful AI assistant that specializes in providing concise responses. "
        * 600
    )

    test_messages = [
        {"role": "system", "content": large_context},
        {
            "role": "user",
            "content": "Hello, can you write a short paragraph about Paris? This is just a test of the context caching feature.",
        },
    ]

    try:
        # First call - should be uncached
        logger.info("Making first call to Gemini (uncached)")
        logger.info(f"System message length: {len(large_context)} characters")
        response1 = await make_llm_api_call(
            model_name="gemini/gemini-2.5-pro-latest",
            messages=test_messages,
            temperature=0.7,
            enable_thinking=True,
            reasoning_effort="low",
        )
        print(f"First response: {response1.choices[0].message.content}")
        print(f"First usage: {response1.usage}")

        # Second call - should use cache
        logger.info("Making second call to Gemini (should use cache)")
        response2 = await make_llm_api_call(
            model_name="gemini/gemini-2.5-pro-latest",
            messages=test_messages,
            temperature=0.7,
            enable_thinking=True,
            reasoning_effort="low",
        )
        print(f"Second response: {response2.choices[0].message.content}")
        print(f"Second usage: {response2.usage}")

        # Compare token usage
        if response1.usage.prompt_tokens > response2.usage.prompt_tokens:
            print(
                f"✅ Context caching worked! First call used {response1.usage.prompt_tokens} prompt tokens, second call used {response2.usage.prompt_tokens} prompt tokens"
            )
        else:
            print(
                f"❓ Context caching might not be working as expected. First: {response1.usage.prompt_tokens} tokens, Second: {response2.usage.prompt_tokens} tokens"
            )

        return True
    except Exception as e:
        print(f"Error testing Gemini: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio

    test_to_run = "gemini"  # Change this to run a different test

    if test_to_run == "bedrock":
        test_success = asyncio.run(test_bedrock())
        result_name = "Bedrock"
    elif test_to_run == "openrouter":
        test_success = asyncio.run(test_openrouter())
        result_name = "OpenRouter"
    elif test_to_run == "gemini":
        test_success = asyncio.run(test_gemini())
        result_name = "Gemini"
    else:
        print(f"Unknown test: {test_to_run}")
        test_success = False
        result_name = test_to_run

    if test_success:
        print(f"\n✅ {result_name} integration test completed successfully!")
    else:
        print(f"\n❌ {result_name} integration test failed!")
