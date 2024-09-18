"""
This module handles interactions with Azure OpenAI (AOAI) directly through the API without using orchestrators or frameworks.

Requirements:
    openai==1.45.1
    gpt4o model version: 2024-08-06 
    API version: 2024-08-01 preview
"""

import os
import logging
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AzureOpenAI
import openai
from openai.types import CreateEmbeddingResponse

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_VERSION = "2024-08-01-preview"

# Azure OpenAI configuration
aoai_deployment = os.environ.get("AOAI_DEPLOYMENT_NAME")
aoai_key = os.environ.get("AOAI_API_KEY")
aoai_endpoint = os.environ.get("AOAI_ENDPOINT")

logger.info("AOAI Endpoint: %s", aoai_endpoint)
logger.info("AOAI Deployment: %s", aoai_deployment)
logger.info("AOAI Key: %s", aoai_key[:5] + "*" * (len(aoai_key) - 5) if aoai_key else None)

# Initialize Azure OpenAI client
try:
    aoai_client = AzureOpenAI(
        azure_endpoint=aoai_endpoint,
        api_key=aoai_key,
        api_version=API_VERSION
    )
except Exception as e:
    logger.error("Failed to initialize Azure OpenAI client: %s", e)
    raise

def generate_embeddings_aoai(text: str, model: str = "text-embedding-ada-002") -> Optional[CreateEmbeddingResponse]:
    """
    Generate embeddings for the given text using Azure OpenAI.

    Parameters
    ----------
    text : str
        The text to generate embeddings for.
    model : str, optional
        The name of the embedding model to use (default is "text-embedding-ada-002").

    Returns
    -------
    Optional[CreateEmbeddingResponse]
        The generated embedding response or None if an error occurs.

    """
    try:
        response = aoai_client.embeddings.create(input=[text], model=model)
        logger.info("Embeddings generated successfully")
        return response
    except Exception as e:
        logger.error("Error generating embeddings: %s", e)
        return None

def inference_structured_output_aoai(messages: List[Dict[str, str]], deployment: str, schema: BaseModel) -> dict:
    """
    Perform inference on structured output using Azure OpenAI.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        The list of messages in the conversation.
    deployment : str
        The model deployment name.
    schema : pydantic.BaseModel
        The Pydantic schema for parsing the structured output.

    Returns
    -------
    Optional[Union[ChatCompletion, Dict]]
        The full response or parsed structured output, or None if an error occurs.

    """
    try:
        completion = aoai_client.beta.chat.completions.parse(
            model=deployment,
            messages=messages,
            response_format=schema,
        )
        logger.info("Structured output inference completed")
        logger.debug("Completion content: %s", completion.choices[0].message.content)
        logger.debug("Parsed event: %s", completion.choices[0].message.parsed)
        return completion
    except Exception as e:
        logger.error("Error in structured output inference: %s", e)
        return None

def tool_inference_aoai(messages: List[Dict[str, str]], deployment: str, tools: List[Dict]) -> dict:
    """
    Perform inference using tools in Azure OpenAI.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        The list of messages in the conversation.
    deployment : str
        The model deployment name.
    tools : List[Dict]
        The list of tools to use for inference.

    Returns
    -------
    Optional[ChatCompletion]
        The full response from the model, or None if an error occurs.

    """
    try:
        response = aoai_client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=tools
        )
        logger.info("Tool inference completed")
        logger.debug("Function Call: %s", response.choices[0].message.tool_calls[0].function)
        return response
    except Exception as e:
        logger.error("Error in tool inference: %s", e)
        return None

def stream_inference_aoai(messages: List[Dict[str, str]], deployment: str) -> str:
    """
    Stream a chat completion from Azure OpenAI.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        The list of messages in the conversation.
    deployment : str
        The model deployment name.

    Returns
    -------
    str
        The full response content.

    """
    try:
        response = aoai_client.chat.completions.create(
            model=deployment,
            messages=messages, 
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content is not None:
                    full_response += delta.content
                    print(delta.content, end='', flush=True)  # Print content as it arrives. Replace with yield in an actual application.
        print('\n\n')

        return full_response
    except Exception as e:
        logger.error("Error in streaming chat completion: %s", e)
        return ""

def run_examples():
    """Run example usage of the Azure OpenAI functions."""
    
    # Structured output example
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    messages = [
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ]

    result = inference_structured_output_aoai(messages, aoai_deployment, CalendarEvent)
    if result:
        new_event = CalendarEvent(**result.choices[0].message.parsed.dict())
        logger.info("Event name: %s", new_event.name)
        logger.info("Event date: %s", new_event.date)
        logger.info("Event participants: %s", new_event.participants)
    else:
        logger.warning("Failed to process structured output")

    # Tool call example
    class GetDeliveryDate(BaseModel):
        order_id: str

    tools = [openai.pydantic_function_tool(GetDeliveryDate)]

    messages = [
        {"role": "system", "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."},
        {"role": "user", "content": "Hi, can you tell me the delivery date for my order #12345?"}
    ]

    response = tool_inference_aoai(messages, aoai_deployment, tools)
    if response:
        function_call = response.choices[0].message.tool_calls[0].function
        logger.info("Function called: %s", function_call.name)
        logger.info("Arguments: %s", function_call.arguments)
        logger.info("Total tokens used: %s", response.usage.total_tokens)
    else:
        logger.warning("Failed to process tool inference")


    #Streaming example
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "What color is the sky?"},
    ]

    full_response = stream_inference_aoai(messages, aoai_deployment)
    logger.info("Streaming completed successfully")
    logger.info("Full response: %s", full_response)


if __name__ == "__main__":
    run_examples()