"""

langchain==0.3.0
langchain-core==0.3.1
langchain-openai==0.2.0
langchain-text-splitters==0.3.0

"""


from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool



load_dotenv()


# Azure OpenAI configuration
aoai_deployment = os.getenv("AOAI_DEPLOYMENT_NAME")
aoai_key = os.getenv("AOAI_API_KEY")
aoai_endpoint = os.getenv("AOAI_ENDPOINT")


# Initialize LangChain Azure Chat OpenAI
llm_aoai = AzureChatOpenAI(
    azure_deployment=aoai_deployment,
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=aoai_key,
    azure_endpoint=aoai_endpoint
)

def _extract_country_names_streaming(input_stream):
    """A function that operates on input streams."""
    country_names_so_far = set()

    for input in input_stream:
        if not isinstance(input, dict):
            continue

        if "countries" not in input:
            continue

        countries = input["countries"]

        if not isinstance(countries, list):
            continue

        for country in countries:
            name = country.get("name")
            if not name:
                continue
            if name not in country_names_so_far:
                yield name
                country_names_so_far.add(name)

@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

def basic_inference_example(llm):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "What color is the sky?"},
    ]
    response = llm.invoke(messages)
    print("Basic Inference Response:")
    print(response.content)
    print(response.usage_metadata)

def streaming_inference_example(llm):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "What color is the sky?"},
    ]
    print("\nStreaming Inference Response:")
    response = llm.stream(messages)
    for chunk in response:
        print(chunk.content, end="|", flush=True)
    print('\n')

def json_parsing_example(llm):
    chain = llm | JsonOutputParser() | _extract_country_names_streaming
    
    print("\nJSON Parsing and Streaming Example:")
    for text in chain.stream(
        "Output a list of the countries France, Spain, and Japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the keys `name` and `population`"
    ):
        print(text, end="|", flush=True)
    print('\n')

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

def tool_calling_example(llm):
    tools = [add, multiply]
    llm_with_tools = llm.bind_tools(tools)
    
    print("\nTool Calling Example:")
    query = "What is 3 * 12? Also, what is 114 + 49?"
    
    messages = [HumanMessage(query)]
    ai_msg = llm_with_tools.invoke(messages)
    
    print("Tool Calls:")
    print(ai_msg.tool_calls)
    
    messages.append(ai_msg)
    
    # Invoke the tools
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    
    # Have LLM answer the question with the tool outputs
    final_ai_msg = llm_with_tools.invoke(messages)
    print("\nFinal Answer:")
    print(final_ai_msg.content)

def run_examples():
    
    #Basic inference
    basic_inference_example(llm_aoai)

    #Streaming inference
    streaming_inference_example(llm_aoai)
    
    #Streaming & parsing JSON
    json_parsing_example(llm_aoai)

    #Tool calling
    tool_calling_example(llm_aoai)

if __name__ == "__main__":
    run_examples()