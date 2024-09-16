"""
Search module 

"""

# Standard library imports
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

# Local imports



# Load environment variables
load_dotenv()

# Azure Cognitive Search configuration
AI_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AI_SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
AI_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]

# Azure OpenAI configuration
AOAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize clients
search_client = SearchClient(AI_SEARCH_ENDPOINT, AI_SEARCH_INDEX, AzureKeyCredential(AI_SEARCH_KEY))

aoai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15"
)


import os
from dotenv import load_dotenv




# Azure Cosmos DB





def hybrid_search(query):
    """

    """
    
    query_vector = generate_embeddings(query)
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="searchVector")
    filter_value = "isSearchable eq true"

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=3,
        filter=filter_value
    )

    formatted_results = []
    for result in results:
        formatted_result = {
            "id": result["id"],
            "title": result["title"],
            "content": result["content"],
            "score": result["@search.score"]
        }
        formatted_results.append(formatted_result)

    return formatted_results


    


def generate_embeddings(text, model="text-embedding-ada-002"):
    """
    Generate embeddings for the given text using Azure OpenAI.

    Args:
        text (str): The text to generate embeddings for.
        model (str): The name of the embedding model to use.

    Returns:
        list: The generated embedding vector.
    """
    return aoai_client.embeddings.create(input=[text], model=model).data[0].embedding

if __name__ == "__main__":
    # Example usage
    hybrid_search("example query")