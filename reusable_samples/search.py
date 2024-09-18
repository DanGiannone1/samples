"""
Search module for hybrid search using Azure Cognitive Search and Azure OpenAI.

This module provides functionality for performing hybrid searches
combining traditional keyword search with vector search.

Requirements:
    azure-search-documents
    azure-identity
    langchain-openai
    openai
"""

import os
import logging
from typing import List, Dict, Any, Optional
from functools import wraps

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

# Local imports
from aoai import generate_embeddings_aoai  # Import from your AOAI module

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Cognitive Search configuration
AI_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AI_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AI_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

@error_handler
def get_search_client() -> SearchClient:

    if not all([AI_SEARCH_ENDPOINT, AI_SEARCH_KEY, AI_SEARCH_INDEX]):
        raise ValueError("Azure AI Search configuration is incomplete")
    
    logger.info("Initializing Azure AI Search client")
    return SearchClient(AI_SEARCH_ENDPOINT, AI_SEARCH_INDEX, AzureKeyCredential(AI_SEARCH_KEY))

@error_handler
def hybrid_search(query: str) -> List[Dict[str, Any]]:
    """
    Perform a hybrid search using both keyword and vector search capabilities.

    Parameters
    ----------
    query : str
        The search query.

    Returns
    -------
    List[Dict[str, Any]]
        A list of search results, each containing id, title, content, and score.

    Raises
    ------
    Exception
        If there's an error during the search process.
    """
    logger.info(f"Performing hybrid search for query: {query}")
    search_client = get_search_client()

    embedding_response = generate_embeddings_aoai(query)
    if embedding_response is None:
        logger.error("Failed to generate embeddings for the query")
        return []

    query_vector = embedding_response.data[0].embedding
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

    logger.info(f"Hybrid search returned {len(formatted_results)} results")
    return formatted_results

def run_examples():
    """Example usage of the search module."""
    try:
        query = "example query"
        logger.info(f"Performing hybrid search with query: {query}")
        results = hybrid_search(query)
        logger.info(f"Search returned {len(results)} results")
        for result in results:
            logger.info(f"Result: {result['title']} (Score: {result['score']})")
    except Exception as e:
        logger.error(f"An error occurred during the search: {str(e)}")

if __name__ == "__main__":
    run_examples()