"""
This module handles Azure Cognitive Search index creation and document uploading.
It provides functionality to create a search index and upload JSON documents to it.

Requirements:
    azure-search-documents==11.4.0
"""

import os
import logging
from typing import List, Dict, Any
from functools import wraps
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Cognitive Search Configuration
SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX")

def azure_error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

@azure_error_handler
def get_search_index_client() -> SearchIndexClient:
    
    if not SEARCH_ENDPOINT or not SEARCH_KEY:
        raise ValueError("Azure Cognitive Search configuration is missing")
    logger.info("Search Index client initialized")
    return SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))

@azure_error_handler
def get_search_client() -> SearchClient:

    if not SEARCH_ENDPOINT or not SEARCH_KEY or not SEARCH_INDEX_NAME:
        raise ValueError("Azure Cognitive Search configuration is missing")
    logger.info("Search client initialized")
    return SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX_NAME, AzureKeyCredential(SEARCH_KEY))

@azure_error_handler
def create_search_index() -> None:
    """
    Create the search index if it doesn't exist.
    """
    search_index_client = get_search_index_client()

    try:
        # Check if index exists
        search_index_client.get_index(SEARCH_INDEX_NAME)
        logger.info("Index already exists")
        return
    except:
        logger.info("Creating new index")

    # Define the index fields
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="date", type=SearchFieldDataType.DateTimeOffset, filterable=True, facetable=True),
        SimpleField(name="jobTitle", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="experienceLevel", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="sourceFileName", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="searchVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
    ]

    # Define vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="myHnsw")
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ]
    )

    # Create the index
    index = SearchIndex(name=SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)
    search_index_client.create_or_update_index(index)
    logger.info("Index has been created")

@azure_error_handler
def upload_document_to_index(document: Dict[str, Any]) -> None:
    """
    Upload a single JSON document to the search index.

    Parameters
    ----------
    document : Dict[str, Any]
        The JSON document to be uploaded to the index.
    """
    search_client = get_search_client()
    result = search_client.upload_documents(documents=[document])
    logger.info(f"Document uploaded with status: {result[0].succeeded}")

def run_examples():
    """Example usage of the indexing functions."""
    # Create the search index
    create_search_index()
    
    # Example of uploading a document
    sample_document = {
        "id": "1",
        "date": "2024-09-18T12:00:00Z",
        "jobTitle": "Software Engineer",
        "experienceLevel": "Mid",
        "content": "This is a sample document content.",
        "sourceFileName": "sample.pdf",
        "searchVector": [0.1, 0.2, 0.3]  # This should be a 1536-dimensional vector in practice
    }
    upload_document_to_index(sample_document)

if __name__ == "__main__":
    run_examples()