"""
### ai_search.py ###

This module handles Azure Cognitive Search operations, including index creation, document uploading,
and hybrid search capabilities. It provides functionality to create a search index, upload and embed
documents, and perform hybrid searches combining keyword and vector search.

Requirements:
    azure-search-documents==11.4.0
    aoai.py -> this module uses embeddings and inference from the aoai.py sample in the same repo. You can swap it out with your own module if you prefer.
    
"""

import os
from typing import List, Dict, Any, Optional, Union, Iterable
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
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
import json
import time

from azure.identity import DefaultAzureCredential

# Import the generate_embeddings_aoai function from aoai.py
from aoai import generate_embeddings_aoai

class AISearchManager:
    def __init__(self, search_endpoint=None, search_index_name=None):
        """
        Initialize the AISearchManager with Azure Cognitive Search configuration.
        """
        load_dotenv()  # Load environment variables
        self._load_env_variables(search_endpoint, search_index_name)
        self.search_index_client = self._get_search_index_client()
        self.search_client = self._get_search_client()

    def _load_env_variables(self, search_endpoint=None, search_index_name=None):
        """
        Load environment variables required for Azure Cognitive Search operations.
        """
        self.search_endpoint = search_endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT")
        self.search_index_name = search_index_name or os.environ.get("AZURE_SEARCH_INDEX")
        self.search_key = os.environ.get("AZURE_SEARCH_KEY")
        self.tenant_id = os.environ.get("TENANT_ID", '16b3c013-d300-468d-ac64-7eda0820b6d3')

        if not all([self.search_endpoint, self.search_index_name]):
            raise ValueError("Azure Cognitive Search configuration is incomplete")

    def _get_credential(self):
        """
        Get the appropriate credential for Azure Cognitive Search.
        Prioritize key-based authentication, fall back to DefaultAzureCredential if no key is available.
        """
        if self.search_key:
            print("Using key-based authentication for Azure Cognitive Search")
            return AzureKeyCredential(self.search_key)
        else:
            print("Using DefaultAzureCredential for Azure Cognitive Search authentication")
            return DefaultAzureCredential(
                interactive_browser_tenant_id=self.tenant_id,
                visual_studio_code_tenant_id=self.tenant_id,
                workload_identity_tenant_id=self.tenant_id,
                shared_cache_tenant_id=self.tenant_id
            )

    def _get_search_index_client(self) -> SearchIndexClient:
        """
        Get the Search Index Client using the appropriate credential.

        Returns:
            SearchIndexClient: The initialized Search Index Client.
        """
        print("Initializing Search Index client")
        credential = self._get_credential()
        return SearchIndexClient(self.search_endpoint, credential)

    def _get_search_client(self) -> SearchClient:
        """
        Get the Search Client using the appropriate credential.

        Returns:
            SearchClient: The initialized Search Client.
        """
        print("Initializing Search client")
        credential = self._get_credential()
        return SearchClient(self.search_endpoint, self.search_index_name, credential)



    def create_search_index(self) -> bool:
        """
        Create the search index if it doesn't exist.

        Returns:
            bool: True if the index was created or already exists, False if there was an error.
        """
        try:
            # Check if index exists
            self.search_index_client.get_index(self.search_index_name)
            print(f"Index '{self.search_index_name}' already exists")
            return True
        except Exception:
            print(f"Creating new index '{self.search_index_name}'")

        try:
            # Define the index fields
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
                SearchField(name="content_vector2", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
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
            index = SearchIndex(name=self.search_index_name, fields=fields, vector_search=vector_search)
            self.search_index_client.create_or_update_index(index)
            print("Index has been created")
            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            return False

    def upload_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Upload multiple documents to the search index.

        Args:
            documents (List[Dict[str, Any]]): The list of documents to be uploaded to the index.

        Returns:
            bool: True if all documents were uploaded successfully, False otherwise.
        """
        try:
            result = self.search_client.upload_documents(documents=documents)
            print(f"Uploaded {len(result)} documents")
            return True
        except Exception as e:
            print(f"Error uploading documents: {e}")
            return False


    def simple_hybrid_search(self, query: str, top=3):
        """
        Perform a simple hybrid search using both keyword and vector search capabilities.

        Args:
            query (str): The search query.
            top (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: Search results.
        """
        print(f"Performing simple hybrid search for query: {query}")

        query_vector = generate_embeddings_aoai(query)
        if not query_vector:
            print("Failed to generate embedding for the query")
            return []

        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top, fields="content_vector")
        
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["id", "content"],
            top=top
        )

        return list(results)

    def simple_text_search(self, query: str, top=3):
        """
        Perform a simple text search using keyword search capabilities.

        Args:
            query (str): The search query.
            top (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: Search results.
        """
        print(f"Performing simple text search for query: {query}")

        results = self.search_client.search(
            search_text=query,
            select=["id", "content"],
            top=top
        )

        return list(results)

    def simple_vector_search(self, query: str, top=3):
        """
        Perform a simple vector search using vector search capabilities.

        Args:
            query (str): The search query.
            top (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: Search results.
        """
        print(f"Performing simple vector search for query: {query}")

        query_vector = generate_embeddings_aoai(query)
        if not query_vector:
            print("Failed to generate embedding for the query")
            return []
        
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top, fields="content_vector")
        
        results = self.search_client.search(
            vector_queries=[vector_query],
            select=["id", "content"],
            top=top
        )

        return list(results)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete multiple documents from the search index.

        Args:
            document_ids (List[str]): A list of document IDs to delete.
            
        Returns:
        bool: True if the documents were deleted successfully, False otherwise.
    """
        try:
            result = self.search_client.delete_documents(documents=[{"id": doc_id} for doc_id in document_ids])
            print(f"Deleted {len(result)} documents")
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def hybrid_search_simple(self, query: str, top=3):
        """
        Perform a hybrid search using both keyword and vector search capabilities.

        Args:
            query (str): The search query.

        Returns:
            The raw search results from Azure Cognitive Search.
        """
        print(f"Performing hybrid search for query: {query}")

        # Generate embedding for the query
        query_vector = generate_embeddings_aoai(query)
        if not query_vector:
            print("Failed to generate embedding for the query")
            return []
        
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="content_vector")
        
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["id", "content"],
            top=3
        )

        return results



    def dynamic_search(self, query: str, config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """
        Perform a dynamic search based on the provided configuration.
        The type of search is inferred from the presence of 'text_fields' and 'vector_fields' in the config.

        Args:
            query (str): The search query.
            config (Dict[str, Any]): A dictionary containing search parameters from the config file.

        Returns:
            Iterable[Dict[str, Any]]: Search results as an iterator.
        """
        print(f"Performing dynamic search for query: '{query}'")

        # Explicitly construct search parameters
        search_params = {
            "top": config.get("top", 3),  # Default to 10 if not specified
            "select": config.get("select", ["*"]),  # Default to all fields if not specified
        }

        text_fields = config.get('text_fields')
        vector_fields = config.get('vector_fields')

        # Handle text search
        if text_fields:
            search_params['search_text'] = query
            search_params['search_fields'] = text_fields

        # Handle vector search
        if vector_fields:
            query_vector = generate_embeddings_aoai(query)
            if not query_vector:
                print("Failed to generate embedding for the query")
                return []


            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=config.get('k_nearest_neighbors', 3),
                fields='content_vector'
            )
            search_params['vector_queries'] = [vector_query]

        # Determine and print the type of search being performed
        if text_fields and vector_fields:
            print("Performing hybrid search")
        elif text_fields:
            print("Performing text search")
        elif vector_fields:
            print("Performing vector search")
        else:
            raise ValueError("Either 'text_fields' or 'vector_fields' must be specified in the config")

        # print the constructed search parameters

        print(f"Constructed search parameters: {search_params}")

        # Perform the search with the constructed parameters
        results = self.search_client.search(**search_params)

        return results

def run_ai_search_examples():
    """
    Comprehensive example demonstrating the usage of AISearchManager with simple search functions.
    """
    # Initialize AISearchManager
    ai_search = AISearchManager(search_index_name='test')

    # Create search index
    ai_search.create_search_index()
    time.sleep(1)  # Wait for index creation

    # Prepare sample documents with embeddings
    sample_documents = [
        {
            "id": "1",
            "content": "This is a sample document about artificial intelligence and machine learning."
        },
        {
            "id": "2",
            "content": "Natural language processing is a subfield of artificial intelligence."
        }
    ]

    # Generate embeddings for documents
    for doc in sample_documents:
        content_vector = generate_embeddings_aoai(doc['content'])
        if content_vector:
            doc['content_vector'] = content_vector
            doc['content_vector2'] = content_vector # Using the same embedding for demonstration
        else:
            print(f"Failed to generate embedding for document: {doc['id']}")

    # Upload documents
    for doc in sample_documents:
        ai_search.upload_documents(doc)

    time.sleep(2)  # Wait for document indexing

    # Perform simple searches
    query = "artificial intelligence"

    print("\nSimple Hybrid Search Results:")
    hybrid_results = ai_search.simple_hybrid_search(query)
    for result in hybrid_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

    print("\nSimple Text Search Results:")
    text_results = ai_search.simple_text_search(query)
    for result in text_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

    print("\nSimple Vector Search Results:")
    vector_results = ai_search.simple_vector_search(query)
    for result in vector_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

    

    # Delete the uploaded documents
    ai_search.delete_documents(["1", "2"])

    time.sleep(2)  # Wait before deletion

    print("\nDocuments deleted. Performing another search to confirm deletion:")
    final_results = ai_search.simple_text_search(query)
    if not final_results:
        print("No results found after deletion, as expected.")
    else:
        print("Unexpected results found after deletion:")
        for result in final_results:
            print(f"ID: {result['id']}, Content: {result['content']}")

if __name__ == "__main__":
    run_ai_search_examples()