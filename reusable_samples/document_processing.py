"""
This module handles document processing operations using Azure Blob Storage and Azure Document Intelligence.
It provides functionality to upload documents to Blob Storage and analyze them using Document Intelligence.

Requirements:
    azure-storage-blob==12.22.0
    azure-ai-documentintelligence==1.0.0b2
    Azure Document Intelligence API Version: 2024-7-31 preview
"""

import os
import logging
from typing import Union, Dict, Any, List
from functools import wraps
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import io

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Blob Storage Configuration
STORAGE_ACCOUNT_CONNECTION_STRING = os.environ.get("STORAGE_ACCOUNT_CONNECTION_STRING")
STORAGE_ACCOUNT_CONTAINER = os.environ.get("STORAGE_ACCOUNT_CONTAINER", "documents")
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME")

# Azure Document Intelligence Configuration
DOCUMENT_INTELLIGENCE_ENDPOINT = os.environ.get("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.environ.get("DOCUMENT_INTELLIGENCE_KEY")

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
def get_blob_service_client() -> BlobServiceClient:
    """
    Create and return an Azure Blob Service Client.

    Returns
    -------
    BlobServiceClient
        An instance of the Azure Blob Service Client.

    Raises
    ------
    ValueError
        If the Blob storage connection string is missing.
    """
    if not STORAGE_ACCOUNT_CONNECTION_STRING:
        raise ValueError("Blob storage connection string is missing")
    logger.info("Blob service client initialized")
    return BlobServiceClient.from_connection_string(STORAGE_ACCOUNT_CONNECTION_STRING)

@azure_error_handler
def get_document_intelligence_client() -> DocumentIntelligenceClient:
    """
    Create and return an Azure Document Intelligence Client.

    Returns
    -------
    DocumentIntelligenceClient
        An instance of the Azure Document Intelligence Client.

    Raises
    ------
    ValueError
        If the Document Intelligence configuration is missing.
    """
    if not DOCUMENT_INTELLIGENCE_ENDPOINT or not DOCUMENT_INTELLIGENCE_KEY:
        raise ValueError("Document Intelligence configuration is missing")
    logger.info("Document Intelligence client initialized")
    return DocumentIntelligenceClient(DOCUMENT_INTELLIGENCE_ENDPOINT, AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY))

@azure_error_handler
def upload_to_blob(file_content: Union[bytes, io.IOBase], filename: str, container_name: str = None) -> Dict[str, str]:
    """
    Upload a file to Azure Blob Storage.

    Parameters
    ----------
    file_content : Union[bytes, io.IOBase]
        The content of the file to upload. This can be either:
        - bytes: Raw file content (e.g., result of reading a file in binary mode)
        - io.IOBase: A file-like object (e.g., an open file handle)
    filename : str
        The name to give the file in Blob Storage.
    container_name : str, optional
        The name of the container to upload to. 
        If not provided, uses the default container from environment variables.

    Returns
    -------
    Dict[str, str]
        A dictionary containing:
        - 'message': A success message
        - 'blob_url': The URL of the uploaded blob

    Raises
    ------
    Exception
        If there's an error during the upload process.
    """
    blob_service_client = get_blob_service_client()
    container_name = container_name or STORAGE_ACCOUNT_CONTAINER
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(filename)
    
    if isinstance(file_content, io.IOBase):
        blob_client.upload_blob(file_content.read(), overwrite=True)
    else:
        blob_client.upload_blob(file_content, overwrite=True)
    
    logger.info(f"File {filename} uploaded successfully")
    return {"message": f"File {filename} uploaded successfully", "blob_url": blob_client.url}

@azure_error_handler
def analyze_document(filename: str) -> AnalyzeResult:
    """
    Analyze a document using Azure Document Intelligence.

    Parameters
    ----------
    filename : str
        The name of the file in Blob Storage to analyze.

    Returns
    -------
    AnalyzeResult
        The result of the document analysis.

    Raises
    ------
    Exception
        If there's an error during the analysis process.
    """
    document_intelligence_client = get_document_intelligence_client()
    blob_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{STORAGE_ACCOUNT_CONTAINER}/{filename}"

    logger.info(f"Analyzing document from blob storage: {blob_url}")
    analyze_request = {"urlSource": blob_url}
    poller = document_intelligence_client.begin_analyze_document("prebuilt-layout", analyze_request=analyze_request)
    result: AnalyzeResult = poller.result()
    logger.info("Successfully read the PDF from blob storage with doc intelligence and extracted text.")
    return result

def chunk_document(document_content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Chunk a document into smaller sections.
    This function is a placeholder and currently returns an empty list.
    It will be implemented in the future to split the document content into overlapping chunks.

    Parameters
    ----------
    document_content : str
        The content of the document to chunk.
    chunk_size : int, optional
        The size of each chunk in characters. Defaults to 1000.
    overlap : int, optional
        The number of characters to overlap between chunks. Defaults to 100.

    Returns
    -------
    List[str]
        An empty list (placeholder for future implementation).
    """
    logger.warning("Document chunking is not implemented yet.")
    return []  # Placeholder for future implementation

@azure_error_handler
def list_blobs_in_folder(folder_name: str, container_name: str = None) -> List[Any]:
    """
    List all blobs in a specific folder within a container.

    Parameters
    ----------
    folder_name : str
        The name of the folder to list blobs from.
    container_name : str, optional
        The name of the container to list blobs from.
        If not provided, uses the default container from environment variables.

    Returns
    -------
    List[Any]
        A list of blob objects in the specified folder.
    """
    blob_service_client = get_blob_service_client()
    container_name = container_name or STORAGE_ACCOUNT_CONTAINER
    container_client = blob_service_client.get_container_client(container_name)
    
    return [blob for blob in container_client.list_blobs() if blob.name.startswith(folder_name)]

@azure_error_handler
def move_blob(source_blob_name: str, 
              destination_blob_name: str, 
              source_container_name: str = None, 
              destination_container_name: str = None) -> None:
    """
    Move a blob from one location to another within the same storage account.

    Parameters
    ----------
    source_blob_name : str
        The name of the source blob.
    destination_blob_name : str
        The name of the destination blob.
    source_container_name : str, optional
        The name of the source container.
        If not provided, uses the default container from environment variables.
    destination_container_name : str, optional
        The name of the destination container.
        If not provided, uses the same as the source container.
    """
    blob_service_client = get_blob_service_client()
    source_container_name = source_container_name or STORAGE_ACCOUNT_CONTAINER
    destination_container_name = destination_container_name or source_container_name

    source_container_client = blob_service_client.get_container_client(source_container_name)
    destination_container_client = blob_service_client.get_container_client(destination_container_name)

    source_blob = source_container_client.get_blob_client(source_blob_name)
    destination_blob = destination_container_client.get_blob_client(destination_blob_name)
    
    destination_blob.start_copy_from_url(source_blob.url)
    source_blob.delete_blob()
    logger.info(f"Moved blob from {source_blob_name} to {destination_blob_name}")



def run_examples():
    """Example usage of the document processing functions."""
    sample_file_path = 'path/to/your/sample/document.pdf'
    
    # Local file upload scenario
    logger.info("Uploading local file...")
    with open(sample_file_path, 'rb') as file:
        upload_result = upload_to_blob(file, os.path.basename(sample_file_path))
    logger.info(f"Local file upload: {upload_result['message']}")
    
    # Bytestream upload scenario
    logger.info("Uploading file as bytestream...")
    with open(sample_file_path, 'rb') as file:
        file_content = file.read()
    upload_result = upload_to_blob(file_content, "bytestream_" + os.path.basename(sample_file_path))
    logger.info(f"Bytestream upload: {upload_result['message']}")
    
    # Analyze the uploaded document
    logger.info("Analyzing uploaded document...")
    analysis_result = analyze_document("bytestream_" + os.path.basename(sample_file_path))
    
    # Extract and display a sample of the full text from the result
    full_text = analysis_result.content
    logger.info(f"Extracted full text length: {len(full_text)} characters")
    logger.info("Sample of extracted text:")
    logger.info(full_text[:500] + "..." if len(full_text) > 500 else full_text)

    # Example of listing blobs in a folder
    logger.info("Listing blobs in 'source' folder...")
    blobs = list_blobs_in_folder("source/")
    logger.info(f"Found {len(blobs)} blobs in the 'source' folder")

    # Example of moving a blob
    if blobs:
        source_blob_name = blobs[0].name
        destination_blob_name = source_blob_name.replace("source/", "processed/")
        logger.info(f"Moving blob {source_blob_name} to {destination_blob_name}...")
        move_blob(source_blob_name, destination_blob_name)
        logger.info("Blob moved successfully")

if __name__ == "__main__":
    run_examples()