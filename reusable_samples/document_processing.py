"""
This module handles document processing operations using Azure Blob Storage and Azure Document Intelligence.
It provides functionality to upload documents to Blob Storage and analyze them using Document Intelligence.

***Requirements***

azure-storage-blob==12.22.0
azure-ai-documentintelligence==1.0.0b2
Azure Document Intelligence API Version: 2024-7-31 preview

"""
import os
from typing import Union, Dict, Any
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import io

load_dotenv()

# Azure Blob Storage Configuration
STORAGE_ACCOUNT_CONNECTION_STRING = os.getenv("STORAGE_ACCOUNT_CONNECTION_STRING")
STORAGE_ACCOUNT_CONTAINER = os.getenv("STORAGE_ACCOUNT_CONTAINER", "documents")
STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")

# Azure Document Intelligence Configuration
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")

def get_blob_service_client() -> BlobServiceClient:
    """
    Create and return an Azure Blob Service Client.
    Returns:
        BlobServiceClient: An instance of the Azure Blob Service Client.
    Raises:
        ValueError: If the Blob storage connection string is missing.
    """
    if not STORAGE_ACCOUNT_CONNECTION_STRING:
        raise ValueError("Blob storage connection string is missing")
    print("Blob service client initialized")
    return BlobServiceClient.from_connection_string(STORAGE_ACCOUNT_CONNECTION_STRING)

def get_document_intelligence_client() -> DocumentIntelligenceClient:
    """
    Create and return an Azure Document Intelligence Client.
    Returns:
        DocumentIntelligenceClient: An instance of the Azure Document Intelligence Client.
    Raises:
        ValueError: If the Document Intelligence configuration is missing.
    """
    if not DOCUMENT_INTELLIGENCE_ENDPOINT or not DOCUMENT_INTELLIGENCE_KEY:
        raise ValueError("Document Intelligence configuration is missing")
    print("Document Intelligence client initialized")
    return DocumentIntelligenceClient(DOCUMENT_INTELLIGENCE_ENDPOINT, AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY))

def upload_to_blob(file_content: Union[bytes, io.IOBase], filename: str, container_name: str = None) -> Dict[str, str]:
    """
    Upload a file to Azure Blob Storage.

    This function can handle two types of file content:
    1. Bytes: Raw file content as a bytes object.
    2. File-like object: An open file or file-like object.

    The Union[bytes, io.IOBase] type hint indicates that file_content can be either of these types.

    Args:
        file_content (Union[bytes, io.IOBase]): The content of the file to upload. This can be either:
            - bytes: Raw file content (e.g., result of reading a file in binary mode)
            - io.IOBase: A file-like object (e.g., an open file handle)
        filename (str): The name to give the file in Blob Storage.
        container_name (str, optional): The name of the container to upload to. 
                                        If not provided, uses the default container from environment variables.
    Returns:
        Dict[str, str]: A dictionary containing:
            - 'message': A success message
            - 'blob_url': The URL of the uploaded blob
    Raises:
        Exception: If there's an error during the upload process.
    """

    blob_service_client = get_blob_service_client()
    container_name = container_name or STORAGE_ACCOUNT_CONTAINER
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(filename)
    try:
        if isinstance(file_content, io.IOBase):
            blob_client.upload_blob(file_content.read(), overwrite=True)
        else:
            blob_client.upload_blob(file_content, overwrite=True)
        return {"message": f"File {filename} uploaded successfully", "blob_url": blob_client.url}
    except Exception as e:
        print(f"An error occurred while uploading the file: {str(e)}")
        raise

def analyze_document(filename: str) -> AnalyzeResult:
    """
    Analyze a document using Azure Document Intelligence.
    Args:
        filename (str): The name of the file in Blob Storage to analyze.
    Returns:
        AnalyzeResult: The result of the document analysis.
    Raises:
        Exception: If there's an error during the analysis process.
    """
    document_intelligence_client = get_document_intelligence_client()
    document_intelligence_client = DocumentIntelligenceClient(
        DOCUMENT_INTELLIGENCE_ENDPOINT, AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
    )
    blob_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{STORAGE_ACCOUNT_CONTAINER}/{filename}"

    print(f"Analyzing document from blob storage: {blob_url}")
    try:
        analyze_request = {"urlSource": blob_url}
        poller = document_intelligence_client.begin_analyze_document("prebuilt-layout", analyze_request=analyze_request)
        result: AnalyzeResult = poller.result()
        print("Successfully read the PDF from blob storage with doc intelligence and extracted text.")
        return result
    except Exception as e:
        print(f"An error occurred while analyzing the document: {str(e)}")
        raise

def chunk_document(document_content: str, chunk_size: int = 1000, overlap: int = 100) -> list:
    """
    Chunk a document into smaller sections.
    This function is a placeholder and currently returns an empty list.
    It will be implemented in the future to split the document content into overlapping chunks.
    Args:
        document_content (str): The content of the document to chunk.
        chunk_size (int, optional): The size of each chunk in characters. Defaults to 1000.
        overlap (int, optional): The number of characters to overlap between chunks. Defaults to 100.
    Returns:
        list: An empty list (placeholder for future implementation).
    """
    return []  # Placeholder for future implementation

if __name__ == "__main__":
    # Example usage of the document processing functions
    sample_file_path = 'path/to/your/sample/document.pdf'
    
    # Local file upload scenario
    print("Uploading local file...")
    with open(sample_file_path, 'rb') as file:
        upload_result = upload_to_blob(file, os.path.basename(sample_file_path))
    print(f"Local file upload: {upload_result['message']}")
    
    # Bytestream upload scenario
    print("\nUploading file as bytestream...")
    with open(sample_file_path, 'rb') as file:
        file_content = file.read()
    upload_result = upload_to_blob(file_content, "bytestream_" + os.path.basename(sample_file_path))
    print(f"Bytestream upload: {upload_result['message']}")
    
    # Analyze the uploaded document
    print("\nAnalyzing uploaded document...")
    analysis_result = analyze_document("bytestream_" + os.path.basename(sample_file_path))
    
    # Extract and display a sample of the full text from the result
    full_text = analysis_result.content
    print(f"Extracted full text length: {len(full_text)} characters")
    print("\nSample of extracted text:")
    print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
    
    print("\nNote: Document chunking is not implemented yet.")