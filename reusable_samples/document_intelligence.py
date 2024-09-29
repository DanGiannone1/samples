"""
### document_intelligence.py ###

This module handles interactions with Azure Document Intelligence.
It provides functionality to analyze documents stored in Azure Blob Storage or local files.

Requirements:
    azure-ai-documentintelligence==1.0.0b2
    Azure Document Intelligence API Version: 2024-7-31 preview
"""

import os
import logging
from typing import Union
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.identity import DefaultAzureCredential

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Set to WARNING to suppress INFO logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set our logger to INFO

# Disable other loggers
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.ai.documentintelligence').setLevel(logging.WARNING)
logging.getLogger('azure.identity').setLevel(logging.WARNING)

class DocumentIntelligenceManager:
    def __init__(self):
        self._load_env_variables()
        self.document_intelligence_client = self._get_document_intelligence_client()

    def _load_env_variables(self):
        load_dotenv()
        self.document_intelligence_endpoint = os.environ.get("DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.document_intelligence_key = os.environ.get("DOCUMENT_INTELLIGENCE_KEY")
        self.tenant_id = os.environ.get("TENANT_ID", '16b3c013-d300-468d-ac64-7eda0820b6d3')

        if not self.document_intelligence_endpoint:
            raise ValueError("DOCUMENT_INTELLIGENCE_ENDPOINT environment variable is not set")

    def _get_document_intelligence_client(self) -> DocumentIntelligenceClient:
        logger.info("Initializing Document Intelligence client")
        if self.document_intelligence_key:
            logger.info("Using key-based authentication for Document Intelligence")
            credential = AzureKeyCredential(self.document_intelligence_key)
        else:
            logger.info("Using DefaultAzureCredential for Document Intelligence authentication")
            credential = DefaultAzureCredential(
                interactive_browser_tenant_id=self.tenant_id,
                visual_studio_code_tenant_id=self.tenant_id,
                workload_identity_tenant_id=self.tenant_id,
                shared_cache_tenant_id=self.tenant_id
            )
        return DocumentIntelligenceClient(endpoint=self.document_intelligence_endpoint, credential=credential)

    def read_document(self, document: Union[str, bytes], model_id: str) -> AnalyzeResult:
        try:
            if isinstance(document, str):
                if document.startswith(('http://', 'https://')):
                    logger.info(f"Reading document from URL: {document}")
                    analyze_request = {"urlSource": document}
                    poller = self.document_intelligence_client.begin_analyze_document(model_id, analyze_request)
                elif os.path.isfile(document):
                    logger.info(f"Reading document from local file: {document}")
                    with open(document, "rb") as file:
                        poller = self.document_intelligence_client.begin_analyze_document(model_id, analyze_request=file, content_type="application/octet-stream")
                else:
                    raise ValueError("Invalid document input. Expected URL or local file path.")
            elif isinstance(document, bytes):
                logger.info("Reading document from bytes content")
                poller = self.document_intelligence_client.begin_analyze_document(model_id, analyze_request=document, content_type="application/octet-stream")
            else:
                raise ValueError("Invalid document input. Expected URL, local file path, or bytes.")

            result = poller.result()
            logger.info("Successfully read the document with Document Intelligence and extracted text.")
            return result

        except Exception as e:
            logger.error(f"Error in read_document: {str(e)}")
            raise

def run_examples():
    try:
        doc_intelligence_manager = DocumentIntelligenceManager()
        model_id = "prebuilt-layout"  # Use an appropriate model ID
        
        storage_account_name = os.environ.get("STORAGE_ACCOUNT_NAME")
        storage_account_container = os.environ.get("STORAGE_ACCOUNT_CONTAINER", "documents")

        # Example 1: Read a document from a blob URL
        blob_url = f"https://{storage_account_name}.blob.core.windows.net/{storage_account_container}/337 Goldman Drive.pdf"
        url_result = doc_intelligence_manager.read_document(blob_url, model_id)
        logger.info("Document read from URL successfully")
        
        # Example 2: Read a document from a local file
        local_file_path = "D:/temp/djg/337 Goldman Drive Inspection Report 20230730.pdf"
        local_result = doc_intelligence_manager.read_document(local_file_path, model_id)
        logger.info("Document read from local file successfully")
        
        # Example 3: Read a document from bytes content
        with open(local_file_path, "rb") as f:
            bytes_content = f.read()
        bytes_result = doc_intelligence_manager.read_document(bytes_content, model_id)
        logger.info("Document read from bytes content successfully")

        # Process and display results
        for i, result in enumerate([url_result, local_result, bytes_result], 1):
            print(f"Example {i} - Extracted text:")
            print(result.content[:500] + "..." if len(result.content) > 500 else result.content)
            print("\n")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    run_examples()