"""
### adls.py ###

This module handles interactions with Azure Data Lake Storage Gen2 (which is built on Azure Blob Storage).
It provides functionality to upload files, list blobs, and move blobs between containers. The goal of this module is to provide 
clear and concise examples of how to run basic ADLS operations using a particular SDK version. 

Requirements:
    azure-storage-blob==12.22.0

"""

import os
from typing import Union, Dict, Any, List
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import io

class ADLSManager:
    def __init__(self, storage_account_name=None, storage_account_container=None):
        """
        Initialize the ADLSManager with environment variables and blob service client.
        """
        self._load_env_variables(storage_account_name, storage_account_container)
        self.blob_service_client = self._get_blob_service_client()

    def _load_env_variables(self, storage_account_name=None, storage_account_container=None):
        """
        Load environment variables required for Azure Data Lake Storage operations.
        """
        load_dotenv()
        self.storage_account_name = storage_account_name or os.environ.get("STORAGE_ACCOUNT_NAME")
        self.storage_account_key = os.environ.get("STORAGE_ACCOUNT_KEY")
        self.storage_account_container = storage_account_container or os.environ.get("STORAGE_ACCOUNT_CONTAINER", "documents")
        self.tenant_id = os.environ.get("TENANT_ID", '16b3c013-d300-468d-ac64-7eda0820b6d3')

        if not self.storage_account_name:
            raise ValueError("STORAGE_ACCOUNT_NAME must be provided or set as an environment variable")

    def _get_blob_service_client(self) -> BlobServiceClient:
        """
        Get the Blob Service Client using either key-based authentication or DefaultAzureCredential.

        Returns:
            BlobServiceClient: The initialized Blob Service Client.
        """
        print("Initializing Blob service client")
        if self.storage_account_key:
            print("Using key-based authentication for Blob storage")
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={self.storage_account_name};AccountKey={self.storage_account_key};EndpointSuffix=core.windows.net"
            return BlobServiceClient.from_connection_string(connection_string)
        else:
            print("Using DefaultAzureCredential for Blob storage authentication")
            account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
            credential = DefaultAzureCredential(
                interactive_browser_tenant_id=self.tenant_id,
                visual_studio_code_tenant_id=self.tenant_id,
                workload_identity_tenant_id=self.tenant_id,
                shared_cache_tenant_id=self.tenant_id
            )
            return BlobServiceClient(account_url=account_url, credential=credential)

    def upload_to_blob(self, file_content: Union[bytes, io.IOBase], filename: str, container_name: str = None) -> Dict[str, str]:
        """
        Upload a file to Azure Blob Storage.

        Args:
            file_content (Union[bytes, io.IOBase]): The content of the file to upload.
            filename (str): The name of the file in the blob storage.
            container_name (str, optional): The name of the container to upload to. Defaults to self.storage_account_container.

        Returns:
            Dict[str, str]: A dictionary containing the upload message and the blob URL.
        """
        container_name = container_name or self.storage_account_container
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(filename)
        
        if isinstance(file_content, io.IOBase):
            blob_client.upload_blob(file_content.read(), overwrite=True)
        else:
            blob_client.upload_blob(file_content, overwrite=True)
        
        print(f"File {filename} uploaded successfully")
        return {"message": f"File {filename} uploaded successfully", "blob_url": blob_client.url}

    def list_blobs_in_folder(self, folder_name: str, container_name: str = None) -> List[Any]:
        """
        List all blobs in a specified folder within a container.

        Args:
            folder_name (str): The name of the folder to list blobs from.
            container_name (str, optional): The name of the container. Defaults to self.storage_account_container.

        Returns:
            List[Any]: A list of blob objects in the specified folder.
        """
        container_name = container_name or self.storage_account_container
        container_client = self.blob_service_client.get_container_client(container_name)
        
        return [blob for blob in container_client.list_blobs() if blob.name.startswith(folder_name)]

    def move_blob(self, source_blob_name: str, 
                  destination_blob_name: str, 
                  source_container_name: str = None, 
                  destination_container_name: str = None) -> Dict[str, str]:
        """
        Move a blob from one location to another within Azure Blob Storage.

        Args:
            source_blob_name (str): The name of the source blob.
            destination_blob_name (str): The name of the destination blob.
            source_container_name (str, optional): The name of the source container. Defaults to self.storage_account_container.
            destination_container_name (str, optional): The name of the destination container. Defaults to source_container_name.

        Returns:
            Dict[str, str]: A dictionary containing the move message.
        """
        source_container_name = source_container_name or self.storage_account_container
        destination_container_name = destination_container_name or source_container_name

        source_container_client = self.blob_service_client.get_container_client(source_container_name)
        destination_container_client = self.blob_service_client.get_container_client(destination_container_name)

        source_blob = source_container_client.get_blob_client(source_blob_name)
        destination_blob = destination_container_client.get_blob_client(destination_blob_name)
        
        destination_blob.start_copy_from_url(source_blob.url)
        source_blob.delete_blob()
        message = f"Moved blob from {source_blob_name} to {destination_blob_name}"
        print(message)
        return {"message": message}
    
    
def example_upload_local_file(sample_file_path):
    """Example usage of upload_to_blob function for uploading a local file."""
    adls_manager = ADLSManager()
    print("Uploading local file...")
    with open(sample_file_path, 'rb') as file:
        upload_result = adls_manager.upload_to_blob(file, os.path.basename(sample_file_path))
    print(f"Local file upload: {upload_result['message']}")

def example_upload_bytestream(sample_file_path):
    """Example usage of upload_to_blob function for uploading a file as bytestream."""
    adls_manager = ADLSManager()
    print("Uploading file as bytestream...")
    with open(sample_file_path, 'rb') as file:
        file_content = file.read()
    upload_result = adls_manager.upload_to_blob(file_content, "bytestream_" + os.path.basename(sample_file_path))
    print(f"Bytestream upload: {upload_result['message']}")

def example_list_blobs():
    """Example usage of list_blobs_in_folder function."""
    adls_manager = ADLSManager()
    print("Listing blobs in 'source' folder...")
    blobs = adls_manager.list_blobs_in_folder("source/")
    print(f"Found {len(blobs)} blobs in the 'source' folder")
    return blobs

def example_move_blob(blobs):
    """Example usage of move_blob function."""
    adls_manager = ADLSManager()
    if blobs:
        source_blob_name = blobs[0].name
        destination_blob_name = source_blob_name.replace("source/", "processed/")
        print(f"Moving blob {source_blob_name} to {destination_blob_name}...")
        move_result = adls_manager.move_blob(source_blob_name, destination_blob_name)
        print(move_result['message'])

if __name__ == "__main__":
    sample_file_path = "D:/temp/djg/337 Goldman Drive Inspection Report 20230730.pdf"
    
    example_upload_local_file(sample_file_path)
    example_upload_bytestream(sample_file_path)
    blobs = example_list_blobs()
    example_move_blob(blobs)