"""
### cosmos_db.py ###

This module handles interactions with Azure Cosmos DB, including database and container creation,
and CRUD operations on documents. It automatically selects between key-based and DefaultAzureCredential
authentication based on the presence of COSMOS_MASTER_KEY. Logging is configured to show only
custom messages.

Requirements:
    azure-cosmos==4.5.1
    azure-identity==1.12.0
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from azure.cosmos.container import ContainerProxy
from azure.cosmos.database import DatabaseProxy
from azure.identity import DefaultAzureCredential

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Set to WARNING to suppress INFO logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set our logger to INFO

# Disable other loggers
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)
logging.getLogger('azure.identity').setLevel(logging.WARNING)

class CosmosDBManager:
    def __init__(self):
        self._load_env_variables()
        self.client = self._get_cosmos_client()
        self.database: Optional[DatabaseProxy] = None
        self.container: Optional[ContainerProxy] = None
        self._initialize_database_and_container()

    def _load_env_variables(self):
        load_dotenv()
        self.cosmos_host = os.environ.get("COSMOS_HOST")
        self.cosmos_master_key = os.environ.get("COSMOS_MASTER_KEY")
        self.cosmos_database_id = os.environ.get("COSMOS_DATABASE_ID")
        self.cosmos_container_id = os.environ.get("COSMOS_CONTAINER_ID")
        self.tenant_id = os.environ.get("TENANT_ID", '16b3c013-d300-468d-ac64-7eda0820b6d3')

        if not all([self.cosmos_host, self.cosmos_database_id, self.cosmos_container_id]):
            raise ValueError("Cosmos DB configuration is incomplete")

    def _get_cosmos_client(self) -> CosmosClient:
        logger.info("Initializing Cosmos DB client")
        if self.cosmos_master_key:
            logger.info("Using key-based authentication for Cosmos DB")
            return CosmosClient(self.cosmos_host, {'masterKey': self.cosmos_master_key})
        else:
            logger.info("Using DefaultAzureCredential for Cosmos DB authentication")
            credential = DefaultAzureCredential(
                interactive_browser_tenant_id=self.tenant_id,
                visual_studio_code_tenant_id=self.tenant_id,
                workload_identity_tenant_id=self.tenant_id,
                shared_cache_tenant_id=self.tenant_id
            )
            return CosmosClient(self.cosmos_host, credential=credential)

    def _initialize_database_and_container(self) -> None:
        try:
            self.database = self._create_or_get_database()
            self.container = self._create_or_get_container()
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f'An error occurred: {e.message}')
            raise

    def _create_or_get_database(self) -> DatabaseProxy:
        try:
            database = self.client.create_database(id=self.cosmos_database_id)
            logger.info(f'Database with id \'{self.cosmos_database_id}\' created')
        except exceptions.CosmosResourceExistsError:
            database = self.client.get_database_client(self.cosmos_database_id)
            logger.info(f'Database with id \'{self.cosmos_database_id}\' was found')
        return database

    def _create_or_get_container(self) -> ContainerProxy:
        try:
            container = self.database.create_container(id=self.cosmos_container_id, partition_key=PartitionKey(path='/partitionKey'))
            logger.info(f'Container with id \'{self.cosmos_container_id}\' created')
        except exceptions.CosmosResourceExistsError:
            container = self.database.get_container_client(self.cosmos_container_id)
            logger.info(f'Container with id \'{self.cosmos_container_id}\' was found')
        return container

    def create_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        created_item = self.container.create_item(body=item)
        logger.info(f"Item created with id: {created_item['id']}")
        return created_item

    def read_item(self, item_id: str, partition_key: str) -> Dict[str, Any]:
        item = self.container.read_item(item=item_id, partition_key=partition_key)
        logger.info(f"Item read with id: {item['id']}")
        return item

    def update_item(self, item_id: str, updates: Dict[str, Any], partition_key: str) -> Dict[str, Any]:
        item = self.read_item(item_id, partition_key)
        item.update(updates)
        updated_item = self.container.upsert_item(body=item)
        logger.info(f"Item updated with id: {updated_item['id']}")
        return updated_item

    def delete_item(self, item_id: str, partition_key: str) -> None:
        self.container.delete_item(item=item_id, partition_key=partition_key)
        logger.info(f"Item deleted with id: {item_id}")

    def query_items(self, query: str, parameters: Optional[List[Dict[str, Any]]] = None, partition_key: Optional[str] = None) -> List[Dict[str, Any]]:
        items = list(self.container.query_items(
            query=query,
            parameters=parameters,
            partition_key=partition_key,
            enable_cross_partition_query=(partition_key is None)
        ))
        logger.info(f"Query returned {len(items)} items")
        return items

    def get_items_by_partition_key(self, partition_key: str) -> List[Dict[str, Any]]:
        query = "SELECT * FROM c WHERE c.partitionKey = @partitionKey"
        parameters = [{"name": "@partitionKey", "value": partition_key}]
        return self.query_items(query, parameters, partition_key)

def run_examples():
    try:
        cosmos_db = CosmosDBManager()
        logger.info("Connected to Cosmos DB")

        new_item = {
            'id': 'item1',
            'partitionKey': 'example_partition',
            'name': 'John Doe',
            'age': 30
        }
        created_item = cosmos_db.create_item(new_item)
        read_item = cosmos_db.read_item('item1', 'example_partition')
        updated_item = cosmos_db.update_item('item1', {'age': 31}, 'example_partition')
        items = cosmos_db.get_items_by_partition_key('example_partition')
        cosmos_db.delete_item('item1', 'example_partition')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    run_examples()