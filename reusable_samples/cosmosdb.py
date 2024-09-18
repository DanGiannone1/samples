"""
This module handles interactions with Azure Cosmos DB, including database and container creation,
and CRUD operations on documents.

Requirements:
    azure-cosmos==4.5.1
"""

import os
import logging
from functools import wraps
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from azure.cosmos.container import ContainerProxy
from azure.cosmos.database import DatabaseProxy

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cosmos DB Configuration
COSMOS_HOST = os.environ.get("COSMOS_HOST")
COSMOS_MASTER_KEY = os.environ.get("COSMOS_MASTER_KEY")
COSMOS_DATABASE_ID = os.environ.get("COSMOS_DATABASE_ID")
COSMOS_CONTAINER_ID = os.environ.get("COSMOS_CONTAINER_ID")

def cosmos_error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Cosmos DB error in {func.__name__}: {e.message}")
            raise
    return wrapper

class CosmosDB:
    def __init__(self):
        """
        Initialize the CosmosDB client and create database and container if they don't exist.

        Raises
        ------
        ValueError
            If any of the required Cosmos DB configuration values are missing.
        """
        if not all([COSMOS_HOST, COSMOS_MASTER_KEY, COSMOS_DATABASE_ID, COSMOS_CONTAINER_ID]):
            raise ValueError("Cosmos DB configuration is incomplete")

        self.client: CosmosClient = CosmosClient(COSMOS_HOST, {'masterKey': COSMOS_MASTER_KEY}, user_agent="CosmosDBPythonSDK", user_agent_overwrite=True)
        self.database: Optional[DatabaseProxy] = None
        self.container: Optional[ContainerProxy] = None

        self._initialize_database_and_container()

    def _initialize_database_and_container(self) -> None:
        """
        Create database and container if they don't exist.

        Raises
        ------
        exceptions.CosmosHttpResponseError
            If there's an error in creating or getting the database or container.
        """
        try:
            try:
                self.database = self.client.create_database(id=COSMOS_DATABASE_ID)
                logger.info(f'Database with id \'{COSMOS_DATABASE_ID}\' created')
            except exceptions.CosmosResourceExistsError:
                self.database = self.client.get_database_client(COSMOS_DATABASE_ID)
                logger.info(f'Database with id \'{COSMOS_DATABASE_ID}\' was found')

            try:
                self.container = self.database.create_container(id=COSMOS_CONTAINER_ID, partition_key=PartitionKey(path='/partitionKey'))
                logger.info(f'Container with id \'{COSMOS_CONTAINER_ID}\' created')
            except exceptions.CosmosResourceExistsError:
                self.container = self.database.get_container_client(COSMOS_CONTAINER_ID)
                logger.info(f'Container with id \'{COSMOS_CONTAINER_ID}\' was found')
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f'An error occurred: {e.message}')
            raise

    @cosmos_error_handler
    def create_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new item in the container.

        Parameters
        ----------
        item : Dict[str, Any]
            The item to create.

        Returns
        -------
        Dict[str, Any]
            The created item.
        """
        created_item = self.container.create_item(body=item)
        logger.info(f"Item created with id: {created_item['id']}")
        return created_item

    @cosmos_error_handler
    def read_item(self, item_id: str, partition_key: str) -> Dict[str, Any]:
        """
        Read an item from the container.

        Parameters
        ----------
        item_id : str
            The ID of the item to read.
        partition_key : str
            The partition key of the item.

        Returns
        -------
        Dict[str, Any]
            The read item.
        """
        item = self.container.read_item(item=item_id, partition_key=partition_key)
        logger.info(f"Item read with id: {item['id']}")
        return item

    @cosmos_error_handler
    def update_item(self, item_id: str, updates: Dict[str, Any], partition_key: str) -> Dict[str, Any]:
        """
        Update an item in the container.

        Parameters
        ----------
        item_id : str
            The ID of the item to update.
        updates : Dict[str, Any]
            The updates to apply to the item.
        partition_key : str
            The partition key of the item.

        Returns
        -------
        Dict[str, Any]
            The updated item.
        """
        item = self.read_item(item_id, partition_key)
        item.update(updates)
        updated_item = self.container.upsert_item(body=item)
        logger.info(f"Item updated with id: {updated_item['id']}")
        return updated_item

    @cosmos_error_handler
    def delete_item(self, item_id: str, partition_key: str) -> None:
        """
        Delete an item from the container.

        Parameters
        ----------
        item_id : str
            The ID of the item to delete.
        partition_key : str
            The partition key of the item.
        """
        self.container.delete_item(item=item_id, partition_key=partition_key)
        logger.info(f"Item deleted with id: {item_id}")

    @cosmos_error_handler
    def query_items(self, query: str, parameters: Optional[List[Dict[str, Any]]] = None, partition_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query items from the container.

        Parameters
        ----------
        query : str
            The query string.
        parameters : Optional[List[Dict[str, Any]]], optional
            Query parameters.
        partition_key : Optional[str], optional
            The partition key to query within.

        Returns
        -------
        List[Dict[str, Any]]
            The query results.
        """
        items = list(self.container.query_items(
            query=query,
            parameters=parameters,
            partition_key=partition_key,
            enable_cross_partition_query=(partition_key is None)
        ))
        logger.info(f"Query returned {len(items)} items")
        return items

    def get_items(self, condition: str, parameters: Optional[List[Dict[str, Any]]] = None, partition_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve items based on a condition.

        Parameters
        ----------
        condition : str
            The WHERE clause of the query.
        parameters : Optional[List[Dict[str, Any]]], optional
            Query parameters.
        partition_key : Optional[str], optional
            The partition key to query within.

        Returns
        -------
        List[Dict[str, Any]]
            The items matching the condition.
        """
        query = f"SELECT * FROM c WHERE {condition}"
        return self.query_items(query, parameters, partition_key)

    def get_items_by_partition_key(self, partition_key: str) -> List[Dict[str, Any]]:
        """
        Retrieve all items for a specific partition key.

        Parameters
        ----------
        partition_key : str
            The partition key to query.

        Returns
        -------
        List[Dict[str, Any]]
            The items in the specified partition.
        """
        return self.get_items("c.partitionKey = @partitionKey", [{"name": "@partitionKey", "value": partition_key}], partition_key)

    def get_items_with_field(self, field_name: str, partition_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all items that have a specific field.

        Parameters
        ----------
        field_name : str
            The name of the field to check for.
        partition_key : Optional[str], optional
            The partition key to query within.

        Returns
        -------
        List[Dict[str, Any]]
            The items containing the specified field.
        """
        return self.get_items(f"IS_DEFINED(c.{field_name})", partition_key=partition_key)

def run_examples():
    """Example usage of the CosmosDB class."""
    try:
        cosmos_db = CosmosDB()
        
        # Create an item
        new_item = {
            'id': 'item1',
            'partitionKey': 'people',
            'name': 'John Doe',
            'age': 30
        }
        created_item = cosmos_db.create_item(new_item)
        logger.info(f"Created item: {created_item['id']}")

        # Read the item
        read_item = cosmos_db.read_item('item1', 'sample_partition')
        logger.info(f"Read item: {read_item['name']}")

        # Update the item
        updated_item = cosmos_db.update_item('item1', {'age': 31}, 'sample_partition')
        logger.info(f"Updated item age: {updated_item['age']}")

        # Query items by partition key
        partition_key = 'sample_partition'
        items = cosmos_db.get_items("c.partitionKey = @partitionKey", [{"name": "@partitionKey", "value": partition_key}], partition_key)
        logger.info(f"Found {len(items)} items with partition key: {partition_key}")

        # Get items by field
        field_name = 'age'
        items = cosmos_db.get_items(f"IS_DEFINED(c.{field_name})", partition_key=partition_key)
        logger.info(f"Found {len(items)} items with field: {field_name}")

        # Delete the item
        cosmos_db.delete_item('item1', 'sample_partition')
        logger.info("Item deleted")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    run_examples()