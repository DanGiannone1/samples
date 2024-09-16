"""
This module handles interactions with Azure Cosmos DB, including database and container creation,
and CRUD operations on documents.

***Requirements***
azure-cosmos==4.5.1

"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, exceptions, PartitionKey

# Load environment variables
load_dotenv()

# Cosmos DB Configuration
COSMOS_HOST = os.getenv("COSMOS_HOST")
COSMOS_MASTER_KEY = os.getenv("COSMOS_MASTER_KEY")
COSMOS_DATABASE_ID = os.getenv("COSMOS_DATABASE_ID")
COSMOS_CONTAINER_ID = os.getenv("COSMOS_CONTAINER_ID")

class CosmosDB:
    def __init__(self):
        """
        Initialize the CosmosDB client and create database and container if they don't exist.
        """
        if not all([COSMOS_HOST, COSMOS_MASTER_KEY, COSMOS_DATABASE_ID, COSMOS_CONTAINER_ID]):
            raise ValueError("Cosmos DB configuration is incomplete")

        self.client = CosmosClient(COSMOS_HOST, {'masterKey': COSMOS_MASTER_KEY}, user_agent="CosmosDBPythonSDK", user_agent_overwrite=True)
        self.database = None
        self.container = None

        self._initialize_database_and_container()

    def _initialize_database_and_container(self):
        """Create database and container if they don't exist."""
        try:
            try:
                self.database = self.client.create_database(id=COSMOS_DATABASE_ID)
                print(f'Database with id \'{COSMOS_DATABASE_ID}\' created')
            except exceptions.CosmosResourceExistsError:
                self.database = self.client.get_database_client(COSMOS_DATABASE_ID)
                print(f'Database with id \'{COSMOS_DATABASE_ID}\' was found')

            try:
                self.container = self.database.create_container(id=COSMOS_CONTAINER_ID, partition_key=PartitionKey(path='/partitionKey'))
                print(f'Container with id \'{COSMOS_CONTAINER_ID}\' created')
            except exceptions.CosmosResourceExistsError:
                self.container = self.database.get_container_client(COSMOS_CONTAINER_ID)
                print(f'Container with id \'{COSMOS_CONTAINER_ID}\' was found')
        except exceptions.CosmosHttpResponseError as e:
            print(f'An error occurred: {e.message}')
            raise

    def create_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new item in the container.
        
        Args:
            item (Dict[str, Any]): The item to create.
        
        Returns:
            Dict[str, Any]: The created item.
        """
        return self.container.create_item(body=item)

    def read_item(self, item_id: str, partition_key: str) -> Dict[str, Any]:
        """
        Read an item from the container.
        
        Args:
            item_id (str): The ID of the item to read.
            partition_key (str): The partition key of the item.
        
        Returns:
            Dict[str, Any]: The read item.
        """
        return self.container.read_item(item=item_id, partition_key=partition_key)

    def update_item(self, item_id: str, updates: Dict[str, Any], partition_key: str) -> Dict[str, Any]:
        """
        Update an item in the container.
        
        Args:
            item_id (str): The ID of the item to update.
            updates (Dict[str, Any]): The updates to apply to the item.
            partition_key (str): The partition key of the item.
        
        Returns:
            Dict[str, Any]: The updated item.
        """
        item = self.read_item(item_id, partition_key)
        item.update(updates)
        return self.container.upsert_item(body=item)

    def delete_item(self, item_id: str, partition_key: str) -> None:
        """
        Delete an item from the container.
        
        Args:
            item_id (str): The ID of the item to delete.
            partition_key (str): The partition key of the item.
        """
        self.container.delete_item(item=item_id, partition_key=partition_key)

    def query_items(self, query: str, parameters: List[Dict[str, Any]] = None, partition_key: str = None) -> List[Dict[str, Any]]:
        """
        Query items from the container.
        
        Args:
            query (str): The query string.
            parameters (List[Dict[str, Any]], optional): Query parameters.
            partition_key (str, optional): The partition key to query within.
        
        Returns:
            List[Dict[str, Any]]: The query results.
        """
        return list(self.container.query_items(
            query=query,
            parameters=parameters,
            partition_key=partition_key,
            enable_cross_partition_query=(partition_key is None)
        ))

    def get_items_by_partition_key(self, partition_key: str) -> List[Dict[str, Any]]:
        """
        Retrieve all items for a specific partition key.
        
        Args:
            partition_key (str): The partition key to query.
        
        Returns:
            List[Dict[str, Any]]: The items in the specified partition.
        """
        query = "SELECT * FROM c WHERE c.partitionKey = @partitionKey"
        parameters = [{"name": "@partitionKey", "value": partition_key}]
        return self.query_items(query, parameters, partition_key)

    def get_items_with_field(self, field_name: str, partition_key: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve all items that have a specific field.
        
        Args:
            field_name (str): The name of the field to check for.
            partition_key (str, optional): The partition key to query within.
        
        Returns:
            List[Dict[str, Any]]: The items containing the specified field.
        """
        query = f"SELECT * FROM c WHERE IS_DEFINED(c.{field_name})"
        return self.query_items(query, partition_key=partition_key)

if __name__ == "__main__":
    # Example usage
    try:
        cosmos_db = CosmosDB()
        
        # Create an item
        new_item = {
            'id': 'item1',
            'partitionKey': 'sample_partition',
            'name': 'John Doe',
            'age': 30
        }
        created_item = cosmos_db.create_item(new_item)
        print(f"Created item: {created_item['id']}")

        # Read the item
        read_item = cosmos_db.read_item('item1', 'sample_partition')
        print(f"Read item: {read_item['name']}")

        # Update the item
        updated_item = cosmos_db.update_item('item1', {'age': 31}, 'sample_partition')
        print(f"Updated item age: {updated_item['age']}")

        # Query items by partition key
        partition_items = cosmos_db.get_items_by_partition_key('sample_partition')
        print(f"Items in partition: {len(partition_items)}")

        # Query items with a specific field
        items_with_age = cosmos_db.get_items_with_field('age')
        print(f"Items with 'age' field: {len(items_with_age)}")

        # Delete the item
        cosmos_db.delete_item('item1', 'sample_partition')
        print("Item deleted")

    except Exception as e:
        print(f"An error occurred: {str(e)}")