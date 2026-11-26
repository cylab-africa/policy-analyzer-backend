from pymongo import MongoClient
from pymongo.errors import OperationFailure
from app.core.config import settings
import json



class MongoDBClient:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

    def close(self):
        self.client.close()

    def is_connected(self):
        try:
            # print(self.client.server_info())  # This will raise an exception if not connected
            self.client.admin.command('ping')
            return True
        except (ConnectionError, OperationFailure):
            return False
        
    def create_index(self, collection_name: str, index_fields: list, unique: bool = False):

        collection = self.get_collection(collection_name)
        index_spec = [(field, 1) for field in index_fields]
        
        try:
            collection.create_index(index_spec, unique=unique)
            # print(f"Index created on {collection_name} for fields {index_fields}")
        except Exception as e:
            # print(f"Error creating index on {collection_name}: {str(e)}")
            return False
        return True
    
        
# print(f"Initializing MongoDB client-> {settings.mongodb_connection_string}") 
mongo_client = MongoDBClient(
    uri=settings.mongodb_connection_string,
    db_name=settings.mongodb_database_name
)
# print(f"MongoDB connection established: {mongo_client.is_connected()}")