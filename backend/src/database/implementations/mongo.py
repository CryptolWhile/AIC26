import logging
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from typing import List, Dict, Any, Optional

from src.core.config import MongoSettings
from src.database.interface import Database


logger = logging.getLogger(__name__)

class MongoDatabase(Database):
    
    instance = None
    
    def __init__(self, settings: MongoSettings):
        pass
    
    def __new__(cls, settings: MongoSettings):
        if cls.instance is None:
            cls.instance = super(MongoDatabase, cls).__new__(cls)
            cls.instance.initialize(settings=settings)
        return cls.instance

    def initialize(self, settings: MongoSettings):
        super().__init__()

        self.settings = settings
        self.client = MongoClient = None
        self.db = None
        self.connect()

    def connect(self)-> None:
        try:
            if not self.settings.MONGODB_URI:
                raise ValueError("MongoDB URI is not set in environment variables")

            self.client = MongoClient(self.settings.MONGODB_URI)
            self.db = self.client[self.settings.MONGO_DATABASE]

            self.ping()
            self.is_connected = True
            logger.info(f"Connected to MongoDatabase successfully - Database: {self.settings.MONGO_DATABASE}")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.is_connected = False
            logger.error(f"Failed to connect to MongoDatabase: {str(e)}")
            raise
        
        except Exception as e:
            self.is_connected = False
            logger.error(f"Unexpected error connecting to MongoDatabase: {str(e)}")
            raise

    def create_collection(self, collection_name: str, schema: Optional[Dict[str, Any]] = None):
        # schema: bsonType,required,properties
        self.ensure_connection()
        if self.db is None:
            raise RuntimeError("Database connection not established")

        if schema:
            collection = self.db.create_collection(
                collection_name,
                validator={"$jsonSchema": schema}
            )
        else:
            collection = self.db[collection_name]

        logger.info(f"Collection '{collection_name}' created successfully")
        return collection

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        self.ensure_connection()
        if self.db is not None:
            return self.db[collection_name]
        return None

    def delete_collection(self, collection_name: str) -> None:
        self.ensure_connection()
        if self.db is None:
            raise RuntimeError("Database connection not established")
        
        try:
            if collection_name in self.db.list_collection_names():
                self.db.drop_collection(collection_name)
                logger.info(f"Collection '{collection_name}' dropped successfully")
            else:
                logger.warning(f"Collection '{collection_name}' does not exist")
        
        except Exception as e:
            logger.error(f"Failed to drop collection '{collection_name}': {str(e)}")
            raise

    def index_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                raise ValueError(f"Collection '{collection_name}' does not exist")
        
            document_copy = data.copy()
            document_copy['_id'] = data_id
            
            try:
                result = collection.insert_one(document_copy)
                logger.info(f"Data '{data_id}' indexed successfully in collection '{collection_name}'")
                return result
            
            except Exception as e:
                logger.error(f"Failed to index data '{data_id}': {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error indexing data '{data_id}': {str(e)}")
            raise

    def update_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            result = collection.update_one(
                {"_id": data_id},
                {"$set": data} #$set: ghi đè dữ liệu
            )
            
            logger.info(f"Data '{data_id}' updated successfully in collection '{collection_name}'")
            return result
        
        except Exception as e:
            logger.error(f"Error updating data '{data_id}': {str(e)}")
            raise

    def delete_data(self, collection_name: str, data_id: str) -> None:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                raise ValueError(f"Collection '{collection_name}' does not exist")
        
            result = collection.delete_one({"_id": data_id}) #BSON {}
            
            if result.deleted_count > 0:
                logger.info(f"Data '{data_id}' deleted successfully from collection '{collection_name}'")
            else:
                logger.warning(f"Data '{data_id}' not found in collection '{collection_name}'")
        
        except Exception as e:
            logger.error(f"Failed to delete data '{data_id}': {str(e)}")
            raise

    def search(self, collection_name: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            results = list(collection.find(query))
            logger.info(f"Search completed in collection '{collection_name}', found {len(results)} documents")
            return results
        
        except Exception as e:
            logger.error(f"Search failed in collection '{collection_name}': {str(e)}")
            raise

    def ping(self) -> bool:
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        
        except Exception as e:
            logger.error(f"MongoDatabase ping failed: {str(e)}")
            return False
    
    def close(self) -> None:
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("MongoDatabase connection closed")