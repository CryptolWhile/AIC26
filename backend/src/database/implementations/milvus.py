import logging
from typing import List, Dict, Any, Optional

from pymilvus import (
    connections, 
    Collection,
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType
)

from src.core.config import MilvusSettings
from src.database.interface import Database


logger = logging.getLogger(__name__)


class MilvusDatabase(Database):
    
    instance = None
    
    def __init__(self, settings: MilvusSettings):
        pass
    
    def __new__(cls, settings: MilvusSettings):
        if cls.instance is None:
            cls.instance = super(MilvusDatabase, cls).__new__(cls)
            cls.instance.initialize(settings=settings)
        return cls.instance
    
    def initialize(self, settings: MilvusSettings):
        super().__init__()
        
        self.settings = settings
        self.connect()
    
    def connect(self) -> None:
        try:
            connections.connect(
                alias="default",
                host=self.settings.MILVUS_HOST,
                port=self.settings.MILVUS_PORT
            )
            
            self.is_connected = True
            logger.info("Connected to Milvus successfully")
            
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def create_collection(self, collection_name: str, schema: List[FieldSchema], index_params: Optional[Dict[str, Any]] = None) -> Collection:
        self.ensure_connection()
        if utility.has_collection(collection_name=collection_name):
            logger.warning(f"Collection '{collection_name}' already exists. Skipping creation.")
            return Collection(name=collection_name)
        
        collection_schema = CollectionSchema(fields=schema, description=collection_name)
        collection = Collection(name=collection_name, schema=collection_schema)
        
        if index_params is None:
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        collection.load()
        
        logger.info(f"Collection '{collection_name}' created successfully")
        return collection
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        self.ensure_connection()
        if utility.has_collection(collection_name=collection_name):
            collection = Collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' loaded successfully")
            return collection
        else:
            logger.error(f"Collection '{collection_name}' does not exist")
            return None
    
    def delete_collection(self, collection_name: str) -> None:
        self.ensure_connection()
        try:
            if utility.has_collection(collection_name=collection_name):
                collection = Collection(name=collection_name)
                collection.drop()
                logger.info(f"Collection '{collection_name}' dropped successfully")
            
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {str(e)}")
            raise
    
    def index_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name=collection_name)
            if 'id' not in data:
                data['id'] = data_id
            
            result = collection.insert(data=[data])
            collection.flush()
            collection.load()
            logger.info(f"Data indexed successfully in collection '{collection_name}'")
            return result
        
        except Exception as e:
            logger.error(f"Error indexing data '{data_id}': {str(e)}")
            raise

    def index_bulk_data(self, collection_name: str, data_list: List[Dict[str, Any]]) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name=collection_name)
            if not data_list:
                return None
            result = collection.insert(data=data_list)
            logger.info(f"{len(data_list)} items indexed successfully in collection '{collection_name}'")
            return result
        except Exception as e:
            logger.error(f"Error bulk indexing data: {str(e)}")
            raise

    def update_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name=collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")        
            result = collection.upsert(data=[{"id": data_id, **data}])
            collection.flush()
            collection.load()
            logger.info(f"Data '{data_id}' updated successfully in collection '{collection_name}'")
            return result
        
        except Exception as e:
            logger.error(f"Error updating data '{data_id}': {str(e)}")
            raise
    
    def delete_data(self, collection_name: str, data_id: str) -> None:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name=collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            expr = f"id == '{data_id}'"
            result = collection.delete(expr=expr)
            collection.flush()
            collection.load()
            logger.info(f"Data '{data_id}' deleted successfully from collection '{collection_name}'")
            return result
        
        except Exception as e:
            logger.error(f"Error deleting data '{data_id}': {str(e)}")
            raise
    
    def search(self, collection_name: str, query: Dict[str, Any]) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name=collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            data = query.get('data', [])
            anns_field = query.get('anns_field', 'embedding')
            param = query.get('param', {})
            limit = query.get('limit', 10)
            output_fields = query.get("output_fields", ["id"])
            collection.load()
            
            # Check if collection has any data before searching
            # This helps avoid the "Unsupported field type: 0" error when collection is empty
            try:
                # Use num_entities to check if collection has data
                if collection.num_entities == 0:
                    # Collection is empty, return empty search results
                    logger.info(f"Collection '{collection_name}' is empty, returning empty search results")
                    return [[]]  # Return empty results in the expected format
            except Exception:
                # If num_entities fails, proceed with search anyway
                pass
            
            results = collection.search(
                data=data,
                anns_field=anns_field,
                param=param,
                limit=limit,
                output_fields=output_fields,
                consistency_level="Strong"
            )
            return results
        
        except Exception as e:
            # Handle the specific "Unsupported field type: 0" error that occurs with empty collections
            if "Unsupported field type: 0" in str(e):
                logger.warning(f"Collection '{collection_name}' appears to be empty or corrupted, returning empty results")
                return [[]]  # Return empty results in the expected format
            
            logger.error(f"Error searching in collection '{collection_name}': {str(e)}")
            raise
    
    def ping(self) -> bool:
        try:
            utility.list_collections()
            return True
        
        except Exception as e:
            logger.error(f"Milvus ping failed: {str(e)}")
            return False
    
    def close(self) -> None:
        try:
            connections.disconnect("default")
            self.is_connected = False
            logger.info("Milvus connection closed")
            
        except Exception as e:
            logger.error(f"Error closing Milvus connection: {str(e)}") 