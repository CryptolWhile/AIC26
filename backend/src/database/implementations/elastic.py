import logging
from elasticsearch import Elasticsearch
from typing import Dict, Any, Optional

from src.core.config import ElasticSettings
from src.database.interface import Database


logger = logging.getLogger(__name__)


class ElasticDatabase(Database):
    
    instance = None

    def __init__(self, settings: ElasticSettings):
        pass
    
    def __new__(cls, settings: ElasticSettings):
        if cls.instance is None:
            cls.instance = super(ElasticDatabase, cls).__new__(cls)
            cls.instance.initialize(settings=settings)
        return cls.instance
    
    def initialize(self, settings: ElasticSettings):
        super().__init__()
        
        self.settings = settings
        self.client: Elasticsearch = None
        self.connect()

    def connect(self) -> None:
        try:
            if not self.settings.ELASTIC_URI:
                raise ValueError("Elasticsearch URI is not set in environment variables")
            
            self.client = Elasticsearch(
                self.settings.ELASTIC_URI,
                verify_certs=False
            )
            
            self.is_connected = True
            logger.info("Connected to Elasticsearch successfully")
            
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            raise

    def create_collection(self, collection_name: str, schema: Dict[str, Any]) -> Any:
        self.ensure_connection()
        try:
            if not self.client.indices.exists(index=collection_name):
                result = self.client.indices.create(index=collection_name, **schema)
                logger.info(f"Created index '{collection_name}' successfully")
                return result
            else:
                logger.info(f"Index '{collection_name}' already exists")
                return None
            
        except Exception as e:
            logger.error(f"Error creating index '{collection_name}': {str(e)}")
            raise

    def get_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        self.ensure_connection()
        try:
            if self.client.indices.exists(index=collection_name):
                response = self.client.search(
                    index=collection_name,
                    body={
                        "query": {"match_all": {}},
                        "size": 1000
                    }
                )
                return response
            else:
                logger.error(f"Index '{collection_name}' does not exist")
                return None
            
        except Exception as e:
            logger.error(f"Error getting index '{collection_name}': {str(e)}")
            return None

    def delete_collection(self, collection_name: str) -> None:
        self.ensure_connection()
        try:
            if self.client.indices.exists(index=collection_name):
                self.client.indices.delete(index=collection_name)
                logger.info(f"Deleted index '{collection_name}' successfully")
            else:
                logger.error(f"Index '{collection_name}' does not exist")
                
        except Exception as e:
            logger.error(f"Error deleting index '{collection_name}': {str(e)}")
            raise

    def index_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        self.ensure_connection()
        try:
            result = self.client.index(
                index=collection_name,
                document=data,
                id=data_id
            )
            logger.info(f"Indexed data '{data_id}' successfully: {result['result']}")
            return result
        
        except Exception as e:
            logger.error(f"Error indexing data '{data_id}': {str(e)}")
            raise

    def update_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        self.ensure_connection()
        try:
            result = self.client.update(
                index=collection_name,
                doc=data,
                id=data_id
            )
            logger.info(f"Updated data '{data_id}' successfully: {result['result']}")
            return result
        
        except Exception as e:
            logger.error(f"Error updating data '{data_id}': {str(e)}")
            raise

    def delete_data(self, collection_name: str, data_id: str) -> None:
        self.ensure_connection()
        try:
            result = self.client.delete(index=collection_name, id=data_id)
            logger.info(f"Deleted data '{data_id}' successfully from index '{collection_name}'")
            return result
        
        except Exception as e:
            logger.error(f"Error deleting data '{data_id}' from index '{collection_name}': {str(e)}")
            raise

    def search(self, collection_name: str, query: Dict[str, Any]) -> Any:
        self.ensure_connection()
        try:
            result = self.client.search(
                index=collection_name,
                query=query
            )
            return result
        
        except Exception as e:
            logger.error(f"Error searching in index '{collection_name}': {str(e)}")
            raise

    def ping(self) -> bool:
        try:
            if self.client:
                return self.client.ping()
            return False
        
        except Exception as e:
            logger.error(f"Elasticsearch ping failed: {str(e)}")
            return False
    
    def close(self) -> None:
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("Elasticsearch connection closed") 