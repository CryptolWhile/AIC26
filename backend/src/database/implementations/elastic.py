import logging
import time
from elasticsearch import Elasticsearch
from typing import Dict, Any, Optional, List

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
                verify_certs=False,
                request_timeout=60, 
                max_retries=3,
                retry_on_timeout=True,
                retry_on_status={502, 503, 504},
                http_compress=True,
                connections_per_node=10,
                http_auth=None
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
                # Extract mappings from schema if it contains a 'mappings' wrapper
                mappings = schema.get('mappings', schema) if 'mappings' in schema else schema
                result = self.client.indices.create(index=collection_name, mappings=mappings)
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
    
    

    def index_single_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
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

    def index_data(self, collection_name: str, data_list: List[Dict[str, Any]], chunk_size: int = 500) -> Any:
        """
        Index multiple data items efficiently using Elasticsearch bulk API with chunking.
        
        Args:
            collection_name (str): Name of the index/collection
            data_list (List[Dict[str, Any]]): List of data dictionaries to index
            chunk_size (int): Number of documents to process in each batch (default: 1000)
            
        Returns:
            Any: Aggregated bulk operation response
            
        Raises:
            Exception: If bulk indexing fails
        """
        self.ensure_connection()
        
        if not data_list:
            logger.warning("No data provided for bulk indexing")
            return {"items": []}
        
        try:
            total_successful = 0
            total_failed = 0
            all_errors = []
            all_items = []
            
            for i in range(0, len(data_list), chunk_size):
                chunk = data_list[i:i + chunk_size]
                logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(data_list) + chunk_size - 1)//chunk_size} "
                           f"({len(chunk)} documents)")
                
                bulk_operations = []
                for data in chunk:
                    bulk_operations.append({
                        "index": {
                            "_index": collection_name,
                            "_id": data["frame_id"]
                        }
                    })
                    bulk_operations.append(data)
                
                max_retries = 3
                retry_count = 0
                backoff_delay = 1  
                
                while retry_count < max_retries:
                    try:
                        result = self.client.bulk(
                            body=bulk_operations, 
                            refresh=False,
                            timeout='60s'
                        )
                        all_items.extend(result.get('items', []))
                        break  
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Failed to index chunk {i//chunk_size + 1} after {max_retries} retries: {str(e)}")
                            raise
                        else:
                            logger.warning(f"Retry {retry_count}/{max_retries} for chunk {i//chunk_size + 1} after error: {str(e)}")
                            time.sleep(backoff_delay)
                            backoff_delay *= 2  
                
                chunk_successful = 0
                chunk_failed = 0
                
                for item in result.get('items', []):
                    if 'index' in item:
                        if item['index'].get('status') in [200, 201]:
                            chunk_successful += 1
                        else:
                            chunk_failed += 1
                            all_errors.append(item['index'].get('error', 'Unknown error'))
                
                total_successful += chunk_successful
                total_failed += chunk_failed
                
                logger.info(f"Chunk {i//chunk_size + 1}: {chunk_successful} successful, {chunk_failed} failed")
            
            self.client.indices.refresh(index=collection_name)
            
            logger.info(f"Bulk indexed {total_successful} documents successfully to '{collection_name}' "
                       f"(total processed: {len(data_list)})")
            if total_failed > 0:
                logger.error(f"Failed to index {total_failed} documents. Errors: {all_errors[:10]}...")  # Show first 10 errors
            
            return {"items": all_items, "total_successful": total_successful, "total_failed": total_failed}
        
        except Exception as e:
            logger.error(f"Error bulk indexing data to '{collection_name}': {str(e)}")
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
            # Check if this is a full search body or just a query
            # Full search body contains top-level keys like size, aggs, query, highlight, etc.
            search_body_keys = ['size', 'aggs', 'aggregations', 'sort', '_source', 'query', 'highlight']
            if any(key in query for key in search_body_keys):
                # This is a full search body
                result = self.client.search(
                    index=collection_name,
                    body=query
                )
            else:
                # This is just a query object
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
            self.is_connected = False
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            raise
    
    def create_collection(self, collection_name: str, schema: Dict[str, Any]) -> Any:
        self.ensure_connection()
        try:
            if not self.client.indices.exists(index=collection_name):
                mappings = schema.get('mappings', schema) if 'mappings' in schema else schema
                result = self.client.indices.create(index=collection_name, mappings=mappings)
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
    
    def index_single_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
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
            # Check if this is a full search body or just a query
            # Full search body contains top-level keys like size, aggs, query, highlight, etc.
            search_body_keys = ['size', 'aggs', 'aggregations', 'sort', '_source', 'query', 'highlight']
            if any(key in query for key in search_body_keys):
                # This is a full search body
                result = self.client.search(
                    index=collection_name,
                    body=query
                )
            else:
                # This is just a query object
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
    