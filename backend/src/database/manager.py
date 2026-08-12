import logging
from typing import Dict, Type, List, Optional

from src.core.config import Settings
from src.database.interface import Database
from src.database.implementations import MongoDatabase, MilvusDatabase, ElasticDatabase, FaissDatabase, MetadataStore


logger = logging.getLogger(__name__)


class DatabaseManager:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.database_registry: Dict[str, Database] = {}
        
        self.databases = {
            "mongo": MongoDatabase,
            "milvus": MilvusDatabase,
            "elastic": ElasticDatabase,
            "faiss": FaissDatabase,
            "metadata_store": MetadataStore
        }
        
        self.initialize(settings=settings)
    
    def initialize(self, settings: Settings) -> None:
        for db_name, db_class in self.databases.items():
            try:
                if db_name == "faiss":
                    faiss_settings = getattr(settings, db_name)
                    self.database_registry[db_name] = db_class(
                        index_dir=faiss_settings.FAISS_INDEX_DIR,
                        metadata_file=faiss_settings.FAISS_METADATA_FILE
                    )
                elif db_name == "metadata_store":
                    metadata_store_settings = getattr(settings, db_name)
                    self.database_registry[db_name] = db_class(
                        metadata_dir=metadata_store_settings.METADATA_STORE_DIR
                    )
                else:
                    self.database_registry[db_name] = db_class(getattr(settings, db_name))
                logger.info(f"Initialized {db_name} database successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {db_name} database: {str(e)}")
    
    def get_database(self, db_name: str) -> Optional[Database]:
        return self.database_registry.get(db_name)
    
    def get_mongo(self) -> Optional[MongoDatabase]:
        return self.database_registry.get('mongo')
    
    def get_milvus(self) -> Optional[MilvusDatabase]:
        return self.database_registry.get('milvus')
    
    def get_elastic(self) -> Optional[ElasticDatabase]:
        return self.database_registry.get('elastic')
    
    def get_faiss(self) -> Optional[FaissDatabase]:
        return self.database_registry.get('faiss')
    
    def get_metadata_store(self) -> Optional[MetadataStore]:
        return self.database_registry.get('metadata_store')
    
    def register_database(self, db_name: str, db_instance: Database) -> None:
        self.database_registry[db_name] = db_instance
        logger.info(f"Added {db_name} database to manager")
    
    def remove_database(self, db_name: str) -> None:
        if db_name in self.databases:
            self.database_registry[db_name].close()
            del self.database_registry[db_name]
            logger.info(f"Removed {db_name} database from manager")
    
    def get_available_databases(self) -> List[str]:
        return list(self.database_registry.keys())
    
    def check_connections(self) -> Dict[str, bool]:
        status = {}
        for db_name, db_instance in self.database_registry.items():
            try:
                status[db_name] = db_instance.ping()
            except Exception as e:
                logger.error(f"Error checking {db_name} connection: {str(e)}")
                status[db_name] = False
        return status
    
    def reconnect_all(self) -> None:
        for db_name, db_instance in self.database_registry.items():
            try:
                if not db_instance.is_connected():
                    db_instance.connect()
                    logger.info(f"Reconnected to {db_name} successfully")
                    
            except Exception as e:
                logger.error(f"Failed to reconnect to {db_name}: {str(e)}")
    
    def close_connections(self) -> None:
        for db_name, db_instance in self.database_registry.items():
            try:
                db_instance.close()
                logger.info(f"Closed {db_name} connection")
                
            except Exception as e:
                logger.error(f"Error closing {db_name} connection: {str(e)}")
        
        logger.info("All database connections closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connections()
    
    def __del__(self):
        try:
            self.close_connections()
        except:
            pass 