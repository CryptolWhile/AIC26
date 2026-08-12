import logging
import json
from typing import Dict, Any, List

from src.core.config import load_settings
from src.database.manager import DatabaseManager
from src.models.video_model import Video
from src.models.keyframe_model import Keyframe
from src.database.utils import (
    pydantic_to_milvus_schema,
    pydantic_to_mongo_schema,
    pydantic_to_elastic_schema
)

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        try:
            settings = load_settings()
            self.db_manager = DatabaseManager(settings)
            
            self.milvus_db = self.db_manager.get_milvus()
            self.elastic_db = self.db_manager.get_elastic()
            self.mongo_db = self.db_manager.get_mongo()
            
            if not self.milvus_db or not self.elastic_db or not self.mongo_db:
                raise ValueError("Databases were not properly initialized by DatabaseManager")
            
            # Setup Collections
            self._setup_collections()
            logger.info("IngestionService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IngestionService: {str(e)}")
            raise

    def _setup_collections(self):
        # Elasticsearch - keyframe
        keyframe_elastic_schema = pydantic_to_elastic_schema(Keyframe)
        self.elastic_db.create_collection("keyframe", keyframe_elastic_schema)

        # MongoDB - video metadata
        video_mongo_schema = pydantic_to_mongo_schema(Video)
        self.mongo_db.create_collection("video", video_mongo_schema)

    def setup_milvus_collection(self, collection_name: str, embedding_dim: int):
        video_milvus_schema = pydantic_to_milvus_schema(
            model=Video,
            primary_key="id",
            embedding_field="embedding",
            embedding_dim=embedding_dim
        )
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.milvus_db.create_collection(
            collection_name=collection_name,
            schema=video_milvus_schema,
            index_params=index_params
        )

    def insert_video_metadata(self, video_data: Dict[str, Any]):
        self.mongo_db.index_data(
            collection_name="video",
            data=video_data,
            data_id=video_data["id"]
        )
        logger.info(f"Inserted video metadata to Mongo: {video_data['id']}")
        
    def insert_keyframes(self, keyframes: List[Dict[str, Any]]):
        if not keyframes:
            return
        self.elastic_db.index_data(
            collection_name="keyframe",
            data_list=keyframes
        )
        logger.info(f"Inserted {len(keyframes)} keyframes to Elasticsearch")
        
    def insert_embeddings(self, collection_name: str, embeddings_data: List[Dict[str, Any]]):
        if not embeddings_data:
            return
        if hasattr(self.milvus_db, 'index_bulk_data'):
            self.milvus_db.index_bulk_data(collection_name, embeddings_data)
        else:
            for data in embeddings_data:
                self.milvus_db.index_data(collection_name, data, data["id"])
        logger.info(f"Inserted {len(embeddings_data)} embeddings to Milvus collection {collection_name}")
