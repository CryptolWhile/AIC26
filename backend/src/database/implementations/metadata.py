import logging
from src.database.interface import Database
import json
from typing import Any, Dict
import os

logger = logging.getLogger(__name__)

class MetadataStore(Database):
    def __init__(self, metadata_dir: str):
        super().__init__()
        self.metadata_dir = metadata_dir
        self.keyframe_collection = None
        self.video_collection = None
        self._is_connected = False
        self.connect()
    
    def connect(self) -> None:
        try:
            with open(os.path.join(self.metadata_dir, "keyframe.json"), "r") as f:
                self.keyframe_collection = json.load(f)
            with open(os.path.join(self.metadata_dir, "video.json"), "r") as f:
                self.video_collection = json.load(f)

            self._is_connected = True
            logger.info("Connected to Metadata Store successfully")
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to Metadata Store: {str(e)}")

    def search(self, collection_name: str, query: Dict[str, Any]) -> Any:
        if collection_name == "keyframe":
            return self.keyframe_collection[query["id"]]
        elif collection_name == "video":
            return self.video_collection[query["id"]]
        else:
            raise ValueError(f"Invalid collection name: {collection_name}")
    
    def ping(self) -> bool:
        return self._is_connected

    def get_collection(self, collection_name: str) -> Any:
        if collection_name == "keyframe":
            return self.keyframe_collection
        elif collection_name == "video":
            return self.video_collection
        else:
            raise ValueError(f"Invalid collection name: {collection_name}")

    def create_collection(self, collection_name: str, schema: Any) -> Any:
        return super().create_collection(collection_name, schema)
    
    def delete_collection(self, collection_name: str) -> None:
        return super().delete_collection(collection_name)

    def index_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        return super().index_data(collection_name, data, data_id)

    def delete_data(self, collection_name: str, data_id: str) -> None:
        return super().delete_data(collection_name, data_id)

    def update_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        return super().update_data(collection_name, data, data_id)
    
    def close(self) -> None:
        """Close connection to metadata store"""
        self.keyframe_collection = None
        self.video_collection = None
        self._is_connected = False
        logger.info("Closed Metadata Store connection")
    
    def ping(self) -> bool:
        return self._is_connected