import os
import json
import logging
from abc import ABC
from typing import List, Dict, Any

import numpy as np
import faiss

from src.database.interface import Database

logger = logging.getLogger(__name__)

class FaissDatabase(Database):

    def __init__(self, index_dir: str, metadata_file: str):
        super().__init__()
        self.index_dir = index_dir
        self.metadata_file = metadata_file
        self._indexes: Dict[str, faiss.Index] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}  # Dict of lists per collection
        self._ensure_directories()
        self.connect()
    
    def _ensure_directories(self) -> None:
        """Ensure index directory exists"""
        try:
            os.makedirs(self.index_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directories: {e}")

    def connect(self) -> None:
        try:
            # Load metadata if exists
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        logger.info(f"Converting old metadata format (list) to new format (dict)")
                        self._metadata = {}
                        for name in ["hf_clip_H", "hf_clip_L", "hf_siglip"]:
                            self._metadata[name] = loaded_data 
                    elif isinstance(loaded_data, dict):
                        self._metadata = loaded_data
                    else:
                        logger.warning(f"Unexpected metadata format, starting with empty metadata")
                        self._metadata = {}
            else:
                self._metadata = {}
                logger.info(f"No metadata file found at {self.metadata_file}, starting with empty metadata")

            # Load or create indexes
            for name in ["hf_clip_H", "hf_clip_L", "hf_siglip"]:
                bin_path = os.path.join(self.index_dir, f"{name}.bin")
                if os.path.exists(bin_path):
                    try:
                        self._indexes[name] = faiss.read_index(bin_path)
                        logger.info(f"Loaded FAISS index {name} from {bin_path}, total vectors: {self._indexes[name].ntotal}")
                    except Exception as e:
                        logger.error(f"Failed to load index {name}: {e}")
                        # Create empty index as fallback
                        self._create_empty_index(name)
                else:
                    logger.warning(f"Index file not found: {bin_path}, creating empty index")
                    # Create empty index so searches don't fail
                    self._create_empty_index(name)
                
                # Ensure metadata for this collection exists
                if name not in self._metadata:
                    self._metadata[name] = []

            self._is_connected = True
            logger.info(f"FAISS Database connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to FAISS DB: {e}")
            self._is_connected = False

    def close(self) -> None:
        """Clear memory"""
        self._indexes.clear()
        self._metadata = {}
        self._is_connected = False

    def _create_empty_index(self, collection_name: str) -> None:
        """Create an empty FAISS index with default dimension"""
        try:
            # Default dimensions for each model
            dimensions = {
                "hf_clip_H": 1024,
                "hf_clip_L": 768, 
                "hf_siglip": 1152
            }
            dimension = dimensions.get(collection_name, 768)
            
            # Create a flat index for inner product (cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(dimension)
            self._indexes[collection_name] = index
            logger.info(f"Created empty FAISS index for {collection_name} with dimension {dimension}")
        except Exception as e:
            logger.error(f"Failed to create empty index for {collection_name}: {e}")
    
    def ping(self) -> bool:
        return self._is_connected

    def get_collection(self, collection_name: str) -> Any:
        return self._indexes.get(collection_name)

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

    def search(self, collection_name: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search in FAISS index and return metadata"""
        if collection_name not in self._indexes:
            logger.warning(f"Collection {collection_name} not found in indexes")
            return []
        
        index = self._indexes[collection_name]
        
        # Check if index is empty
        if index.ntotal == 0:
            logger.warning(f"Index {collection_name} is empty")
            return []
        
        try:
            vector = np.array(query["embedding"], dtype="float32")[np.newaxis, :]
            faiss.normalize_L2(vector) 

            if hasattr(index, "hnsw"):
                efSearch = query.get("hnsw", 400)  # Changed from efSearch to hnsw as in the query
                index.hnsw.efSearch = efSearch
            
            k = min(query.get("k", 5), index.ntotal)  # Don't search for more than available
            distances, indices = index.search(vector, k)
            
            results = []
            collection_metadata = self._metadata.get(collection_name, [])
            
            for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx < 0:  # Invalid index from FAISS
                    continue
                if idx < len(collection_metadata):
                    results.append({
                        "rank": rank + 1,
                        "distance": float(dist),
                        "metadata": collection_metadata[idx]
                    })
                else:
                    logger.warning(f"Index {idx} out of range for metadata (size: {len(collection_metadata)})")

            metric = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT)
            if metric == faiss.METRIC_L2:
                results.sort(key=lambda x: x["distance"])
            else:
                results.sort(key=lambda x: x["distance"], reverse=True) 

            return results
            
        except Exception as e:
            logger.error(f"Error during FAISS search in {collection_name}: {str(e)}")
            return []