import os
import json
import logging
import argparse
import numpy as np
from typing import Dict, Any, List

from src.services.ingestion.service import IngestionService

logger = logging.getLogger(__name__)

def run_ingestion_pipeline(
    image_dir: str,
    embedding_dir: str,
    keyframe_dir: str,
    model_name: str
):
    logger.info(f"Starting DB Ingestion Pipeline for model: {model_name}")
    
    try:
        service = IngestionService()
    except Exception as e:
        logger.error(f"Failed to initialize IngestionService: {e}")
        return

    model_emb_dir = os.path.join(embedding_dir, model_name)
    if not os.path.isdir(model_emb_dir):
        logger.error(f"Embedding directory not found: {model_emb_dir}")
        return

    # Scan and process
    for k_folder in sorted(os.listdir(model_emb_dir)):
        if not (k_folder.startswith("K") or k_folder.startswith("L")): continue
        k_path = os.path.join(model_emb_dir, k_folder)
        
        for v_file in sorted(os.listdir(k_path)):
            if not v_file.endswith(".npy"): continue
            
            v_id = v_file.replace(".npy", "")
            video_id = f"{k_folder}_{v_id}"
            
            # Load Embeddings
            npy_path = os.path.join(k_path, v_file)
            embeddings = np.load(npy_path)
            
            # Ensure Milvus Collection is created with right dim
            dim = embeddings.shape[1]
            service.setup_milvus_collection(model_name, dim)
            
            # Load Keyframe JSON to get timestamps
            json_path = os.path.join(keyframe_dir, f"{video_id}.json")
            timestamps = {}
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    jf = json.load(f)
                    for kf in jf.get("distilled_keyframes", []):
                        timestamps[str(kf["frame_index"])] = kf["timestamp"]
            
            # Read Image dir to match rows
            v_image_dir = os.path.join(image_dir, k_folder, v_id)
            if not os.path.isdir(v_image_dir):
                logger.warning(f"Image dir missing for {video_id}, skipping")
                continue
                
            image_files = [f for f in os.listdir(v_image_dir) if f.lower().endswith(".jpg")]
            image_files = sorted(
                image_files,
                key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else float("inf")
            )
            
            if len(image_files) != embeddings.shape[0]:
                logger.error(f"Mismatch! {len(image_files)} images vs {embeddings.shape[0]} vectors in {video_id}")
                continue
                
            # Build objects
            milvus_data = []
            elastic_data = []
            
            for i, img_file in enumerate(image_files):
                frame_idx = os.path.splitext(img_file)[0]
                keyframe_id = f"{video_id}_{frame_idx}"
                img_path = os.path.join(v_image_dir, img_file)
                ts = timestamps.get(frame_idx, 0.0)
                
                # Elastic Payload (Keyframe)
                elastic_data.append({
                    "id": keyframe_id,
                    "frame_id": keyframe_id,
                    "path": img_path,
                    "timestamp": ts,
                    "is_deleted": False,
                    "is_processed": True,
                    "dataset_id": k_folder,
                    "video_id": v_id
                })
                
                # Milvus Payload (Vector)
                milvus_data.append({
                    "id": keyframe_id,
                    "path": img_path,
                    "fps": 30, # default
                    "metadata": json.dumps({"timestamp": ts}),
                    "is_deleted": False,
                    "is_processed": True,
                    "embedding": embeddings[i].tolist()
                })
                
            # Inject to databases
            service.insert_keyframes(elastic_data)
            service.insert_embeddings(model_name, milvus_data)
            
            # Inject Video Metadata
            video_data = {
                "id": video_id,
                "path": f"sample/video/{video_id}.mp4",
                "fps": 30,
                "metadata": {},
                "is_deleted": False,
                "is_processed": True
            }
            service.insert_video_metadata(video_data)
            
            logger.info(f"Successfully processed and ingested {video_id} ({len(milvus_data)} frames)")

import yaml

def run_pipeline(config_path: str, model_name: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    image_dir = cfg["data"]["image_dir"]
    embedding_dir = cfg["data"]["embedding_dir"]
    keyframe_dir = cfg["data"]["keyframe_dir"]

    run_ingestion_pipeline(
        image_dir=image_dir,
        embedding_dir=embedding_dir,
        keyframe_dir=keyframe_dir,
        model_name=model_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_name", type=str, required=True)
    
    args = parser.parse_args()
    run_pipeline(args.config, args.model_name)
