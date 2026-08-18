import os
import json
import yaml
import logging
import argparse

logger = logging.getLogger(__name__)

def run_ocr_ingestion_pipeline(ocr_dir: str):
    logger.info("Starting OCR Ingestion Pipeline")
    
    try:
        from src.services.vision.ocr.service import OCRService
        ocr_service = OCRService()
    except Exception as e:
        logger.error(f"Failed to initialize OCRService: {e}")
        return

    ocr_root = os.path.abspath(ocr_dir)
    if not os.path.isdir(ocr_root):
        logger.error(f"OCR directory not found: {ocr_root}")
        return

    for json_file in sorted(os.listdir(ocr_root)):
        if not json_file.endswith(".json"): continue
        
        json_path = os.path.join(ocr_root, json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            video_id = data.get("video_id")
            if not video_id:
                continue
                
            k_folder = video_id.split("_")[0]
            v_id = video_id.split("_")[1]
            
            ocr_results = data.get("ocr_results", [])
            elastic_data = []
            
            for res in ocr_results:
                frame_idx = res.get("frame_id")
                boxes = res.get("boxes", [])
                
                full_text = " ".join([b.get("text", "") for b in boxes if b.get("text")]).strip()
                
                keyframe_id = f"{video_id}_{frame_idx}"
                img_path = f"processed/images/{k_folder}/{v_id}/{frame_idx}.jpg"
                
                elastic_data.append({
                    "id": keyframe_id,
                    "frame_id": keyframe_id,
                    "path": img_path,
                    "dataset_id": k_folder,
                    "video_id": v_id,
                    "ocr": full_text
                })
                
            if elastic_data:
                msg, status = ocr_service.insert(elastic_data)
                if status == 200:
                    logger.info(f"Successfully ingested OCR for {video_id} ({len(elastic_data)} frames)")
                else:
                    logger.error(f"Failed to ingest OCR for {video_id}: {msg}")
                    
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")

def run_pipeline(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ocr_dir = cfg["data"]["ocr_dir"]
    run_ocr_ingestion_pipeline(ocr_dir=ocr_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    
    args = parser.parse_args()
    run_pipeline(args.config)
