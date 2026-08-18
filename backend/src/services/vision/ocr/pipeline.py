import os
import gc
import json
import yaml
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from src.services.vision.ocr.service import OCRService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ocr_extraction_pipeline(
    image_dir: str,
    save_dir: str,
    provider_name: str,
    model_name: str,
    config: Dict[str, Any],
    K_start: int = 1,
    K_end: int = 20,
    V_start: int = 1,
    V_end: int = 9999,
    num_gpus: int = 1,
    gpu_id: int = 0,
):
    logger.info("Starting OCR extraction pipeline")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"OCR save directory: {save_dir}")

    image_root = Path(image_dir)
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    ocr_service = OCRService()
    ocr_service.register_model(
        config=config,
        provider_name=provider_name,
        model_name=model_name,
    )
    logger.info(f"Registered OCR model '{model_name}' with provider '{provider_name}'")

    k_folders = [d for d in image_root.iterdir() if d.is_dir() and (d.name.startswith("K") or d.name.startswith("L"))]
    k_folders = sorted(k_folders, key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else float('inf'))

    for k_folder in k_folders:
        try:
            k_idx = int(k_folder.name[1:])
            if not (K_start <= k_idx <= K_end):
                continue
        except ValueError:
            continue

        v_folders = [d for d in k_folder.iterdir() if d.is_dir() and d.name.startswith("V")]
        v_folders = sorted(v_folders, key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else float('inf'))

        for v_folder in v_folders:
            try:
                v_idx = int(v_folder.name[1:])
                if not (V_start <= v_idx <= V_end):
                    continue
            except ValueError:
                continue
                
            # Filter for GPUs
            video_hash = k_idx * 10000 + v_idx
            if video_hash % num_gpus != gpu_id:
                continue
            
            video_id = f"{k_folder.name}_{v_folder.name}"
            save_path = save_root / f"{video_id}.json"
            if save_path.exists():
                logger.info(f"Skipping {video_id}, OCR result already exists")
                continue

            logger.info(f"Processing OCR for video: {video_id}")
            image_files = sorted([f for f in v_folder.iterdir() if f.is_file() and f.suffix.lower() == '.jpg'],
                                 key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf'))
            
            if not image_files:
                logger.warning(f"No images found for {video_id}")
                continue

            ocr_results = []
            try:
                for img_file in tqdm(image_files, desc=f"OCR {video_id}", leave=False):
                    try:
                        result = ocr_service.extract_text_from_image(
                            image_path=str(img_file),
                            model_name=model_name
                        )
                        ocr_results.append({
                            "frame_id": img_file.stem,
                            "boxes": [{"text": box.text, "bbox": box.BBox} for box in result.boxes] if result.boxes else []
                        })
                    except Exception as e:
                        logger.error(f"Failed OCR on {img_file}: {e}")
                
                with save_path.open("w", encoding="utf-8") as f:
                    json.dump({"video_id": video_id, "ocr_results": ocr_results}, f, ensure_ascii=False, indent=4)
                
                logger.info(f"Saved OCR result -> {save_path}")
            except Exception as e:
                logger.exception(f"Failed to process {video_id}: {e}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

def run_pipeline(config_path: str, K_start: int, K_end: int, V_start: int, V_end: int, num_gpus: int = 1, gpu_id: int = 0):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        logger.error(f"Config file {config_path} does not exist")
        return

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ocr_extraction_pipeline(**cfg.get("ocr_extraction", {}), K_start=K_start, K_end=K_end, V_start=V_start, V_end=V_end, num_gpus=num_gpus, gpu_id=gpu_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR extraction pipeline")
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--K_start", type=int, default=1, help="Starting folder index for keyframe extraction"
    )
    parser.add_argument(
        "--K_end", type=int, default=20, help="Ending folder index for keyframe extraction"
    )
    parser.add_argument(
        "--V_start", type=int, default=1, help="Starting video index"
    )
    parser.add_argument(
        "--V_end", type=int, default=9999, help="Ending video index"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to split the workload across"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="Internal use only. The specific GPU ID this process handles"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error(f"Config file {args.config} not found")
    else:
        if args.num_gpus > 1 and args.gpu_id == 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
            import subprocess
            import sys
            logger.info(f"Auto-spawning {args.num_gpus} processes for multi-GPU execution...")
            processes = []
            for i in range(args.num_gpus):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(i)
                cmd = [sys.executable, "-m", "src.services.vision.ocr.pipeline"] + sys.argv[1:]
                
                if "--gpu_id" in cmd:
                    idx = cmd.index("--gpu_id")
                    cmd[idx+1] = str(i)
                else:
                    cmd.extend(["--gpu_id", str(i)])
                
                logger.info(f"Spawning GPU {i}: {' '.join(cmd)}")
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)
            
            for p in processes:
                p.wait()
            logger.info("All multi-GPU processes finished.")
        else:
            run_pipeline(args.config, args.K_start, args.K_end, args.V_start, args.V_end, args.num_gpus, args.gpu_id)
