import os
import torch
import logging
import argparse
import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image
from typing import Dict, Any
from src.services.embedding.service import EmbeddingService


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_images_from_folder(folder_path: str, valid_exts=(".jpg", ".jpeg", ".png")):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    image_files = sorted(
        image_files,
        key=lambda x: int(os.path.splitext(x)[0])
        if os.path.splitext(x)[0].isdigit()
        else float("inf"),
    )

    images = []
    for file in image_files:
        img_path = os.path.join(folder_path, file)
        images.append(img_path)
    return images


def embedding_extraction_pipeline(
    image_dir: str,
    embedding_dir: str,
    provider_name: str,
    model_name: str,
    config: Dict[str, Any],
    batch_size: int = 32,
    K_start: int = None,
    K_end: int = None,
    num_gpus: int = 1,
    gpu_id: int = 0
):
    logger.info("Starting embedding extraction pipeline")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Embedding directory: {embedding_dir}")

    embedding_service = EmbeddingService()
    embedding_service.registry_model(
        config=config,
        provider_name=provider_name,
        model_name=model_name,
    )

    list_image_folder = []
    for k_folder in sorted(os.listdir(image_dir)):
        # if not k_folder.startswith("K"): continue
        if not k_folder.startswith(("K", "L")): continue
        try:
            k_num = int(k_folder[1:])
        except ValueError:
            continue
        if K_start is not None and k_num < K_start:
            continue
        if K_end is not None and k_num > K_end:
            continue
        k_path = os.path.join(image_dir, k_folder)
        if not os.path.isdir(k_path): continue
        for v_folder in sorted(os.listdir(k_path)):
            if not v_folder.startswith("V"): continue
            if os.path.isdir(os.path.join(k_path, v_folder)):
                if num_gpus > 1:
                    try:
                        v_part = v_folder.split(".")[0] # V001
                        v_num = int(v_part[1:])
                        if v_num % num_gpus != gpu_id:
                            continue
                    except Exception:
                        pass
                list_image_folder.append(os.path.join(k_folder, v_folder))
    logger.info(f"Total folders to process: {len(list_image_folder)}, {list_image_folder}")

    for image_folder in tqdm(list_image_folder, desc="Processing folders", unit="folder"):
        image_path = os.path.join(image_dir, image_folder)
        if not os.path.isdir(image_path):
            continue

        images = load_images_from_folder(folder_path=image_path)

        if not images:
            logger.info(f"No valid images found in {image_folder}")
            continue

        output_path = os.path.join(embedding_dir, model_name, image_folder + ".npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        result = embedding_service.embed_image(
            images=images,
            model_name=model_name,
            batch_size=batch_size,
        )

        embeddings = result.embeddings
        if embeddings is not None:
            np.save(output_path, embeddings)
            logger.info(f"Saved embeddings to {output_path}")

def get_models_by_name(cfg, model_name):

    for model in cfg["models"]:
        if model["name"] == model_name:
            return model
    return None

def run_pipeline(config_path: str, model_name: str, K_start: int=None, K_end: int=None, num_gpus: int=1, gpu_id: int=0):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    image_dir = cfg["data"]["image_dir"]
    embedding_dir = cfg["data"]["embedding_dir"]
    batch_size = cfg.get("batch_size", 32)

    model = get_models_by_name(cfg, model_name)
    if not model:
        logger.error(f"No models found for provider: {model_name}")
        return

    provider_name = model["provider"]
    model_name = model["name"]
    model_config = model.get("config", {})

    logger.info(f"Running embedding for model: {provider_name}/{model_name}")
    embedding_extraction_pipeline(
        image_dir=image_dir,
        embedding_dir=embedding_dir,
        provider_name=provider_name,
        model_name=model_name,
        config=model_config,
        batch_size=batch_size,
        K_start=K_start,
        K_end=K_end,
        num_gpus=num_gpus,
        gpu_id=gpu_id
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_name", type=str, required=True, help="Model provider name")
    parser.add_argument("--K_start", type=int, help="Start index for processing")
    parser.add_argument("--K_end", type=int, help="End index for processing")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the workload across")
    parser.add_argument("--gpu_id", type=int, default=0, help="Internal use only")

    args = parser.parse_args()

    if args.num_gpus > 1 and args.gpu_id == 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        import subprocess
        import sys
        logger.info(f"Auto-spawning {args.num_gpus} processes for multi-GPU execution...")
        processes = []
        for i in range(args.num_gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            cmd = [sys.executable, "-m", "src.services.embedding.pipeline"] + sys.argv[1:]
            
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
        run_pipeline(args.config, args.model_name, args.K_start, args.K_end, args.num_gpus, args.gpu_id)