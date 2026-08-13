import os
import av
import cv2
import json
import argparse
import yaml
import logging
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, Any

import gc
import torch

from src.services.processing.utils import default_serializer, get_fps, iter_videos_in_range, iter_json_in_range
from src.services.embedding.service import EmbeddingService
from src.services.processing.shot_extraction.service import ShotExtractionService
from src.services.processing.keyframe_extraction.interface import Shot, VideoShots
from src.services.processing.keyframe_extraction.service import KeyframeExtractionService


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def shot_extraction_pipeline(
    video_dir: str,
    save_dir: str,
    extractor_name: str,
    provider_name: str,
    config: Optional[Dict[str, Any]] = None,
    K_start: int = 1,
    K_end: int = 20,
    V_start: int = 1,
    V_end: int = 9999,
    num_gpus: int = 1,
    gpu_id: int = 0,
):
    video_root = Path(video_dir)
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    logger.info("Starting shot extraction pipeline")
    logger.info(f"Video directory: {video_root}")
    logger.info(f"Save directory: {save_root}")

    shot_extraction_service = ShotExtractionService()
    shot_extraction_service.register_extractor(
        config=config,
        extractor_name=extractor_name,
        provider_name=provider_name,
    )
    logger.info(f"Registered extractor '{extractor_name}' with provider '{provider_name}'")

    for video_path in tqdm(iter_videos_in_range(video_root, K_start, K_end, V_start, V_end, num_gpus, gpu_id),
                           desc="Processing videos", unit="video"):
        filename = video_path.name
        save_path = save_root / f"{video_path.stem}.json"

        logger.info(f"Processing video: {video_path}")
        try:
            shot_result = shot_extraction_service.extract_shots(
                video_path=str(video_path),
                extractor_name=extractor_name,
            )

            with save_path.open("w", encoding="utf-8") as f:
                json.dump(
                    asdict(shot_result),
                    f,
                    default=default_serializer,
                    ensure_ascii=False,
                    indent=4,
                )

            logger.info(f"Saved result for {filename} -> {save_path}")
        except Exception as e:
            logger.exception(f"Failed to process {filename}: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def keyframe_extraction_pipeline(
    shot_dir: str,
    keyframe_dir: str,
    image_dir: str,
    provider_name: str,
    model_name: str,
    config: Dict[str, Any],
    keyframe_ratio: float,
    compare_length: int = 2,
    threshold: float = 0.92,
    K_start: int = 1,
    K_end: int = 20,
    V_start: int = 1,
    V_end: int = 9999,
    num_gpus: int = 1,
    gpu_id: int = 0,
):
    logger.info("Starting keyframe extraction pipeline")
    logger.info(f"Shot directory: {shot_dir}")
    logger.info(f"Keyframe JSON directory: {keyframe_dir}")
    logger.info(f"Image directory: {image_dir}")

    shot_root = Path(shot_dir)
    keyframe_root = Path(keyframe_dir)
    image_root = Path(image_dir)

    keyframe_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    default_cfg = {
        "model_name": "laion/CLIP-ViT-L-14-laion2B-b82K",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    default_cfg.update(config or {})
    config = default_cfg

    embedding_service = EmbeddingService()
    embedding_service.registry_model(
        config=config,
        provider_name=provider_name,
        model_name=model_name,
    )
    logger.info(f"Registered model '{model_name}' with provider '{provider_name}'")

    keyframe_extraction_service = KeyframeExtractionService(
                min_keyframes_per_shot=1,
                max_keyframes_per_shot=100,
            )
    logger.info("Initialized KeyframeExtractionService")

    for shot_path in tqdm(iter_json_in_range(shot_root, K_start, K_end, V_start, V_end, num_gpus, gpu_id),
                          desc="Processing shot files", unit="file"):
        keyframe_path = keyframe_root / shot_path.name
        if keyframe_path.exists():
            logger.info(f"Skipping {shot_path.name}, already exists")
            continue
            
        logger.info(f"Processing shot file: {shot_path}")

        try:
            with shot_path.open("r", encoding="utf-8") as f:
                shot_data = json.load(f)

            shots_raw = shot_data.get("shots", [])
            if not shots_raw:
                logger.warning(f"No 'shots' found inside {shot_path}; skipping.")
                continue

            shots = [
                Shot(
                    shot_index=shot["shot_index"],
                    start_frame=shot["start_frame"],
                    end_frame=shot["end_frame"],
                )
                for shot in shot_data.get("shots", [])
            ]

            video_path = Path(shot_data.get("video_path", ""))
            video_fps = get_fps(str(video_path))

            video_shots = VideoShots(
                video_path=str(video_path),
                video_fps=video_fps,
                shots=shots,
            )

            keyframe_result = keyframe_extraction_service.extract_keyframes_from_shots(
                video_shots=video_shots,
                embedding_model=embedding_service.get_model(model_name=model_name),
                keyframe_ratio=keyframe_ratio,
                compare_length=compare_length,
                threshold=threshold,
            )

            video_name = video_path.stem
            parts = video_name.split("_")
            if len(parts) < 2:
                raise ValueError(f"Unexpected video name format: {video_name}")
            K_name, V_name = parts[0], parts[1]
            # if not K_name.startswith("K") or not V_name.startswith("V"):
            if not K_name.startswith(("K", "L")) or not V_name.startswith("V"):
                raise ValueError(f"Unexpected video name format: {video_name}")

            save_img_folder = image_root / K_name / V_name
            save_img_folder.mkdir(parents=True, exist_ok=True)

            target_indices = {kf.frame_index for kf in keyframe_result.distilled_keyframes}
            if not target_indices:
                logger.warning(f"No keyframes extracted for video: {video_path}")
            else:
                container = None
                try:
                    container = av.open(str(video_path))
                    video_stream = next(s for s in container.streams if s.type == "video")

                    for current_index, frame in tqdm(
                        enumerate(container.decode(video_stream)),
                        desc=f"Saving keyframes for {video_name}",
                        unit="frame",
                        disable=True
                    ):
                        if current_index in target_indices:
                            try:
                                img = frame.to_ndarray(format="bgr24")
                                out_path = save_img_folder / f"{current_index}.jpg"
                                wrote = cv2.imwrite(str(out_path), img)

                                if not wrote:
                                    logger.error(f"Failed to write image to {out_path}")

                                del img  # Free memory

                            except Exception as e:
                                logger.exception(f"Error saving frame {current_index} of {video_name}: {e}")

                finally:
                    if container is not None:
                        try:
                            container.close()
                        except Exception as e:
                            pass
                    del container
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            with keyframe_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(keyframe_result), f, default=default_serializer, ensure_ascii=False, indent=4)

            logger.info(f"Saved keyframe result -> {keyframe_path}")
            logger.info(f"Saved keyframe images -> {save_img_folder}")

        except Exception as e:
            logger.exception(f"Failed to process {shot_path.name}: {e}")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def run_pipeline(pipeline: str, config_path: str, K_start: int, K_end: int, V_start: int, V_end: int, num_gpus: int = 1, gpu_id: int = 0):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        logger.error(f"Config file {config_path} does not exist")
        return

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if pipeline == "shot_extraction":
        shot_extraction_pipeline(**cfg.get("shot_extraction", {}), K_start=K_start, K_end=K_end, V_start=V_start, V_end=V_end, num_gpus=num_gpus, gpu_id=gpu_id)
    elif pipeline == "keyframe_extraction":
        keyframe_extraction_pipeline(**cfg.get("keyframe_extraction", {}), K_start=K_start, K_end=K_end, V_start=V_start, V_end=V_end, num_gpus=num_gpus, gpu_id=gpu_id)
    else:
        logger.error(f"Unknown pipeline: {pipeline}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video processing pipelines")
    parser.add_argument(
        "pipeline", choices=["shot_extraction", "keyframe_extraction"],
        help="Which pipeline to run"
    )
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
                cmd = [sys.executable, "-m", "src.services.processing.pipeline"] + sys.argv[1:]
                
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
            run_pipeline(args.pipeline, args.config, args.K_start, args.K_end, args.V_start, args.V_end, args.num_gpus, args.gpu_id)