import json
from datetime import datetime
from dataclasses import asdict, is_dataclass
import sys
import os
import torch

os.chdir("backend")

from src.services.processing.keyframe_extraction.service import KeyframeExtractionService
from src.services.processing.keyframe_extraction.interface import VideoShots, Shot
from src.services.processing.keyframe_extraction.utils import get_fps
from src.services.embedding.service import EmbeddingService

print("Starting keyframe extraction test...")

config = {
    "model_name": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

embedding_service = EmbeddingService()
embedding_service.register_model(
    config=config,
    provider_name="hf_clip_multimodal",
    model_name="CLIP-ViT-L-14"
)

with open("sample/processed/shots/sample_video.json", "r", encoding="utf-8") as f:
    shot_data = json.load(f)

shots = [
    Shot(
        shot_index=shot['shot_index'],
        start_frame=shot['start_frame'],
        end_frame=shot['end_frame']
    )
    for shot in shot_data["shots"]
]

# Ensure the video path is relative to backend since it was probably saved as backend/sample/...
video_path = shot_data["video_path"]
if video_path.startswith("backend/"):
    video_path = video_path.replace("backend/", "", 1)

video_shots = VideoShots(
    video_path=video_path,
    video_fps=get_fps(video_path),
    shots=shots
)

keyframe_extraction_service = KeyframeExtractionService(
    min_keyframes_per_shot=1,
    max_keyframes_per_shot=100
)

keyframe_result = keyframe_extraction_service.extract_keyframes_from_shots(
    video_shots=video_shots,
    embedding_model=embedding_service.get_model(model_name="CLIP-ViT-L-14"),
    keyframe_ratio=0.1
)

def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

with open("sample/processed/keyframes/sample_video.json", "w", encoding="utf-8") as f:
    json.dump(asdict(keyframe_result), f, default=default_serializer, ensure_ascii=False, indent=4)

print("SUCCESS: Keyframes extracted and saved to sample/processed/keyframes/sample_video.json")
