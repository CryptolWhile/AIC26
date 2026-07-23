from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import numpy as np

from src.services.embedding.interface import EmbeddingModel

@dataclass
class Shot:
    shot_index: int
    start_frame: int
    end_frame: int

@dataclass
class VideoShots:
    video_path: str
    video_fps: float
    shots: List[Shot]

@dataclass
class Keyframe:
    frame_index:int
    timestamp: float

@dataclass
class SelectionResult:
    video_path: str
    keyframes: List[Keyframe]
    keyframe_ratio: float
    created_at: Optional[datetime] = None

@dataclass
class DistillationResult:
    video_path: str
    original_keyframes: List[Keyframe]
    distilled_keyframes: List[Keyframe]
    threshold: float
    created_at: Optional[datetime] = None

@dataclass
class KeyframeResult:
    video_path: str
    original_keyframes: List[Keyframe]
    distilled_keyframes: List[Keyframe]
    keyframe_ratio: float
    threshold: float
    created_at: Optional[datetime] = None
