import cv2
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Optional
from datetime import datetime


class ExtractionMethod(Enum):
    """Emun chứa model cut video"""
    AUTOSHOT = "autoshot"
    TRANSNETV2 = "transnetv2"
    

@dataclass
class Shot:
    shot_index: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    

@dataclass
class ShotResult:
    video_path: Path
    shots: List[Shot] = field(default_factory=list)
    threshold: Optional[float] = field(default=None)
    created_at: datetime = field(default_factory=datetime.now)


class ShotExtractor(ABC):
    """ Abstract base class for shot extraction models."""

    def get_support_formats(self) -> List[str]:
        return ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']

    @abstractmethod
    def get_method_name(self) -> ExtractionMethod:
        pass
    
    @abstractmethod
    def extract_shots_from_video(self, video_path: str, threshold: float) -> ShotResult:
        pass