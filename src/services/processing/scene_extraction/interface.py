from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Scene:
    """Data class representing a video scene."""
    scene_index: int
    start: float
    end: float
    transcript: str


@dataclass
class SceneResult:
    """Data class representing the segmentation result."""
    scenes: List[Scene]
    created_at: Optional[datetime] = None
    
    
@dataclass
class ExtractionResult:
    """Data class representing transcript extraction result."""
    segments: List[Dict[str, Any]]
    created_at: Optional[datetime] = None


@dataclass
class SegmentationResult:
    """Data class representing transcript processing result."""
    original_segments: List[Dict[str, Any]]
    processed_segments: List[Dict[str, Any]]
    created_at: Optional[datetime] = None