import time
from pathlib import Path
from typing import Union

from src.services.processing.shot_extraction.models.autoshot.autoshot import Autoshot
from src.services.processing.shot_extraction.interface import ExtractionMethod, Shot, ShotResult, ShotExtractor
from src.services.processing.shot_extraction.utils import get_current_time, get_fps


class AutoshotExtractor(ShotExtractor):
    
    def __init__(self):
        self.model = Autoshot()
    
    @classmethod
    def get_method_name(cls) -> ExtractionMethod:
        return ExtractionMethod.AUTOSHOT
    
    def extract_shots_from_video(
        self, 
        video_path: Union[str, Path], 
        threshold: float = 0.293
    ) -> ShotResult:
        
        try:
            video_path = Path(video_path)
            
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            video_fps = get_fps(video_path)
            
            predictions = self.model.predict_video(video_path)
            
            shot_boundaries = self.model.predictions_to_scenes(
                predictions=predictions, 
                threshold=threshold
            ).tolist()
            
            video_shots = []
            for shot_index, shot_boundary in enumerate(shot_boundaries, start=1):
                start_frame = int(shot_boundary[0])
                end_frame = int(shot_boundary[1])
                
                video_shot = Shot(
                    shot_index=shot_index,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=round(start_frame / video_fps, 2) if video_fps > 0 else 0,
                    end_time=round(end_frame / video_fps, 2) if video_fps > 0 else 0,
                    duration=round((end_frame - start_frame) / video_fps, 2) if video_fps > 0 else 0
                )
                video_shots.append(video_shot)
            
            return ShotResult(
                video_path=str(video_path),
                shots=video_shots,
                threshold=threshold,
                created_at=get_current_time()
            )
            
        except Exception as e:
            error_msg = f"Failed to extract shots from {video_path}: {str(e)}"
            raise RuntimeError(error_msg) from e