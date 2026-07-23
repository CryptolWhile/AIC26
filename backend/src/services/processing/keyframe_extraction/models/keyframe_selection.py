import time
from typing import List, Optional
import numpy as np
from datetime import datetime

from src.services.processing.keyframe_extraction.interface import (
    Keyframe, VideoShots, SelectionResult
)
from src.services.processing.keyframe_extraction.utils import get_fps, get_current_time


class KeyframeSelector:
    def __init__(self, min_keyframes_per_shot: int = 1, max_keyframes_per_shot: int = 50):
        self.min_keyframes_per_shot = min_keyframes_per_shot
        self.max_keyframes_per_shot = max_keyframes_per_shot

    def select_keyframes_from_shots(
        self,
        video_shots: VideoShots,
        keyframe_ratio: float
    ) -> SelectionResult:

        all_keyframes = []
        
        for i, shot in enumerate(video_shots.shots):
            try:
                shot_frame_number = shot.end_frame - shot.start_frame + 1
                if shot_frame_number <= 0:
                    raise ValueError(f"Shot {i} has invalid frame range.")

                video_fps = get_fps(video_shots.video_path)

                keyframe_number = max(
                    self.min_keyframes_per_shot,
                    min(int(shot_frame_number * keyframe_ratio), self.max_keyframes_per_shot)
                )

                selected_frames = np.linspace(
                    shot.start_frame,
                    shot.end_frame,
                    num=keyframe_number,
                    dtype=np.uint32
                ).tolist()

                keyframes = [
                    Keyframe(
                        frame_index=int(frame_index),
                        timestamp=round(frame_index / video_fps, 2),
                    )
                    for frame_index in selected_frames
                ]

                all_keyframes.extend(keyframes)

            except Exception as e:
                raise RuntimeError(f"Error processing shot {i}: {e}") from e

        return SelectionResult(
            video_path=video_shots.video_path,
            keyframes=all_keyframes,
            keyframe_ratio=keyframe_ratio,
            created_at=get_current_time()
        )