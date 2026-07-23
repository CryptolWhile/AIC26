from src.services.processing.keyframe_extraction.models import (
    KeyframeSelector,
    KeyframeDistillator
)
from src.services.processing.keyframe_extraction.interface import (
    VideoShots,
    KeyframeResult,
    EmbeddingModel
)
from src.services.processing.keyframe_extraction.utils import get_current_time


class KeyframeExtractionService:
    
    def __init__(
        self,
        min_keyframes_per_shot: int = 1,
        max_keyframes_per_shot: int = 50
    ):
        self.keyframe_selector = KeyframeSelector(
            min_keyframes_per_shot=min_keyframes_per_shot,
            max_keyframes_per_shot=max_keyframes_per_shot
        )
        self.keyframe_distillator = KeyframeDistillator()

    def extract_keyframes_from_shots(
        self, 
        video_shots: VideoShots,
        embedding_model: EmbeddingModel,
        keyframe_ratio: float = 0.1,
        compare_length: int = 2,
        threshold: float = 0.9
    ) -> KeyframeResult:
        
        selection_result = self.keyframe_selector.select_keyframes_from_shots(
            video_shots=video_shots,
            keyframe_ratio=keyframe_ratio
        )
        
        distillation_result = self.keyframe_distillator.distill_keyframes(
            video_path=selection_result.video_path,
            embedding_model=embedding_model,
            keyframes=selection_result.keyframes,
            compare_length=compare_length,
            threshold=threshold
        )
        
        return KeyframeResult(
            video_path=video_shots.video_path,
            original_keyframes=distillation_result.original_keyframes,
            distilled_keyframes=distillation_result.distilled_keyframes,
            keyframe_ratio=keyframe_ratio,
            threshold=threshold,
            created_at=get_current_time()
        )