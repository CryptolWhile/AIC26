from typing import Optional, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.services.processing.keyframe_extraction.interface import (
    Keyframe,
    EmbeddingModel,
    DistillationResult
)
from src.services.processing.keyframe_extraction.utils import calculate_similarity, get_current_time


class KeyframeDistillator:
    
    def __init__(self):
        super().__init__()
    
    def generate_embeddings(
        self,
        video_path: str,
        keyframes: List[Keyframe],
        embedding_model: EmbeddingModel,
        batch_size=32,
        **kwargs
    ) -> List[np.ndarray]:
        """Generate embeddings for keyframes from a video."""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        embeddings: List[np.ndarray] = []

        try:
            for i in tqdm(range(0, len(keyframes), batch_size), desc="Generating embeddings"):
                batch_images = []
                
                for keyframe in keyframes[i:i+batch_size]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe.frame_index)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        raise RuntimeError(
                            f"Failed to read frame {keyframe.frame_index} in video"
                        )
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_images.append(Image.fromarray(frame_rgb))

                result = embedding_model.embed_image(
                    images=batch_images,
                    normalize=True,
                    **kwargs
                )
                embeddings.extend(result.embeddings)
            
            return embeddings

        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {e}") from e

        finally:
            cap.release()
    
    def distill_keyframes(
        self,
        video_path: str,
        keyframes: List[Keyframe],
        embedding_model: EmbeddingModel,
        compare_length: int = 2,
        threshold: float = 0.9
    ) -> DistillationResult:
        """Remove similar keyframes based on embedding similarity"""
        
        sorted_keyframes = sorted(keyframes, key=lambda x: x.frame_index)
        
        distilled_keyframes: List[Keyframe] = []
        distilled_embeddings: List[np.ndarray] = []
        
        try:
            embeddings = self.generate_embeddings(
                video_path=video_path,
                keyframes=sorted_keyframes, 
                embedding_model=embedding_model
            )
            
            for keyframe, embedding in tqdm(
                zip(sorted_keyframes, embeddings),
                total=len(sorted_keyframes),
                desc="Distilling keyframes"
            ):
                compare_embeddings = distilled_embeddings[-compare_length:]
                if len(compare_embeddings) == 0:
                    distilled_keyframes.append(keyframe)
                    distilled_embeddings.append(embedding)
                    continue
                
                adding_condition = True
                for compare_embedding in compare_embeddings:
                    similarity = calculate_similarity(compare_embedding, embedding)
                
                    if similarity > threshold:
                        adding_condition = False
                        break
                
                if adding_condition:
                    distilled_keyframes.append(keyframe)
                    distilled_embeddings.append(embedding)
            
            return DistillationResult(
                video_path=video_path,
                original_keyframes=keyframes,
                distilled_keyframes=distilled_keyframes,
                threshold=threshold,
                created_at=get_current_time()
            )
            
        except Exception as e:
            raise RuntimeError(f"Error during distillation: {e}") from e