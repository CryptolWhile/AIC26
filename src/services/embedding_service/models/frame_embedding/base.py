from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from PIL import Image
from pathlib import Path
from ..base import EmbeddingModel

class FrameEmbeddingModel(EmbeddingModel):
    """Base interface for frame embedding models."""

    @abstractmethod
    def get_image_embedding(self, image: Union[str, Image.Image, Path]) -> np.ndarray:
        """Generate normalized embedding vector for an image.

        Args:
            image: Input image as file path, Path object or PIL Image

        Returns:
            np.ndarray: L2-normalized embedding vector (float32)
        """
        pass

    @abstractmethod
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate normalized embedding vector for text.

        Args:
            text: Input text string

        Returns:
            np.ndarray: L2-normalized embedding vector (float32)
        """
        pass
    
    def get_embedding_type(self) -> str:
        return "FrameEmbeddingModel"