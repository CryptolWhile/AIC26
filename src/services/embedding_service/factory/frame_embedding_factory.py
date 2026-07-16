import logging
from typing import Type, Dict
from .embedding_factory import EmbeddingFactory
from ..models.frame_embedding import (
    OpenCLIPEmbedding, HuggingFaceCLIPEmbedding, HuggingFaceInternVLEmbedding, 
    HuggingFaceSigLIPEmbedding, HuggingFaceBLIPEmbedding, HuggingFaceBLIP2Embedding
)
from ..models.frame_embedding import FrameEmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FrameEmbeddingFactory(EmbeddingFactory):
    """Factory for creating frame embedding models."""
    
    _models: Dict[str, Type[FrameEmbeddingModel]] = {
        "open_clip": OpenCLIPEmbedding,
        "hf_clip": HuggingFaceCLIPEmbedding,
        "hf_internvl": HuggingFaceInternVLEmbedding,
        "hf_siglip": HuggingFaceSigLIPEmbedding,
        "hf_blip": HuggingFaceBLIPEmbedding,
        "hf_blip2": HuggingFaceBLIP2Embedding
    }

    def __init__(self):
        super().__init__()

    @classmethod
    def create_embedding_model(cls, model_type: str, **kwargs) -> FrameEmbeddingModel:
        """Create and initialize an embedding model instance.

        Args:
            model_type: Type of model to create
            **kwargs: Các thông số cấu hình dành riêng cho từng model

        Returns:
            Initialized embedding model instance

        Raises:
            ValueError: If model_type is not registered
            Exception: If model creation fails
        """
        return super().create_embedding_model(model_type=model_type, **kwargs)