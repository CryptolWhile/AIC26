import logging
from typing import Dict, Type
from ..models.base import EmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EmbeddingFatory(EmbeddingModel):
    """Factory class for creating embedding models."""
    _models: Dict[str, Type[EmbeddingModel]] = {}

    @classmethod
    def register_model(cls, model_type: str,  model_class: Type[EmbeddingModel]) -> None:
        """Register a new embedding model type.
        
        Args:
            model_type: Mã định danh kiểu cho mô hình (clip, bclip)
            model_class: Lớp triển khai giao diện EmbeddingModel (HuggingFaceBLIPEmbedding,...)
        """

        cls._models[model_type.lower()] = model_class
        logger.info(f"Registered model type: {model_type}")

    @classmethod
    def create_embedding_model(cls, model_type: str, **kwargs) -> EmbeddingModel:
        model_type = model_type.lower()
        if model_type not in cls._models[model_type]:
            raise ValueError(f"Unsupported model type: {model_type}. Available: {list(cls._models.keys())}")
        
        try:

            logger.info(f"Creating model of type: {model_type}")
            model = cls._models[model_type](**kwargs)
            model.load_model(**kwargs)
            return model
        
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise

    @classmethod
    def get_available_models(cls) -> list:
        return list(cls._models.keys())