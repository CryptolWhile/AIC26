from typing import Union, List
import logging

import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from src.services.embedding.interface import EmbeddingType, EmbeddingResult, EmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HuggingFaceBGEM3Embedding(EmbeddingModel):
    """HuggingFace BGE-M3 based text embedding model implementation."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Union[str, torch.device] = "cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.model = None
        
        try:
            logger.info(f"Loading HuggingFace BGE-M3 model: {self.model_name}")
            self.model = BGEM3FlagModel(self.model_name, use_fp16=True)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Model load error: {str(e)}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def embed_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> EmbeddingResult:
        """Get L2-normalized text embedding(s)."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not texts:
            raise ValueError("Text input must be a non-empty string or list.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    sentences=texts, 
                    batch_size=batch_size, 
                    max_length=8192,
                )["dense_vecs"]

                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings)

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

                embeddings = embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Text embedding error: {str(e)}")
            raise

        return EmbeddingResult(
            embeddings=embeddings,
            dimension=embeddings.shape[1],
            model_name=self.model_name,
            input_type=EmbeddingType.TEXT,
            metadata={
                "modality": "text",
                "normalized": normalize
            }
        )
    
    def embed_image(self, *args, **kwargs):
        raise NotImplementedError("embed_image is not implemented for BGE-M3.")
    
    def embed_multimodal(self, *args, **kwargs):
        raise NotImplementedError("embed_multimodal is not implemented for BGE-M3.")
    
    def get_embedding_dimension(self) -> int:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        with torch.no_grad():
            embeddings = self.model.encode(
                sentences=["test"], 
                batch_size=1, 
                max_length=8192,
            )["dense_vecs"]

            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings)

            return int(embeddings.shape[1])
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_support_types(self) -> List[EmbeddingType]:
        return [EmbeddingType.TEXT]