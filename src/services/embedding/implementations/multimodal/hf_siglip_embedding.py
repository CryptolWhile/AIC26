from PIL import Image
from typing import List, Union
import logging

import torch
import numpy as np
from transformers import AutoTokenizer, AutoProcessor, AutoModel

from src.services.embedding.interface import EmbeddingType, EmbeddingResult, EmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HuggingFaceSigLIPEmbedding(EmbeddingModel):
    """HuggingFace SigLIP based embedding model implementation."""
    
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384", 
        device: Union[str, torch.device] = "cpu" 
    ):
        self.device = device
        self.model_name = model_name
        self.model = None
        
        try:
            logger.info(f"Loading HuggingFace SigLIP model: {self.model_name}")
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            self.model.eval()
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
        
        embeddings_list = []
        
        try:
            with torch.no_grad():
                for index in range(0, len(texts), batch_size):
                    batch = texts[index : index + batch_size]
                    text_tokens = self.tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(self.device)
                    text_features = self.model.get_text_features(**text_tokens)
                    if normalize:
                        text_features = torch.nn.functional.normalize(text_features, dim=-1)
                    embeddings_list.append(text_features.cpu())
            
                embeddings = torch.cat(embeddings_list, dim=0).numpy()
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
            
    def embed_image(
        self, 
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        batch_size: int = 32, 
        normalize: bool = True
    ) -> EmbeddingResult:
        """Get L2-normalized image embedding(s)."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not images:
            raise ValueError("Image input must be non-empty.")

        if isinstance(images, (str, Image.Image)):
            images = [images]
        
        processed_images = []
        for img in images:
            if isinstance(img, str):
                with Image.open(img) as im:
                    processed_images.append(im.convert("RGB"))
            elif isinstance(img, Image.Image):
                processed_images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
        embeddings_list = []        
        
        try:
            with torch.no_grad():
                for index in range(0, len(processed_images), batch_size):
                    batch = processed_images[index : index + batch_size]
                    image_tensors = self.processor(images=batch, return_tensors="pt").to(self.device)
                    image_features = self.model.get_image_features(**image_tensors)
                    if normalize:
                        image_features = torch.nn.functional.normalize(image_features, dim=-1)
                    embeddings_list.append(image_features.cpu())
            
                embeddings = torch.cat(embeddings_list, dim=0).numpy()
        except Exception as e:
            logger.error(f"Image embedding error: {str(e)}")
            raise

        return EmbeddingResult(
            embeddings=embeddings,
            dimension=embeddings.shape[1],
            model_name=self.model_name,
            input_type=EmbeddingType.IMAGE,
            metadata={
                "modality": "image", 
                "normalized": normalize
            }
        )
        
    def embed_multimodal(self):
        raise NotImplementedError("embed_multimodal is not implemented yet.")
        
    def get_embedding_dimension(self) -> int:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        with torch.no_grad():
            sample = self.tokenizer(["test"], return_tensors="pt").to(self.device)
            features = self.model.get_text_features(**sample)
            return features.shape[1]
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_support_types(self) -> List[EmbeddingType]:
        return [EmbeddingType.IMAGE, EmbeddingType.TEXT]