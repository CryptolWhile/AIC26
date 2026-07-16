from PIL import Image
from typing import List, Any, Union
import logging
import torch
import numpy as np
from open_clip import create_model_and_transforms, get_tokenizer

from src.services.embedding.interface import EmbeddingType, EmbeddingResult, EmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpenCLIPEmbedding(EmbeddingModel):
    """OpenCLIP-based embedding model implementation."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None

        try:
            logger.info(f"Loading OpenClip model : {self.model_name} ({self.pretrained})")
            self.model, _, self.processor = create_model_and_transforms(
                model_name=self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.model.eval()
            self.tokenizer = get_tokenizer(self.model_name)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Model load error: {str(e)}")
            raise RuntimeError(f"Failed to load model: {e}")

    def get_embedding_dimension(self) -> int:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            return self.model.text_projection.shape[1]
        elif hasattr(self.model, "visual") and hasattr(self.model.visual, "output_dim"):
            return self.model.visual.output_dim
        return 512

    def get_support_types(self) -> List[EmbeddingType]:
        return [EmbeddingType.TEXT, EmbeddingType.IMAGE, EmbeddingType.MULTIMODAL]

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

        text_list = [texts] if isinstance(texts, str) else texts
        embeddings_list = []

        try:
            with torch.no_grad():
                for index in range(0, len(text_list), batch_size):
                    batch = text_list[index : index + batch_size]
                    text_tokens = self.tokenizer(batch).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
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

        image_list = [images] if not isinstance(images, list) else images
        processed_images = []
        for img in image_list:
            if isinstance(img, str):
                processed_images.append(Image.open(img).convert("RGB"))
            else:
                processed_images.append(img.convert("RGB"))

        embeddings_list = []

        try:
            with torch.no_grad():
                for index in range(0, len(processed_images), batch_size):
                    batch = processed_images[index : index + batch_size]
                    tensor_batch = torch.stack([self.processor(img) for img in batch]).to(self.device)
                    image_features = self.model.encode_image(tensor_batch)
                    
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
        with torch.no_grad():
            sample = self.tokenizer(["test"]).to(self.device)
            features = self.model.encode_text(sample)
            return features.shape[1]
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_support_types(self) -> List[EmbeddingType]:
        return [EmbeddingType.IMAGE, EmbeddingType.TEXT]