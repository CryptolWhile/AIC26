from typing import Union, List
import logging

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from src.services.embedding.models.beit3.load_mode import load_beit3_model
from src.services.embedding.interface import EmbeddingModel, EmbeddingResult, EmbeddingType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UnilmBEIT3Embedding(EmbeddingModel):
    """Unilm BEiT-3 based embedding model implementation."""
    
    def __init__(
        self,
        model_name: str = "beit_large_patch16_384_retrieval",
        device: Union[str, torch.device] = "cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.model = None
        
        try:
            logger.info(f"Loading Unilm BEiT-3 model: {self.model_name}")
            self.tokenizer, self.model = load_beit3_model(model_name=model_name)
            self.model.to(device).eval()
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
        
        embedding_list = []
        
        try:
            with torch.no_grad():
                for index in range(0, len(texts), batch_size):
                    batch = texts[index : index + batch_size]
                    text_tokens = self.tokenizer(
                        text=batch,
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    )["input_ids"].to(self.device)
                    
                    _, text_features = self.model(
                        text_description=text_tokens,
                        only_infer=True
                    )
                    if normalize:
                        text_features = torch.nn.functional.normalize(text_features, dim=-1)
                    embedding_list.append(text_features.cpu())
                    
            embeddings = torch.cat(embedding_list, dim=0).numpy()
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
        normalize: bool = True,
        image_size: int = 224
    ) -> EmbeddingResult:
        """Get L2-normalized image embedding(s)."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not images:
            raise ValueError("Image input must be non-empty.")

        if isinstance(images, (str, Image.Image)):
            images = [images]
            
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                            interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        
        processed_images = []
        for img in images:
            if isinstance(img, str):
                with Image.open(img) as im:
                    im = transform(im.convert("RGB")).unsqueeze(0)
                    processed_images.append(im)
            elif isinstance(img, Image.Image):
                img = transform(img.convert("RGB")).unsqueeze(0)
                processed_images.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        embedding_list = []
        
        try:
            processed_images = torch.cat(processed_images, dim=0).to(self.device)
            
            with torch.no_grad():
                for index in range(0, len(images), batch_size):
                    batch = processed_images[index : index + batch_size]
                    
                    image_features, _ = self.model(
                        image=batch,
                        only_infer=True
                    )
                    if normalize:
                        image_features = torch.nn.functional.normalize(image_features, dim=-1)
                    embedding_list.append(image_features.cpu())
                    
            embeddings = torch.cat(embedding_list, dim=0).numpy()
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
            sample = self.tokenizer(["text"], return_tensors="pt")["input_ids"].to(self.device)
            _, features = self.model(
                text_description=sample,
                only_infer=True
            )
            return features.shape[1]
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_support_types(self) -> List[EmbeddingType]:
        return [EmbeddingType.IMAGE, EmbeddingType.TEXT]