from PIL import Image
from typing import List, Union, Optional, Literal
import logging

import torch
import numpy as np
from transformers import AutoTokenizer, CLIPImageProcessor, AutoModel

from src.services.embedding.interface import EmbeddingType, EmbeddingResult, EmbeddingModel
from src.services.embedding.utils import disable_flash_attn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HuggingFaceInternVLEmbedding(EmbeddingModel):
    """HuggingFace InternVL based embedding model implementation."""

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL-14B-Flickr30k-FT-364px",
        device_map: Optional[dict] = None,
        use_fast: bool = False,
    ):
        self.device_map = device_map
        self.use_fast = use_fast
        self.model_name = model_name
        self.model = None

        try:
            logger.info(f"Loading HuggingFace InternVL model: {self.model_name}")
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=self.device_map,
                offload_folder="offload_folder"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                use_fast=self.use_fast,
                add_eos_token=True
            )
            self.processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path=self.model_name)

            self.device = next(self.model.parameters()).device
            if not use_fast:
                disable_flash_attn(self.model)
                
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
                    batch = ['summarize:' + text for text in batch]

                    text_tokens = self.tokenizer(
                        batch, truncation=True, padding=True, return_tensors="pt"
                    ).to(self.device)
                    # InternVL specific input_ids manipulation
                    text_tokens.input_ids = text_tokens.input_ids.to(torch.bfloat16)

                    inputs = {
                        "input_ids": text_tokens.input_ids.to(torch.int64),
                        "attention_mask": text_tokens.input_ids > 0,
                        "output_attentions": False,
                        "output_hidden_states": False,
                        "return_dict": True
                    }

                    text_features = self.model.get_text_features(**inputs)
                    text_features = text_features[
                        torch.arange(text_features.shape[0], device=text_features.device),
                        inputs["attention_mask"].to(text_features.device).sum(1) - 1
                    ]
                    text_features = text_features @ self.model.text_projection.to(text_features.device)

                    if normalize:
                        text_features = torch.nn.functional.normalize(text_features, dim=-1)

                    embeddings_list.append(text_features.cpu())

                embeddings = torch.cat(embeddings_list, dim=0).to(torch.float32).numpy()
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
        mode: Literal["InternVL-C", "InternVL-G"] = "InternVL-G",
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
                    image_tensors = self.processor(images=batch, return_tensors="pt").pixel_values.to(torch.bfloat16).to(self.device)
                    image_features = self.model.encode_image(image_tensors, mode=mode)

                    if normalize:
                        image_features = torch.nn.functional.normalize(image_features, dim=-1)

                    embeddings_list.append(image_features.cpu())

                embeddings = torch.cat(embeddings_list, dim=0).to(torch.float32).numpy()
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
            sample.input_ids = sample.input_ids.to(torch.int64)
            inputs = {
                "input_ids": sample.input_ids,
                "attention_mask": sample.input_ids > 0,
            }
            features = self.model.get_text_features(**inputs)
            return features.shape[-1]

    def get_model_name(self) -> str:
        return self.model_name

    def get_support_types(self) -> List[EmbeddingType]:
        return [EmbeddingType.IMAGE, EmbeddingType.TEXT]