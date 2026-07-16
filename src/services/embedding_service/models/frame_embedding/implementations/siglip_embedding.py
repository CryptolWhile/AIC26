import torch
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union
from ..base import FrameEmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HuggingFaceSigLIPEmbedding(FrameEmbeddingModel):

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'google/siglip-so400m-patch14-384',
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Initialize with pretrained model name or path."""
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device

        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load Hugging Face model and preprocessing tools"""
        from transformers import AutoProcessor, AutoTokenizer, AutoModel
        try:
            logger.info(f"Loading Hugging Face model: {self.pretrained_model_name_or_path}")
            self.model = AutoModel.from_pretrained(self.pretrained_model_name_or_path).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        except Exception as e:
            logger.error(f"Model load error: {str(e)}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _process_image(self, image_list: List[Union[str, Path, Image.Image]]) -> torch.Tensor:
        """Convert input images to RGB PIL format"""
        inputs = []
        for img in image_list:
            if isinstance(img, (str, Path)):
                img_path = Path(img)
                if not img_path.exists():
                    raise FileNotFoundError(f"File not found: {img_path}")
                img = Image.open(img_path)

            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            inputs.append(img)

        inputs = self.processor(images=inputs, return_tensors="pt")

        return inputs.to(self.device)

    def get_image_embedding(self, image: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]) -> np.ndarray:
        """Get L2-normalized image embedding(s)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model()")

        image_list = [image] if not isinstance(image, list) else image
        try:
            with torch.no_grad():
                inputs = self._process_image(image_list)
                features = self.model.get_image_features(**inputs)
                features /= features.norm(dim=1, keepdim=True)
                return features.cpu().numpy().astype(np.float64)
        except Exception as e:
            logger.error(f"Image embedding error: {str(e)}")
            raise

    def get_text_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """Get L2-normalized text embedding(s)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model().")
        if not text or not isinstance(text, (str, list)):
            raise ValueError("Text input must be a non-empty string or list.")

        text_list = [text] if isinstance(text, str) else text

        try:
            with torch.no_grad():
                inputs = self.tokenizer(text_list, padding=True, return_tensors="pt").to(self.device)
                features = self.model.get_text_features(**inputs)
                features /= features.norm(dim=1, keepdim=True)
                return features.cpu().numpy().astype(np.float64)
        except Exception as e:
            logger.error(f"Text embedding error: {str(e)}")
            raise