import torch
import logging
import numpy as np
import torch.nn as nn

from PIL import Image
from pathlib import Path
from typing import List, Union, Optional
from transformers import Blip2Config, Blip2ForImageTextRetrieval
from ..base import FrameEmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HuggingFaceBLIPEmbedding(FrameEmbeddingModel):
    """Hugging Face CLIP-based embedding model implementation."""
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Salesforce/blip-itm-large-flickr",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Initialize with pretrained model name or path."""
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device

        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self) -> None:
        """Load Hugging Face model and preprocessing tools"""
        from transformers import AutoProcessor, BlipModel
        try:
            logger.info(f"Loading Hugging Face model: {self.pretrained_model_name_or_path}")
            self.model = BlipModel.from_pretrained(self.pretrained_model_name_or_path).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name_or_path)
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
                inputs = self.processor(text=text_list, padding=True, return_tensors="pt").to(self.device)
                features = self.model.get_text_features(**inputs)
                features /= features.norm(dim=1, keepdim=True)
                return features.cpu().numpy().astype(np.float64)
        except Exception as e:
            logger.error(f"Text embedding error: {str(e)}")
            raise

class CustomBlip2Model(Blip2ForImageTextRetrieval):

    def __init__(self, config: Blip2Config):
        super().__init__(config)

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )
        image_embeds = query_outputs[0] if not return_dict else query_outputs.last_hidden_state

        # normalized features
        image_embeds = nn.functional.normalize(self.vision_projection(image_embeds), dim=-1)

        return image_embeds

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        query_embeds = self.embeddings(
            input_ids=input_ids,
        )
        text_outputs = self.qformer(
            query_embeds=query_embeds,
            query_length=0,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = text_outputs[0] if not return_dict else text_outputs.last_hidden_state

        # normalized features
        text_embeds = nn.functional.normalize(self.text_projection(question_embeds[:, 0, :]), dim=-1)

        return text_embeds

class HuggingFaceBLIP2Embedding(FrameEmbeddingModel):
    """Hugging Face CLIP-based embedding model implementation."""
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Salesforce/blip2-itm-vit-g",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Initialize with pretrained model name or path."""
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device

        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self) -> None:
        """Load Hugging Face model and preprocessing tools"""
        from transformers import AutoProcessor, AutoTokenizer
        try:
            logger.info(f"Loading Hugging Face model: {self.pretrained_model_name_or_path}")
            self.model = CustomBlip2Model.from_pretrained(self.pretrained_model_name_or_path).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name_or_path)
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
                inputs = self.processor(text=text_list, return_tensors="pt", padding=True).to(self.device, torch.float16)
                features = self.model.get_text_features(**inputs)
                features /= features.norm(dim=1, keepdim=True)
                return features.cpu().numpy().astype(np.float64)
        except Exception as e:
            logger.error(f"Text embedding error: {str(e)}")
            raise