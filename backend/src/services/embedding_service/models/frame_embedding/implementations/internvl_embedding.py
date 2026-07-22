import torch
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Literal
from ..base import FrameEmbeddingModel
from ....embedding_utils import disable_flash_attn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HuggingFaceInternVLEmbedding(FrameEmbeddingModel):

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'OpenGVLab/InternVL-14B-Flickr30K-FT-364px'
    ) -> None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is required for InternVLEmbedding but not available")
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self, device_map=None, use_fast: bool = False) -> None:
        from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
            offload_folder="offload_folder"
        ).eval()

        if not use_fast:
            disable_flash_attn(self.model)

        self.device = next(self.model.parameters()).device

        self.processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            use_fast=False,
            add_eos_token=True
        )
        self.tokenizer.pad_token_id = 0

    def _process_image(self, image_list: List[Union[str, Path, Image.Image]]) -> torch.Tensor:
        """Convert input images to RGB PIL format."""
        images = []
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

            images.append(img)

        pixel_values = self.processor(
            images=images,
            return_tensors='pt'
        ).pixel_values.to(torch.bfloat16).to(self.device)

        return pixel_values

    def get_image_embedding(
        self,
        image: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        mode: Literal['InternVL-C', 'InternVL-G'] = 'InternVL-G'
    ) -> np.ndarray:

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model()")

        image_list = [image] if not isinstance(image, list) else image
        try:
            with torch.no_grad():
                logit_scale = self.model.logit_scale.exp()
                inputs = self._process_image(image_list)
                features = self.model.encode_image(inputs, mode=mode)
                features /= features.norm(dim=1, keepdim=True)
                return features.cpu().to(dtype=torch.float32).numpy().astype(np.float64)
        except Exception as e:
            logger.error(f"Image embedding error: {str(e)}")
            raise

    def get_text_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """Get L2-normalized text embedding(s)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model().")
        if not text or not isinstance(text, (str, list)):
            raise ValueError("Text input must be a non-empty string or list.")

        prefix = 'summarize:'
        text_list = [text] if isinstance(text, str) else text
        text_list = [prefix + text for text in text_list]

        try:
            with torch.no_grad():
                input_ids = self.tokenizer(
                    text_list, return_tensors='pt', max_length=80,
                    truncation=True, padding='max_length'
                ).input_ids.to(self.device)
                attention_mask = input_ids > 0
                text_embeds = self.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                text_embeds = text_embeds[
                    torch.arange(text_embeds.shape[0], device=text_embeds.device),
                    attention_mask.to(text_embeds.device).sum(1) - 1
                ]
                features = text_embeds @ self.model.text_projection.to(text_embeds.device)
                features /= features.norm(dim=1, keepdim=True)
                return features.to(dtype=torch.float32).cpu().numpy().astype(np.float64)
        except Exception as e:
            logger.error(f"Text embedding error: {str(e)}")
            raise