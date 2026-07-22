from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any
from PIL import Image
import numpy as np

class EmbeddingType(Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"

@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    dimension: int
    model_name: str
    input_type: EmbeddingType
    metadata: Optional[dict] = None

class EmbeddingModel(ABC):
     """Abstract base class for all embedding providers"""

     #group getting method
     @abstractmethod
     def get_embedding_dimension(self) -> int:
         """Return the dimension of embeddings produced by this provider"""
         pass
     
     @abstractmethod
     def get_support_types(self) -> List[EmbeddingType]:
        """Return list of supported embedding types"""
        pass
    
     @abstractmethod
     def embed_text(self, texts: Union[str, List[str]], batch_size: int, normalize: bool) -> EmbeddingResult:
         """Generate text embeddings for given inputs"""
         pass
     
     @abstractmethod
     def embed_image(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]], batch_size: int = 32, normalize: bool = True) -> EmbeddingResult:
         """Generate image embeddings for given inputs"""
         pass
