from typing import Dict, Union, Optional, Any, List
import logging
import torch
from PIL import Image

from src.services.embedding.interface import EmbeddingModel, EmbeddingResult
from src.services.embedding.manager import EmbeddingManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EmbeddingService:
    """Main service class for managing and running embedding operations."""

    def __init__(self):
        self.manager = EmbeddingManager()
        self.model_registry: Dict[str, EmbeddingModel] = {}

    def create_config(self, model_name: str) -> Dict[str, Any]:
        """Get default configuration for known models."""
        configs = {
            "hf_clip_L": {
                "provider_name": "hf_clip_multimodal",
                "model_name": "CLIP-ViT-L14",
                "config": {
                    "model_name": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
            },
            "hf_clip_H": {
                "provider_name": "hf_clip_multimodal",
                "model_name": "CLIP-ViT-H14",
                "config": {
                    "model_name": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
            },
            "hf_siglip": {
                "provider_name": "hf_siglip_multimodal",
                "model_name": "SIGLIP-SO400M",
                "config": {
                    "model_name": "google/siglip-so400m-patch14-384",
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
            }
        }
        if model_name not in configs:
            raise ValueError(f"Unknown predefined config for model '{model_name}'")
        return configs[model_name]
    
    def check_registered_model(self, model_name: str) -> bool:
        """Check if a model is already registered in the service."""
        return model_name in self.model_registry
    
    def registry_model(
        self, 
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None
    ) -> None:
        """Register and instantiate a model via the EmbeddingManager.
        
        Args:
            config: Configuration to pass to the model.
            model_name: The internal registry name for this model instance.
            provider_name: The name of the underlying provider in EmbeddingManager.
        """
        if not model_name:
            raise ValueError("model_name là điều kiện bắt buộc để đăng ký (is required for registration)")
        
        if not provider_name:
            raise ValueError("provider_name là điều kiện bắt buộc để đăng ký (is required for registration)")

        if self.check_registered_model(model_name):
            logger.info(f"Model '{model_name}' đã đăng ký, bỏ qua bước đăng ký.")
            return

        try:
            logger.info(f"Registering model '{model_name}' via provider '{provider_name}'...")
            self.model_registry[model_name] = self.manager.create_model(
                config=config,
                provider_name=provider_name
            )
            logger.info(f"Model '{model_name}' registered successfully.")
        except Exception as e:
            logger.error(f"Failed to register model '{model_name}': {str(e)}")
            raise

    def get_model(self, model_name: str) -> EmbeddingModel:
        """Get a registered model instance by name.
        
        Args:
            model_name: The registered name of the model.
            
        Returns:
            The EmbeddingModel instance.
        """
        if not model_name:
            raise ValueError("model_name must be provided.")
            
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' chưa được đăng ký (is not registered)")

        return self.model_registry[model_name]
    
    def embed_text(
        self, 
        texts: Union[str, List[str]],
        model_name: Optional[str] = None,
        batch_size: int = 32, 
        normalize: bool = True,
        **kwargs
    ) -> EmbeddingResult:
        """Generate text embeddings.
        
        Args:
            texts: A string or list of strings to embed.
            model_name: The name of the registered model to use.
            batch_size: Batch size for processing.
            normalize: Whether to apply L2 normalization.
        """
        model = self.get_model(model_name=model_name)
        
        try:
            return model.embed_text(
                texts=texts,
                batch_size=batch_size,
                normalize=normalize,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Service text embedding error with model '{model_name}': {str(e)}")
            raise
    
    def embed_image(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        model_name: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs
    ) -> EmbeddingResult:
        """Generate image embeddings.
        
        Args:
            images: A string path, PIL Image, or list thereof.
            model_name: The name of the registered model to use.
            batch_size: Batch size for processing.
            normalize: Whether to apply L2 normalization.
        """
        model = self.get_model(model_name=model_name)

        try:
            return model.embed_image(
                images=images,
                batch_size=batch_size,
                normalize=normalize,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Service image embedding error with model '{model_name}': {str(e)}")
            raise
    
    def embed_multimodal(
        self,
        inputs: Any,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs
    ) -> EmbeddingResult:
        """Generate multimodal embeddings.
        
        Args:
            inputs: Multimodal inputs (text + images) specific to the model.
            model_name: The name of the registered model to use.
            batch_size: Batch size for processing.
            normalize: Whether to apply L2 normalization.
        """
        model = self.get_model(model_name=model_name)

        try:
            return model.embed_multimodal(
                inputs=inputs,
                batch_size=batch_size,
                normalize=normalize,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Service multimodal embedding error with model '{model_name}': {str(e)}")
            raise

    def get_available_providers(self) -> Dict[str, List[EmbeddingType]]:
        """Return a dictionary of all registered providers and their supported types."""
        try:
            return self.manager.list_providers()
        except Exception as e:
            logger.warning(f"Error fetching provider list (could be due to static method calls on classes): {str(e)}")
            return {}

    def get_available_models(self) -> Dict[str, Any]:
        """Return all models that have been instantiated and registered in this service."""
        return dict(self.model_registry)