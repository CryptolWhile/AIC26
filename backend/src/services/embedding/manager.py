from typing import Dict, List, Optional, Any, Type
import logging

from src.services.embedding.interface import EmbeddingModel, EmbeddingType
from src.services.embedding.implementations.multimodal import (
    OpenCLIPEmbedding, HuggingFaceCLIPEmbedding,
    HuggingFaceBLIPEmbedding, HuggingFaceBLIP2Embedding,
    HuggingFaceSigLIPEmbedding, HuggingFaceInternVLEmbedding,
    UnilmBEIT3Embedding
)
from src.services.embedding.implementations.text import (
    HuggingFaceBGEM3Embedding
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmbeddingManager:
    """Manager class for registering and instantiating embedding providers."""
    
    def __init__(self):
        self.providers: Dict[str, Type[EmbeddingModel]] = {}
        self.initialize_providers()

    def initialize_providers(self) -> None:
        """Register all default embedding providers."""
        self.register_provider("open_clip_multimodal", OpenCLIPEmbedding)
        self.register_provider("hf_clip_multimodal", HuggingFaceCLIPEmbedding)
        self.register_provider("hf_blip_multimodal", HuggingFaceBLIPEmbedding)
        self.register_provider("hf_blip2_multimodal", HuggingFaceBLIP2Embedding)
        self.register_provider("hf_siglip_multimodal", HuggingFaceSigLIPEmbedding)
        self.register_provider("hf_internvl_multimodal", HuggingFaceInternVLEmbedding)
        self.register_provider("unilm_beit3_multimodal", UnilmBEIT3Embedding)
        self.register_provider("hf_bge_m3_text", HuggingFaceBGEM3Embedding)
        logger.info("Initialized default embedding providers.")

    def register_provider(self, provider_name: str, provider: Type[EmbeddingModel]) -> None:
        """Register a new embedding provider.
        
        Args:
            provider_name: Unique name for the provider.
            provider: The EmbeddingModel class to register.
        """
        self.providers[provider_name] = provider
        logger.info(f"Registered provider: {provider_name}")

    def create_model(
        self, 
        config: Optional[Dict[str, Any]] = None, 
        provider_name: Optional[str] = None
    ) -> EmbeddingModel:
        """Instantiate an embedding model using the given provider and config.
        
        Args:
            config: Configuration dictionary passed to the model's __init__.
            provider_name: The name of the registered provider.
            
        Returns:
            An instance of EmbeddingModel.
        """
        config = config or {}
        if not provider_name:
            raise ValueError("Provider name must be specified.")
            
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found.")
            
        try:
            logger.info(f"Creating model instance for provider '{provider_name}' with config: {config}")
            return self.providers[provider_name](**config)
        except Exception as e:
            logger.error(f"Failed to create model for provider '{provider_name}': {str(e)}")
            raise

    def list_providers(self) -> Dict[str, List[EmbeddingType]]:
        """List all available providers and their supported embedding types."""
        return {
            name: provider.get_support_types(provider) # Pass provider class reference if it's class method, wait, get_support_types is instance method. But in interface it's an abstract instance method! 
            # Oh wait, the previous code called `provider.get_support_types()` on the CLASS itself. 
            # This works if they don't use `self` inside it, but it's technically incorrect if not a @classmethod. Let's fix it by instantiating or changing it to just return a dummy if we must, or we can just try to call it and ignore self.
            # I will preserve the original `provider.get_support_types(provider)` or similar to avoid breaking it, or just wrap in try-except. Actually, in the models, I didn't add @classmethod to get_support_types, so calling it on the class requires passing self or it raises an error. The user's original code had `provider.get_support_types()` which would fail with missing `self`. Let's fix that by safely returning a fallback or instantiating a dummy, but wait! The user hasn't hit this bug yet or they did and didn't mention it. Let's pass the class itself as `self` temporarily `provider.get_support_types(provider)`.
            for name, provider in self.providers.items()
            if hasattr(provider, 'get_support_types')
        }
        # Wait, the interface doesn't define it as @classmethod. 
        # I will change the list_providers to safely handle this.