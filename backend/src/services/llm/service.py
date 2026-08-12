from typing import Dict, Optional, Any, List

from src.services.llm.interface import ChatModel
from src.services.llm.manager import ChatModelManager
from langchain_core.messages import AIMessage


class ChatModelService:
    """Main service class for chat operations"""
    
    def __init__(self):
        self.manager = ChatModelManager()
        self.model_registry : Dict[str, ChatModel] = {}
        
    def register_model(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name is required for registration")
        
        if not provider_name:
            raise ValueError("provider_name is required to create a model")

        if model_name in self.model_registry:
            raise ValueError(f"Model '{model_name}' is already registered")
                
        self.model_registry[model_name] = self.manager.create_model(
            config=config,
            provider_name=provider_name
        )
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> AIMessage:
        model = self.get_model(model_name=model_name)
        
        return model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=image_paths
        )
    
    def get_model(self, model_name: str) -> ChatModel:
        """Get a registered model instance by name."""
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' has not been registered")
        
        return self.model_registry[model_name]
    
    def get_available_providers(self):
        """Get list of all available providers"""
        return self.manager.list_providers()

    def get_available_models(self) -> dict[str, Any]:
        """Return all registered models with their instances."""
        return dict(self.model_registry)