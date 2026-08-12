from typing import Dict, Optional, List, Type, Any

from src.services.llm.interface import ChatModel
from src.services.llm.implementations import (
    OpenAIChatModelProvider,
    GeminiChatModelProvider,
    FPTChatModelProvider,
    AzureChatModelProvider
)
from src.services.llm.interface import ModelCapability
    

class ChatModelManager:
    """Manages different chat providers"""
    
    def __init__(self):
        self.providers: Dict[str, Type[ChatModel]] = {}
        self.initialize_providers()
    
    def initialize_providers(self) -> None:
        """Initialize all available models"""
        self.register_provider("openai", OpenAIChatModelProvider)        
        self.register_provider("gemini", GeminiChatModelProvider)
        self.register_provider("azure", AzureChatModelProvider)
        self.register_provider("fpt", FPTChatModelProvider)
    
    def register_provider(self, provider_name: str, provider: Type[ChatModel]) -> None:
        """Register a new LLM model"""
        self.providers[provider_name] = provider
    
    def create_model(self, config: Optional[Dict[str, Any]] = None, provider_name: Optional[str] = None) -> ChatModel:
        config = config or {}
        if provider_name:
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not found")
            return self.providers[provider_name](**config)
    
    def list_providers(self) -> Dict[str, List[ModelCapability]]:
        """List all available models and their capabilities"""
        return {
            name: model.get_support_capabilities() 
            for name, model in self.providers.items()
        }