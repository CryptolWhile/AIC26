from typing import List

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from src.services.llm.interface import ChatModel
from src.services.llm.interface import ModelCapability


class AzureChatModelProvider(ChatModel):
    """
    Azure Chat Model implementation using AzureAIChatCompletionsModel.
    """

    def __init__(
        self,
        api_version: str,
        endpoint: str,
        api_key: str,
        model_name: str = "DeepSeek-R1-0528",
        temperature: float = 0.9,
        max_tokens: int = 2048,
        max_retries: int = 3
    ):  
        self.api_version = api_version
        self.endpoint = endpoint
        self.api_key = api_key

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        
        self.client = None
        self.connect()

    def connect(self) -> None:
        """Initialize the Azure AI chat model client."""
        model_config = {
            "model": self.model_name,
            "api_version": self.api_version,
            "endpoint": self.endpoint,
            "credential": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
        }
        self.client = AzureAIChatCompletionsModel(**model_config)
    
    def get_model_name(self) -> str:
        """Return the chat model name used by this provider"""
        return self.model_name
    
    def get_support_capabilities(self) -> List[ModelCapability]:
        """Return list of supported chat model capability"""
        return [ModelCapability.TEXT_GENERATION]