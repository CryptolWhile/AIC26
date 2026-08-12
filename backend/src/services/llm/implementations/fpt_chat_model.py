from typing import List

from langchain_openai import ChatOpenAI

from src.services.llm.interface import ChatModel
from src.services.llm.interface import ModelCapability


class FPTChatModelProvider(ChatModel):
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.9,
        max_tokens: int = 2048,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = base_url
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        
        self.client = None
        self.connect()

    def connect(self) -> None:
        """Initialize the FPT AI chat model client."""
        model_config = {
            "model": self.model_name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries
        }
        self.client = ChatOpenAI(**model_config)
    
    def get_model_name(self) -> str:
        """Return the chat model name used by this provider"""
        return self.model_name
    
    def get_support_capabilities(self) -> List[ModelCapability]:
        """Return list of supported chat model capability"""
        return [ModelCapability.TEXT_GENERATION]