from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.services.llm.utils import encode_image


class ModelCapability(Enum):
    TEXT_GENERATION = "text_generation"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"
    MULTIMODAL = "multimodal"


class ChatModel(ABC):
    def __init__(self):
        self.client: BaseChatModel = None

    @abstractmethod
    def connect(self):
        """Initialize the LLM client (LangChain wrapper)."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the chat model name used by this provider"""
        pass
    
    @abstractmethod
    def get_support_capabilities(self) -> List[ModelCapability]:
        """Return list of supported chat model capability"""
        pass

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Optional[List[str]] = None
    ) -> AIMessage:
        if self.client is None:
            self.connect()
            
        try:
            user_content = self.prepare_chat_input(user_prompt, image_paths)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
            response = self.client.invoke(messages)
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def prepare_chat_input(
        self,
        prompt: str,
        image_paths: Optional[str] = None
    ) -> List[Dict]:
        content = []

        if image_paths:
            for image_path in image_paths:
                encoded_image, img_format = encode_image(image_path, resize=True, size=(896, 896))
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{encoded_image}"
                    }
                })

        content.append({"type": "text", "text": prompt})
        return content