from typing import Dict, Optional, Type, Any

from src.services.vision.ocr.interface import OCRModel
from src.services.vision.ocr.implementations import MMOCRVietOCRModel


class OCRManager:
    
    def __init__(self):
        self.providers: Dict[str, Type[OCRModel]] = {}
        self.initialize_providers()
    
    def initialize_providers(self) -> None:
        self.register_provider("mmocr", MMOCRVietOCRModel)
    
    def register_provider(self, provider_name: str, provider: Type[OCRModel]) -> None:
        self.providers[provider_name] = provider
    
    def create_model(self, config: Optional[Dict[str, Any]] = None, provider_name: Optional[str] = None) -> OCRModel:
        config = config or {}
        if provider_name:
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not found")
            return self.providers[provider_name](**config)
    
    def get_provider(self, provider_name: Optional[str] = None) -> Type[OCRModel]:
        if provider_name:
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not found")
            return self.providers[provider_name]
    
    def list_providers(self) -> Dict[str, str]:
        return {
            name: provider.get_model_name() 
            for name, provider in self.providers.items()
        }