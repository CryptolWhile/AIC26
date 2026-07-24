from typing import Optional, Dict, Any

from src.services.vision.ocr.manager import OCRManager
from src.services.vision.ocr.interface import (
    OCRResult, OCRModel
)


class OCRService:
    
    def __init__(self):
        self.manager = OCRManager()
        self.model_registry: Dict[str, OCRModel] = {}
    
    def register_model(
        self, 
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None
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
    
    def get_model(self, model_name: str) -> OCRModel:
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' has not been registered")
        return self.model_registry[model_name]
    
    def get_available_providers(self) -> Dict[str, str]:
        return self.manager.list_providers()
    
    def get_available_models(self) -> Dict[str, Any]:
        return dict(self.model_registry)
    
    def extract_text_from_image(
        self, 
        image_path: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        model = self.get_model(model_name=model_name)
        return model.extract_text_from_image(image_path=image_path, **kwargs)