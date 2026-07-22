from typing import Optional, Dict, Any

from src.services.processing.shot_extraction.manager import ShotExtractionManager
from src.services.processing.shot_extraction.interface import (
    ShotResult, ShotExtractor, ExtractionMethod
)


class ShotExtractionService:
    
    def __init__(self):
        self.manager = ShotExtractionManager()
        self.extractor_registry: Dict[str, ShotExtractor] = {}
        
    def register_extractor(
        self,
        config: Optional[Dict[str, Any]] = None,
        extractor_name: Optional[str] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        if not extractor_name:
            raise ValueError("extractor_name is required for registration")
        
        if not provider_name:
            raise ValueError("provider_name is required to create a model")

        if extractor_name in self.extractor_registry:
            raise ValueError(f"Model '{extractor_name}' is already registered")
                
        self.extractor_registry[extractor_name] = self.manager.create_extractor(
            config=config,
            provider_name=provider_name
        )
    
    def get_extractor(self, extractor_name: str) -> ShotExtractor:
        if extractor_name not in self.extractor_registry:
            raise ValueError(f"Extractor '{extractor_name}' has not been registered")        
        return self.extractor_registry[extractor_name]
    
    def get_available_providers(self) -> Dict[str, ExtractionMethod]:
        return self.manager.list_providers()

    def get_available_extractors(self) -> Dict[str, Any]:
        return dict(self.extractor_registry)
    
    def extract_shots(
        self,
        video_path: str,
        extractor_name: Optional[str] = None,
        **kwargs
    ) -> ShotResult:
        extractor = self.get_extractor(extractor_name=extractor_name)
        shot_result = extractor.extract_shots_from_video(video_path=video_path, **kwargs)
        return shot_result

    def __call__(self, *args, **kwargs):
        return self.extract_shots(*args, **kwargs)