from typing import Dict, Optional, Type, Any

from src.services.processing.shot_extraction.interface import ExtractionMethod, ShotExtractor
from src.services.processing.shot_extraction.implementations import AutoshotExtractor, TransNetV2Extractor


class ShotExtractionManager:
    """Manages different shot extraction providers"""
    
    def __init__(self):
        self.providers: Dict[str, Type[ShotExtractor]] = {}
        self.initialize_providers()
    
    def initialize_providers(self) -> None:
        self.register_provider("autoshot", AutoshotExtractor)
        self.register_provider("transnetv2", TransNetV2Extractor)
    
    def register_provider(self, provider_name: str, provider: Type[ShotExtractor]) -> None:
        self.providers[provider_name] = provider
        
    def create_extractor(
            self, 
            config: Optional[Dict[str, Any]] = None, 
            provider_name: Optional[str] = None
        ) -> ShotExtractor:
        
        config = config or {}
        if provider_name:
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not found")
            return self.providers[provider_name](**config)
        
        raise ValueError("Either provider name or embedding type must be specified")
    
    def get_provider(self, provider_name: Optional[str] = None) -> Type[ShotExtractor]:
        if provider_name:
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not found")
            return self.providers[provider_name]
    
    def list_providers(self) -> Dict[str, ExtractionMethod]:
        return {
            name: provider.get_method_name() 
            for name, provider in self.providers.items()
        }