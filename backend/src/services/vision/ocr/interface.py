from typing import Optional, Tuple, List
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class OCRBox:
    text: str
    BBox: Tuple[int, int, int, int]
    confidence: Optional[float] = None

@dataclass
class OCRResult:
    image_path: str
    boxes: List[OCRBox] = field(default_factory=list)
    created_at: Optional[datetime] = field(default_factory=datetime.now)

class OCRModel(ABC):
    @abstractmethod
    def get_model_name(self) -> str:
        pass

    def extract_text_from_image(self, image_path: str) ->OCRResult:
        pass


    