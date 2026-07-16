from typing import List, Dict, Any
from pydantic import BaseModel, Field

class Keyframe(BaseModel):
    """Model thể hiện một key frame được trích xuất từ một scene video."""

    id: str = Field(..., description="Mã định danh duy nhất của key frame", examples=["L01_V001_S001_SH001_K001", "L01_V001_S001_SH002_K002"])
    path: str = Field(..., description="Đường dẫn đến tệp hình ảnh key frame", examples=["/resource/keyframes/L01_V001_S001_SH001_K001.jpg"])
    timestamp: float = Field(..., description="Thời điểm xuất hiện keyframe trong video gốc (tính bằng giây)", exmaples=[10.5, 45.7])
    ocr: List[str] = Field(
        default=[],
        description="List văn bản được trích xuất từ keyframe bằng OCR",
        examples=[["Chapter 1: Introduction", "Machine Learning Basics"], ["Equation: E = mc²"]]
    )
    od: List[Dict[str, Any]] = Field(
        default=[],
        description="List các đối tượng được phát hiện cùng với bounding box và confidence scores",
        examples=[[
            {"label": "person", "confidence": 0.95, "bbox": [100, 200, 300, 400]},
            {"label": "laptop", "confidence": 0.87, "bbox": [50, 150, 200, 250]}
        ]]
    )
    description: str = Field(..., description="Mô tả ngữ nghĩa của nội dung hình ảnh của keyframe")
    is_deleted: bool = Field(default=False, description="Trạng thái xóa của keyframe")
    is_processed: bool = Field(default=False, description="Trạng thái xử lý của keyframe")

    class Config:
        """Configuration for the model."""
        json_schema_extra = {
            "example": {
                "id": "L01_V001_S001_SH001_K001",
                "path": "/resource/keyframes/L01_V001_S001_SH001_K001.jpg",
                "timestamp": 10.5,
                "ocr": ["Chapter 1: Introduction", "Machine Learning Basics"],
                "od": [
                    {
                        "label": "person",
                        "confidence": 0.95,
                        "bbox": [100, 200, 300, 400]
                    }
                ],
                "description": "A keyframe showing a person and a laptop",
                "is_deleted": False,
                "is_processed": True
            }
        }