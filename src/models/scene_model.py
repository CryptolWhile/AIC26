from typing import Tuple, Dict, Any
from pydantic import BaseModel, Field

class Scene(BaseModel):
    """Model thể hiện một scene được trích xuất từ video."""
    
    id: str = Field(..., description="Unique identifier of the scene", examples=["L01_V001_S001", "L01_V001_S002"])
    path: str = Field(..., description="Path to the scene file in the system", examples=["/resource/scenes/L01_V001.mp4"])
    transcript: str = Field(default="", description="Mô tả bằng văn bản về nội dung scene", examples=["A person walking in the park", "Cars moving on the highway"])
    start_time: float = Field(
        ...,
        description="Start timestamp (in seconds) of the scene in the original video",
        examples=[15.5, 45.5]
    )
    end_time: float = Field(
        ...,
        description="End timestamp (in seconds) of the scene in the original video",
        examples=[30.2, 60.7]
    )
    # time: Tuple[float, float] = Field(..., description="Thời điểm scene trong video gốc (tính bằng giây)", examples=[(15.5, 30.2), (45.5, 60.7)])
    is_deleted: bool = Field(default=False, description="Cờ cho biết cảnh có được đánh dấu là đã xóa hay không")
    is_processed: bool = Field(default=False, description="Cờ cho biết liệu scene has been processed by downstream pipelines")

    class Config:
        """Configuration for the model."""
        json_schema_extra = {
            "example": {
                "id": "L01_V001_S001",
                "path": "/resource/scenes/L01_V001.mp4",
                "transcript": "A person walking in the park",
                "start_time": 15.5,
                "end_time": 30.2,
                # "time": (15.5, 30.2),
                "is_deleted": False,
                "is_processed": True
            }
        }