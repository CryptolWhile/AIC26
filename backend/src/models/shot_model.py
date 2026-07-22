from typing import Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

class Shot(BaseModel):
    """Model thể hiện một cảnh quay được trích xuất từ video."""

    id: str = Field(..., description="Mã định danh của shot", examples=["L01_V001_S001_SH001","L01_V001_S002_SH002"])
    path: str = Field(..., description="đường dẫn shot trong hệ thống", examples=["/resource/shots/L01_V001.mp4"])
    start_frame: int = Field(
        ...,
        description="Start frame index of the shot in the original video",
        examples=[100, 250]
    )
    end_frame: int = Field(
        ...,
        description="End frame index of the shot in the original video",
        examples=[200, 400]
    )
    start_time: float = Field(
        ...,
        description="Start timestamp (in seconds) of the shot",
        examples=[15.5, 45.5]
    )
    end_time: float = Field(
        ...,
        description="End timestamp (in seconds) of the shot",
        examples=[30.2, 60.7]
    )
    duration: Optional[float] = Field(
        None,
        description="Thời lượng của shot quay (giây) (end_time - start_time)",
        examples=[14.7, 15.2]
    )
    # time: Tuple[float, float] = Field(..., description="Thời điểm shot trong video gốc (tính bằng giây)", examples=[(15.5, 30.2), (45.5, 60.7)])
    is_deleted: bool = Field(default=False, description="Trạng thái xóa của shot")
    is_processed: bool = Field(default=False, description="Trạng thái xử lý của shot")

    class Config:
            """Configuration for the model."""
            json_schema_extra = {
                "example": {
                    "id": "L01_V001_S001_SH001",
                    "path": "/resource/shots/L01_V001.mp4",
                    "start_frame": 100,
                    "end_frame": 200,
                    "start_time": 15.5,
                    "end_time": 30.2,
                    "duration": 14.7,
                    # "time": (15.5, 30.2),
                    "is_deleted": False,
                    "is_processed": True
                }
            }