from typing import Dict, Tuple, Any
from pydantic import Field, BaseModel

class Video(BaseModel):
    "Model thể hiện video trong hệ thống"

    id: str = Field(..., description="Mã định danh của video", examples=["L01_V001","L02_V001"])
    path: str = Field(..., description="đường dẫn video trong hệ thống", examples=["/resource/L01_V001.mp4"])
    fps: int = Field(..., description="Số khung hình mỗi giây của video", examples=[30,60])
    metadata: Dict[str, Any] = Field( default={}, description="Thông tin metadata của video", examples=[{"codec": "h264", "bitrate": "5000k"}])
    is_deleted: bool = Field(default=False, description="Trạng thái xóa video")
    is_processed: bool = Field(default=False, description="Trạng thái xử lý video")

    class Config:
        """Cấu hình cho model """

        json_schema_extra = {
            "example": {
                "id": "L001_V001",
                "path": "/resource/L001_V001.mp4",
                "fps": 30,
                "metadata": {"codec": "h264"},
                "is_deleted": False,
                "is_processed": True
            }
        }