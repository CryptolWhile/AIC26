import av
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, is_dataclass
from typing import Iterable


def get_fps(video_path: str) -> float:
    try:
        container = av.open(video_path)
        stream = next(s for s in container.streams if s.type == "video")
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        container.close()
        return fps
    except Exception as e:
        raise IOError(f"Cannot open video: {video_path}, error: {e}")


def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def iter_videos_in_range(video_dir: Path, k_start: int, k_end: int, num_gpus: int = 1, gpu_id: int = 0) -> Iterable[Path]:
    """Duyệt qua các tệp video trong thư mục có tên bắt đầu bằng K/L{số} nằm trong phạm vi được chỉ định.."""
    for p in sorted(video_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        part = p.name.split("_")[0]
        # if part.startswith("K"):
        if part.startswith(("K", "L")):
            try:
                k_num = int(part[1:])
            except ValueError:
                continue
            if k_start <= k_num <= k_end:
                if num_gpus > 1:
                    try:
                        v_part = p.name.split("_")[1].split(".")[0] # V001
                        v_num = int(v_part[1:])
                        if v_num % num_gpus != gpu_id:
                            continue
                    except Exception:
                        pass
                yield p


def iter_json_in_range(shot_dir: Path, k_start: int, k_end: int, num_gpus: int = 1, gpu_id: int = 0) -> Iterable[Path]:
    """Duyệt qua các tệp JSON trong thư mục có tên bắt đầu bằng K/L{số} nằm trong phạm vi được chỉ định."""
    for p in sorted(shot_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        part = p.name.split("_")[0]
        # if part.startswith("K"):
        if part.startswith(("K", "L")):
            try:
                k_num = int(part[1:])
            except ValueError:
                continue
            if k_start <= k_num <= k_end:
                if num_gpus > 1:
                    try:
                        v_part = p.name.split("_")[1].split(".")[0] # V001
                        v_num = int(v_part[1:])
                        if v_num % num_gpus != gpu_id:
                            continue
                    except Exception:
                        pass
                yield p