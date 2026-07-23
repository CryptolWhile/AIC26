import cv2
import pytz
import numpy as np
from datetime import datetime


def get_current_time() -> datetime:
    """Get the current time in Vietnam timezone."""
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    return datetime.now(tz)


def calculate_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    """
    cosine_score = np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )
    return float(cosine_score)


def get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps