from datetime import datetime
import cv2
import pytz


def get_current_time() -> datetime:
    """Get the current time in Vietnam timezone."""
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    return datetime.now(tz)


def get_date_format():
    """
    Get the current date in the format "dd_mm_yyyy"
    """
    current_time = get_current_time()
    formatted_date = current_time.strftime("%d_%m_%Y")
    return formatted_date


def get_video_duration(video_path: str) -> float:
    """ Tính thời lượng video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return duration

def get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps