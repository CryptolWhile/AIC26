import re
import json
from typing import Dict, List, Any

import pytz
from datetime import datetime

def get_current_time() -> datetime:
    """Get the current time in Vietnam timezone."""
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    return datetime.now(tz)

def extract_video_id_from_url(url: str) -> str:
    """Cắt lấy ID của video YouTube"""
    video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not video_id_match:
        raise ValueError("Định dạnh Youtube url không hợp lệ")
    return video_id_match.group(1)

def clean_and_parse_json_markdown(raw_text: str) -> Any:
    """Lấy đoạn text trong dấu ``` ```` """
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in the input string.")
    
    json_str = match.group(1)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
def calculate_token_count(text: str) -> int:
    """Đếm số token"""
    return len(text.split(" "))

def format_transcript_for_llm(transcript_segments: List[Dict]) ->str:
    """Đánh số thứ tự cho transcrpt"""
    return "/n".join([
        f"index: {index} - {segment['transcript']}"
        for index, segment in enumerate(transcript_segments)
        # index: 0 - Xin chào các bạn
        # index: 1 - Hôm nay trời nắng
    ])

def create_system_prompt_segmentation() -> str:
    return """
    Bạn là một chuyên gia phân tích ngôn ngữ và nội dung video.
    Bạn sẽ được cung cấp một transcript, trong đó mỗi dòng có cấu trúc:
    <time_step>: <text>
    Video có thể bao gồm nhiều **sự kiện khác nhau**. 
    Nhiệm vụ của bạn là **nhóm các dòng transcript liên tiếp nói về một sự kiện** lại với nhau. 
    Lưu ý:
    - Các dòng transcript trong một nhóm phải liền nhau theo thứ tự ban đầu.
    - Không được đảo thứ tự dòng transcript.
    - Các dòng transcript trong một nhóm phải liên tục theo thứ tự ban đầu.
    Trả về **một JSON duy nhất** với định dạng sau:
    [
        {
            "id": <int>,       // số thứ tự của nhóm chủ đề, bắt đầu từ 1
            "group_transcript": [<int>]   // danh sách chỉ số dòng transcript thuộc nhóm này
        },
        ...
    ]
    """


def create_system_prompt_refinement() -> str:
    return """
    Bạn là một chuyên gia phân tích ngôn ngữ và nội dung video.
    Bạn sẽ được cung cấp:
    - Một đoạn transcript **trước đó** (ngữ cảnh)
    - Một đoạn transcript **cần tinh chỉnh**
    - Một đoạn transcript **sau đó** (ngữ cảnh)
    Nhiệm vụ của bạn là:
    - Dựa trên ngữ cảnh trước và sau, **chỉnh sửa (refine)** đoạn transcript chính giữa sao cho:
        - Đầy đủ thông tin
        - Mạch lạc, rõ ràng
        - Tự nhiên về mặt ngôn ngữ
        - Phù hợp với ngữ cảnh xung quanh
    **Chỉ trả về đoạn transcript đã tinh chỉnh**, không thêm nhận xét hoặc định dạng nào khác.
    """


def create_user_prompt_refinement(current_segment: Dict, previous_segment: Dict = None, next_segment: Dict = None) -> str:
    previous_text = previous_segment['transcript'] if previous_segment else ""
    next_text = next_segment['transcript'] if next_segment else ""
    
    return f"""
    Dưới đây là đoạn transcript trước đó:
    {previous_text}
    Dưới đây là đoạn transcript cần tinh chỉnh:
    {current_segment['transcript']}
    Dưới đây là đoạn transcript sau đó:
    {next_text}
    """