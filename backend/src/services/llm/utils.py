import json
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple

def clean_response(response: str) -> str:
    raw_text = response.text.strip()
    if raw_text.startswith("```") and raw_text.endswith("```"):
        raw_text = "\n".join(raw_text.split("\n")[1:-1])
    return raw_text

def extract_json_from_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None

def get_image_format(file_path: str) -> str:
    ext = file_path.lower().split('.')[-1]
    if ext in ("jpg", "jpeg"):
        return "JPEG"
    elif ext == "png":
        return "PNG"
    raise ValueError(f"Unsupported image format: {ext}")


def resize_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size)


def encode_image_from_pil(img: Image.Image, format: str) -> str:
    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_image(
    image_path: str,
    resize: bool = False,
    size: Tuple[int, int] = (1280, 1280)
) -> Tuple[str, str]:
    format = get_image_format(image_path)
    with Image.open(image_path) as img:
        if img.mode == "P":
            img = img.convert("RGB")
        if resize:
            img = resize_image(img, size)
        encoded_string = encode_image_from_pil(img, format)
    return encoded_string, format.lower()