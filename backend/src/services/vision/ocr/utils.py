import cv2
import pytz
import string
import numpy as np
from datetime import datetime

from underthesea import text_normalize


def get_current_time() -> datetime:
    """Get the current time in Vietnam timezone."""
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    return datetime.now(tz)


def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image from path: {image_path}")
    return image


def extract_polygons(result_detect):
    """Extract polygons from the detection result."""
    if not result_detect or 'predictions' not in result_detect or not result_detect['predictions']:
        return []
    pred = result_detect['predictions'][0]
    return pred.get('det_polygons', [])


def get_top_left_corner(polygon):
    """Get the top-left corner of a polygon."""
    if not polygon or len(polygon) < 4:
        return 0, 0
    polygon = np.array(polygon).reshape(-1, 2)
    x_min = np.min(polygon[:, 0])
    y_min = np.min(polygon[:, 1])
    return x_min, y_min


def sort_polygons(polygons):
    """Sort polygons based on their top-left corner."""
    polygons_with_top_left = [(polygon, *get_top_left_corner(polygon)) for polygon in polygons]
    polygons_sorted = sorted(polygons_with_top_left, key=lambda x: (x[2], x[1]))
    sorted_polygons = [item[0] for item in polygons_sorted]
    return sorted_polygons


def crop_polygon(image, polygon, y_threshold):
    """Crop the image based on the polygon coordinates."""
    polygon = np.array(polygon).reshape(-1, 2)
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)

    if float((y_max-y_threshold)) / (y_max-y_min) > 0.5:
        return None

    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(image.shape[1], int(x_max)), min(image.shape[0], int(y_max))
    if x_min >= x_max or y_min >= y_max:
        return None
    return image[int(y_min):int(y_max), int(x_min):int(x_max)]


def text_processing(text):
    processed_text = text.lower()
    processed_text = text_normalize(processed_text)
    processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
    return processed_text