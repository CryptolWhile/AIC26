import cv2
import torch
import warnings
import numpy as np
from PIL import Image
from typing import List, Dict, Any

import easyocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from src.services.vision.ocr.utils import (
    get_current_time, load_image, text_processing
)
from src.services.vision.ocr.interface import OCRBox, OCRResult, OCRModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class EasyOCRVietOCRModel(OCRModel):
    """
    Text Detection: EasyOCR 
    Text Recognition: VietOCR 
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        if not verbose:
            from contextlib import redirect_stdout, redirect_stderr
            import io
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                self.initialize()
        else:
            self.initialize()

    def initialize(self):
        try:
            # Initialize EasyOCR for text detection
            # gpu=True sẽ tự động dùng CUDA nếu có
            self.easyocr_reader = easyocr.Reader(
                ['vi', 'en'], 
                gpu=torch.cuda.is_available(),
                verbose=self.verbose
            )

            # Initialize VietOCR for text recognition 
            config = Cfg.load_config_from_name('vgg_transformer')
            config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            config['cnn']['pretrained'] = False

            if torch.cuda.is_available():
                try:
                    config['predictor'] = config.get('predictor', {})
                    config['predictor']['beamsearch'] = False
                except:
                    pass

            self.vietocr_rec = Predictor(config)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize OCR models: {str(e)}")
    
    def detect_text(self, image_path: str) -> List[np.ndarray]:
        """Detect text regions in an image using EasyOCR."""
        try:
            # EasyOCR trả về list of [bbox, text, confidence]
            # bbox là [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            results = self.easyocr_reader.readtext(image_path)
            
            # Chuyển đổi bbox sang dạng polygon phẳng và sắp xếp theo vị trí (y trước, x sau)
            polygons = []
            for (bbox, text, conf) in results:
                # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                polygon = np.array(bbox).flatten()  # [x1,y1,x2,y2,x3,y3,x4,y4]
                polygons.append(polygon)
            
            # Sắp xếp theo top-left corner (y trước, x sau)
            if polygons:
                polygons.sort(key=lambda p: (min(p[1::2]), min(p[0::2])))
            
            return polygons
        except Exception as e:
            if self.verbose:
                print(f"Warning: Text detection failed: {e}")
            return []

    def recognize_text(self, cropped_image: np.ndarray) -> str:
        """Recognize text from a cropped image region using VietOCR."""
        if cropped_image is None or cropped_image.size == 0:
            return ""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            return self.vietocr_rec.predict(img_pil).strip()
        except Exception as e:
            if self.verbose:
                print(f"Warning: Text recognition failed: {e}")
            return ""

    def _crop_polygon(self, image: np.ndarray, polygon: np.ndarray, y_threshold: int = 650) -> np.ndarray:
        """Crop the image based on the polygon coordinates."""
        polygon = np.array(polygon).reshape(-1, 2)
        x_min, y_min = np.min(polygon, axis=0)
        x_max, y_max = np.max(polygon, axis=0)

        if float((y_max - y_threshold)) / max((y_max - y_min), 1) > 0.5:
            return None

        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(image.shape[1], int(x_max)), min(image.shape[0], int(y_max))
        if x_min >= x_max or y_min >= y_max:
            return None
        return image[y_min:y_max, x_min:x_max]

    def extract_text_from_image(self, image_path: str, y_threshold: int = 650) -> OCRResult:
        """Extract and recognize all text from an image."""
        try:
            image = load_image(image_path)
            boxes = []
            for poly in self.detect_text(image_path):
                cropped = self._crop_polygon(image, poly, y_threshold)
                if cropped is not None:
                    text = self.recognize_text(cropped)
                    if text:
                        points = np.array(poly).reshape(-1, 4)
                        bbox = [tuple(int(x) for x in point) for point in points]
                        boxes.append(
                            OCRBox(
                                text=text_processing(text),
                                BBox=bbox
                            )
                        )
            return OCRResult(
                image_path=image_path,
                boxes=boxes,
                created_at=get_current_time()
            )
        
        except Exception as e:
            raise ValueError(f"Failed to extract text from image: {e}")
    
    def get_model_name(self) -> str:
        return "vietocr"
