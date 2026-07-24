import ssl
import cv2
import torch
import warnings
import numpy as np
from PIL import Image
from typing import List, Dict, Any


from mmocr.apis import MMOCRInferencer
from mmocr.utils import register_all_modules
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


from src.services.vision.ocr.utils import (
    get_current_time, load_image, extract_polygons, 
    sort_polygons, crop_polygon, text_processing
)
from src.services.vision.ocr.interface import OCRBox, OCRResult, OCRModel


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


register_all_modules()
ssl._create_default_https_context = ssl._create_unverified_context


class MMOCRVietOCRModel(OCRModel):
    
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
            # Initialize MMOCR for text detection
            self.mmocr_det = MMOCRInferencer(
                det='DRRG',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            ) # out: {
                #     "predictions": [
                #         {
                #             "det_polygons": [
                #                 [x1,y1,x2,y2,...],
                #                 [x1,y1,x2,y2,...],
                #                 ...
                #             ],
                #             "det_scores": [
                #                 0.98,
                #                 0.95,
                #                 ...
                #             ]
                #         }
                #     ]
                # }

            # Initialize VietOCR for text recognition
            config = Cfg.load_config_from_name('vgg_transformer')
            config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            config['cnn']['pretrained'] = False

            # Optimize for inference
            if torch.cuda.is_available():
                try:
                    config['predictor'] = config.get('predictor', {})
                    config['predictor']['beamsearch'] = False
                except:
                    pass  # Ignore if predictor config fails

            self.vietocr_rec = Predictor(config)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize OCR models: {str(e)}")
    
    def detect_text(self, image_path: str) -> List[np.ndarray]:
        """Detect text regions in an image.
        """
        try:
            if self.verbose:
                text_det = self.mmocr_det(image_path, return_vis=False)
            else:
                import io
                from contextlib import redirect_stdout, redirect_stderr
                f = io.StringIO()
                with redirect_stdout(f), redirect_stderr(f):
                    text_det = self.mmocr_det(image_path, return_vis=False)

            return sort_polygons(extract_polygons(text_det))
        except Exception as e:
            raise ValueError(f"Failed to detect text regions: {e}")

    def recognize_text(self, cropped_image: np.ndarray) -> str:
        """Recognize text from a cropped image region."""
        if cropped_image is None or cropped_image.size == 0:
            return ""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            return self.vietocr_rec.predict(img_pil).strip()
        except Exception as e:
            if self.verbose:
                print(f"Warning: Text recognition failed: {e}")
            return ""

    def extract_text_from_image(self, image_path: str, y_threshold: int = 650) -> OCRResult:
        """Extract and recognize all text from an image."""
        try:
            image = load_image(image_path)
            boxes = []
            for poly in self.detect_text(image_path):
                cropped = crop_polygon(image, poly, y_threshold)
                if cropped is not None:
                    text = self.recognize_text(cropped)
                    if text:
                        points = np.array(poly).reshape(-1, 4)
                        bbox = [tuple(point) for point in points]
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