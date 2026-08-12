from flask import request, jsonify
import logging
from src.services.vision.ocr.service import OCRService
from src.services.vision.asr.service import ASRService
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class OfflineController:
    def __init__(self):
        self.ocr_service = OCRService()
        self.asr_service = ASRService()
    def insert_ocr(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        message, status_code = self.ocr_service.insert(data)
        return message, status_code

    def insert_asr(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        message, status_code = self.asr_service.insert(data)
        return message, status_code