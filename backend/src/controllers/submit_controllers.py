import logging
from typing import List, Dict, Any, Tuple
from src.services.submit.service import SubmitService

logger = logging.getLogger(__name__)

class SubmitController:
    def __init__(self):
        try:
            self.submit_service = SubmitService()
        except Exception as e:
            logger.error(f"Failed to initialize SubmitController: {str(e)}")
            raise

    def submit(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        try:
            if not data:
                return {"error": "No request data provided"}, 400
            
            message, status_code = self.submit_service.submit(
                path = 'submission',
                data = data
            )
            return message, status_code
        except Exception as e:
            logger.error(f"Error in submit: {str(e)}")
            return {"error": f"Submit failed: {str(e)}"}, 500