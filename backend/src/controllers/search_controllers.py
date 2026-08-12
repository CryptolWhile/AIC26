from flask import request, jsonify
from typing import List, Dict, Any, Optional, Tuple
import logging
import os

from src.services.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

class SearchController:
    """Controller for handling video search and retrieval HTTP requests."""
    
    def __init__(self):
        """Initialize the search controller with retrieval service."""
        try:
            # Initialize retrieval service
            self.retrieval_service = RetrievalService()
            
            logger.info("SearchController initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SearchController: {str(e)}")
            raise

    def search_videos(self, request_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Main search endpoint that handles multi-modal video search.
        
        Expected request body:
        {
            "prompt": "con mèo đang chạy",
            "extra_prompt": ["con mèo bắt con chuột", "con mèo cắn con chuột"], // nhập thủ công các subquery cho temporal search
            "ocr_search": "meo meo", // tìm kiếm văn bản trong video
            "dataset": "L01", // chỉ tìm trong dataset này, nếu là "all" thì tìm trong tất cả các dataset
            "video": "V001", // chỉ tìm trong video này, nếu là "all" thì tìm trong tất cả các video
            "n_results": 5, // số lượng kết quả frame trả về
            "models": ["blip", "clip"], 
        }
        
        Returns:
            Tuple containing response dict and status code
        """
        try:
            if not request_data:
                return {"error in search controller": "No request data provided"}, 400

            prompt, extra_prompts, ocr_search, dataset, video, limit, models, rerank_method = request_data.values()

            if not prompt:
                return {"error": "Main query must be provided"}, 400
                
            logger.info(f"Processing search request: prompt='{prompt}', extra_prompts='{extra_prompts}', ocr_search='{ocr_search}'")
            
            search_results = self.retrieval_service.search_videos(
                prompt = prompt,
                extra_prompts=extra_prompts,
                ocr_search=ocr_search,
                dataset_filter=dataset,
                video_filter=video,
                limit=limit,
                models=models,
                rerank_method=rerank_method
            )
            
            return {
                "status": "success",
                "results": search_results,
                "total_results": len(search_results)
            }, 200
            
        except Exception as e:
            logger.error(f"Error in search_videos: {str(e)}")
            return {"error": f"Search failed: {str(e)}"}, 500

    def get_datasets(self) -> Tuple[Dict[str, Any], int]:
        """
        Get list of available datasets.
        
        Returns:
            Tuple containing response dict and status code
        """
        try:
            # Delegate to retrieval service
            datasets = self.retrieval_service.get_datasets()
            
            return {
                "status": "success",
                "datasets": datasets
            }, 200
            
        except Exception as e:
            logger.error(f"Error in get_datasets: {str(e)}")
            return {"error": f"Failed to get datasets: {str(e)}"}, 500

    def get_videos(self, video_id: str = None) -> Tuple[Dict[str, Any], int]:
        """
        Get list of available videos for a video_id.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Tuple containing response dict and status code
        """
        try:
            videos = self.retrieval_service.get_videos(video_id)
            
            return {
                "status": "success",
                "video_id": video_id,
                "videos": videos
            }, 200
            
        except Exception as e:
            logger.error(f"Error in get_videos: {str(e)}")
            return {"error": f"Failed to get videos: {str(e)}"}, 500

    def get_keyframe_image(self, image_path: str) -> Tuple[Any, int]:
        """
        Serve keyframe images.
        """
        try:
            from flask import send_file
            import os
            # Build the absolute path to the image
            # Assuming the images are stored in 'backend/processed/images'
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'processed', 'images'))
            full_path = os.path.join(base_dir, image_path)
            
            if not os.path.exists(full_path):
                return {"error": "Image not found"}, 404
                
            return send_file(full_path, mimetype='image/jpeg'), 200
        except Exception as e:
            logger.error(f"Error serving image: {str(e)}")
            return {"error": f"Failed to serve image: {str(e)}"}, 500




    def submit_results(self, request_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Handle submission of selected search results.
        
        Expected request body:
        {
            "selected_results": ["path1", "path2", ...],
            "search_session_id": "uuid",
            "action": "save|export|analyze"
        }
        
        Returns:
            Tuple containing response dict and status code
        """
        try:
            selected_results, session_id, action = request_data.values()
            
            if not request_data:
                return {"error in submit_results": "No request data provided"}, 400
            
            if not selected_results:
                return {"error": "No results selected"}, 400
                
            # Delegate to retrieval service
            result = self.retrieval_service.process_result_submission(
                selected_results=selected_results,
                session_id=session_id,
                action=action
            )
            
            return {
                "status": "success",
                "message": f"Successfully {action}d {len(selected_results)} results",
                "submission_id": result.get('submission_id'),
                "processed_count": result.get('processed_count')
            }, 200
            
        except Exception as e:
            logger.error(f"Error in submit_results: {str(e)}")
            return {"error": f"Submission failed: {str(e)}"}, 500
