from flask import Blueprint, request
from src.controllers.search_controllers import SearchController

search_routes = Blueprint("search", __name__, url_prefix="/api/search")

search_controller = SearchController()

@search_routes.route("/", methods=["POST"])
def search_videos():
    """
    Main search endpoint for video retrieval.
    
    POST /api/search/
    Body: {
        "prompt": "con mèo đang chạy",
        "extra_prompt": ["con mèo bắt con chuột", "con mèo cắn con chuột"], // nhập thủ công các subquery cho temporal search
        "ocr_search": "meo meo", // tìm kiếm văn bản trong video
        
        "dataset": "L01", // chỉ tìm trong dataset này, nếu là "all" thì tìm trong tất cả các dataset
        "video": "V001", // chỉ tìm trong video này, nếu là "all" thì tìm trong tất cả các video
        "n_results": 5, // số lượng kết quả frame trả về
        "model": ["blip", "clip"], 
        "rerank_method": "rrf"
    }
    """
    request_data = request.get_json()

    prompt = request_data.get('prompt', '').strip()
    extra_prompt = request_data.get('extra_prompt', [])
    ocr_search = request_data.get('ocr_search', '').strip()
    # asr_search = request_data.get('asr_search', '').strip()
    dataset_filter = request_data.get('dataset', '').strip()
    video_filter = request_data.get('video', '').strip()
    try:
        limit = int(request_data.get('n_results', 150))
    except ValueError:
        limit = 150
    models = request_data.get('model', [])
    rerank_method = request_data.get('rerank_method', 'rrf')

    message, status_code = search_controller.search_videos(
        request_data={
            "prompt": prompt,
            "extra_prompt": extra_prompt,
            "ocr_search": ocr_search,
            "dataset": dataset_filter,
            "video": video_filter,
            "limit": limit,
            "models": models,
            "rerank_method": rerank_method
            # "asr": asr_search
    })
    return message, status_code

@search_routes.route("/datasets", methods=["GET"])
def get_datasets():
    """
    Get list of available datasets.
    
    GET /api/search/datasets
    """
    message, status_code = search_controller.get_datasets()
    return message, status_code

@search_routes.route("/videos", methods=["GET"])
def get_videos():
    """
    Get list of available videos for a dataset.
    
    GET /api/search/videos?dataset=D01
    """
    dataset = request.args.get('dataset')
    message, status_code = search_controller.get_videos(dataset)
    return message, status_code

@search_routes.route("/image/<path:image_path>", methods=["GET"])
def get_keyframe_image(image_path):
    """
    Serve keyframe images.
    
    GET /api/search/image/D06/V01/7.jpg
    """
    message, status_code = search_controller.get_keyframe_image(image_path)
    return message, status_code

@search_routes.route("/submit", methods=["POST"])
def submit_results():
    """
    Handle submission of selected search results.
    
    POST /api/search/submit
    Body: {
        "selected_results": ["path1", "path2", ...],
        "search_session_id": "uuid",
        "action": "save|export|analyze"
    }
    """
    request_data = request.get_json()
    message, status_code = search_controller.submit_results(request_data)
    return message, status_code