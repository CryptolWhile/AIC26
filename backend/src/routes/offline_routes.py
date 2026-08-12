from flask import Blueprint, request
from src.controllers.offline_controllers import OfflineController
import json

offline_routes = Blueprint("offline", __name__, url_prefix="/api/offline")
offline_controller = OfflineController()

@offline_routes.route("/ocr/insert", methods=["POST"])
def insert_ocr():
    request_data = request.get_json()
    ocr_path = request_data["path"]
    with open(ocr_path, 'r') as f:
        data = json.load(f)

    message, status_code = offline_controller.insert_ocr(data)
    return message, status_code

@offline_routes.route("/asr/insert", methods=["POST"])
def insert_asr():
    request_data = request.get_json()
    asr_path = request_data["path"]
    with open(asr_path, 'r') as f:
        data = json.load(f)
    message, status_code = offline_controller.insert_asr(data)
    return message, status_code