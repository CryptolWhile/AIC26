from flask import Blueprint, request
from src.controllers.submit_controllers import SubmitController

submit_routes = Blueprint("submit", __name__, url_prefix="/api/submit")
submit_controller = SubmitController()

@submit_routes.route("/", methods=["POST"])
def submit():
    request_data = request.get_json()

    message, status_code = submit_controller.submit(request_data)
    return message, status_code