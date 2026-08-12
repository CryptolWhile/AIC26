from flask import Flask, request
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    from src.routes.search_routes import search_routes
    from src.routes.submit_routes import submit_routes
    from src.routes.user_routes import user_routes
    # from src.routes.offline_routes import offline_routes

    from src.core.config import load_settings
    from src.database.manager import DatabaseManager
    from src.core import extensions

    # Initialize databases
    settings = load_settings()
    extensions.database_manager = DatabaseManager(settings)
    extensions.mongo_db = extensions.database_manager.get_mongo()
    extensions.milvus_db = extensions.database_manager.get_milvus()
    extensions.elastic_db = extensions.database_manager.get_elastic()
    extensions.faiss_db = extensions.database_manager.get_faiss()
    extensions.metadata_store_db = extensions.database_manager.get_metadata_store()

    app.register_blueprint(search_routes)
    app.register_blueprint(submit_routes)
    app.register_blueprint(user_routes)
    # app.register_blueprint(offline_routes)

    @app.route("/health")
    def health():
        return {"status": "ok"}

    @app.before_request
    def log_request_info():
        with open("request_debug.log", "a") as f:
            f.write(f"METHOD: {request.method}, URL: {request.url}, PATH: {request.path}\n")

    return app
