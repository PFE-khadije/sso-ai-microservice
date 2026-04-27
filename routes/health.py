from flask import Blueprint, current_app, jsonify
from flask import request
import os
import app

health_bp = Blueprint("health", __name__)

API_KEY = os.environ.get('API_KEY', None)

@app.before_request
def check_api_key():
    # Permettre le health check sans clé (optionnel)
    if request.path == '/health':
        return
    key = request.headers.get('X-API-Key')
    if not key or key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
@health_bp.get("/health")
def health():
    """Returns service status and model readiness."""
    model_service = current_app.config.get("MODEL_SERVICE")
    ready = model_service is not None and getattr(model_service, "model_loaded", False)
    payload = {
        "status": "ok" if ready else "degraded",
        "model_loaded": ready,
    }
    return jsonify(payload), 200 if ready else 503