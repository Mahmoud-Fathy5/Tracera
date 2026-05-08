"""
Tracera — Flask Backend Server
==============================
Serves the deepfake detection API and static frontend.

Security:
  - Rate limiting (10 req/min per IP)
  - File type validation (magic bytes + extension)
  - Max upload size (10 MB)
  - Content Security Policy headers
  - Secure filename handling
  - Immediate temp file cleanup
"""

import os
import tempfile
import logging
from functools import wraps

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

from inference import GramNetDetector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIMETYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app():
    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="",
    )

    # -- Security Configuration --
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # CORS — restrict in production, allow all for development
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Rate limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["60 per hour"],
        storage_uri="memory://",
    )

    # -- Logging --
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("tracera")

    # -- Load Model (once at startup) --
    logger.info("Loading GramNet v3 model...")
    detector = GramNetDetector(MODEL_DIR, device="cpu")
    logger.info("Model loaded successfully.")

    # -- Security Headers Middleware --
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self';"
        )
        return response

    # -----------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------

    @app.route("/")
    def index():
        """Serve the main frontend page."""
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check endpoint for monitoring."""
        return jsonify({"status": "ok", "model": "GramNet v3"})

    @app.route("/api/predict", methods=["POST"])
    @limiter.limit("10 per minute")
    def predict():
        """
        Analyze an uploaded image for deepfake detection.
        
        Expects: multipart/form-data with field "image"
        Returns: JSON {verdict, confidence, attribution, attribution_confidence}
        """
        # -- Validate request --
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        # -- Validate filename extension --
        filename = secure_filename(file.filename)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Invalid file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # -- Validate MIME type --
        if file.content_type not in ALLOWED_MIMETYPES:
            return jsonify({
                "error": f"Invalid MIME type '{file.content_type}'."
            }), 400

        # -- Read and validate magic bytes --
        file_bytes = file.read()

        if len(file_bytes) == 0:
            return jsonify({"error": "Empty file."}), 400

        if len(file_bytes) > MAX_CONTENT_LENGTH:
            return jsonify({"error": "File too large. Maximum size: 10 MB."}), 413

        if not GramNetDetector.validate_image_bytes(file_bytes):
            return jsonify({
                "error": "File content does not match a valid image format."
            }), 400

        # -- Run inference --
        try:
            result = detector.predict(file_bytes)
            logger.info(
                "Prediction: %s (confidence=%.4f, attribution=%s)",
                result["verdict"],
                result["confidence"],
                result["attribution"],
            )
            return jsonify(result)

        except Exception as e:
            logger.error("Inference error: %s", str(e), exc_info=True)
            return jsonify({
                "error": "An error occurred during analysis. Please try a different image."
            }), 500

    # -----------------------------------------------------------------
    # Error Handlers
    # -----------------------------------------------------------------

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large. Maximum size: 10 MB."}), 413

    @app.errorhandler(429)
    def rate_limited(e):
        return jsonify({
            "error": "Rate limit exceeded. Please wait a moment before trying again."
        }), 429

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error."}), 500

    return app


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    # Development server — do NOT use in production
    # For production, use: gunicorn "app:create_app()" --bind 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000, debug=False)
