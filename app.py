"""
app.py - Flask Server for Parking Slot Detection System.

Serves a premium HTML/CSS/JS dashboard and handles video uploads,
MJPEG live processing streams, and browser webcam processing endpoints.
"""

from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
import json
import time
from werkzeug.utils import secure_filename

from detector import VehicleDetector
from parking import ParkingManager
from utils import FPSCounter, draw_slots, draw_detections, draw_dashboard
from auto_calibrate import auto_detect_slots, save_slots
import config as cfg

# Initialize Flask
app = Flask(__name__, static_folder="static", template_folder="templates")

# Upload folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global shared variables (configured via API)
global_settings = {
    "conf_threshold": 0.25,
    "overlap_threshold": 0.25,
    "selected_model": "yolov8s.pt",
    "video_path": None,
    "is_playing": False,
}

# Shared metrics state for video playback
latest_metrics = {
    "total": 0,
    "occupied": 0,
    "available": 0,
    "fps": 0.0
}

# Lazy loading of models to optimize startup
_detector = None
_parking_manager = None


def get_detector():
    """Get or load the vehicle detector."""
    global _detector
    if _detector is None or _detector.model.overrides.get("model") != global_settings["selected_model"]:
        _detector = VehicleDetector(model_path=global_settings["selected_model"])
    return _detector


def get_parking_manager():
    """Get or load the parking slot manager."""
    global _parking_manager
    if _parking_manager is None:
        _parking_manager = ParkingManager()
    return _parking_manager


# ─────────────────────────────────────────────
# Core Image Processing Pipeline
# ─────────────────────────────────────────────
def process_single_frame(frame, conf, overlap):
    """Process a single frame through detector and slot manager."""
    detector = get_detector()
    parking_manager = get_parking_manager()

    # Detect
    detections = detector.detect(frame, conf=conf)

    # Check slots
    h, w = frame.shape[:2]
    parking_manager.check_occupancy_fast(detections, (h, w), overlap_thresh=overlap)

    # Status
    total, occupied, available = parking_manager.get_status()

    # Annotate frame
    annotated = draw_slots(frame.copy(), parking_manager.get_slots())
    annotated = draw_detections(annotated, detections)

    return annotated, total, occupied, available


# ─────────────────────────────────────────────
# Video Processing Generator (MJPEG Stream)
# ─────────────────────────────────────────────
def generate_video_stream():
    """Generator for streaming processed video frames."""
    video_path = global_settings["video_path"]
    if not video_path or not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    fps_counter = FPSCounter()

    while cap.isOpened() and global_settings["is_playing"]:
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter.tick()
        fps = fps_counter.get_fps()

        # Retrieve thresholds
        conf = global_settings["conf_threshold"]
        overlap = global_settings["overlap_threshold"]

        # Process frame
        processed, total, occupied, available = process_single_frame(frame, conf, overlap)

        # Update latest metrics for status API polling
        latest_metrics["total"] = total
        latest_metrics["occupied"] = occupied
        latest_metrics["available"] = available
        latest_metrics["fps"] = round(fps, 1)

        # Draw dashboard on frame
        processed = draw_dashboard(processed, total, occupied, available, fps)

        # Encode to JPEG
        ret, jpeg = cv2.imencode(".jpg", processed)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()

        # Yield frame in MJPEG multipart format
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        # Regulate playback speed (approx. 25 FPS)
        time.sleep(0.03)

    cap.release()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main frontend dashboard."""
    return render_template("index.html")


@app.route("/api/settings", methods=["GET", "POST"])
def settings_handler():
    """Get or update global configuration settings."""
    if request.method == "POST":
        data = request.json
        if "conf_threshold" in data:
            global_settings["conf_threshold"] = float(data["conf_threshold"])
        if "overlap_threshold" in data:
            global_settings["overlap_threshold"] = float(data["overlap_threshold"])
        if "selected_model" in data:
            model = data["selected_model"]
            if model in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]:
                if global_settings["selected_model"] != model:
                    global_settings["selected_model"] = model
                    # Force reload of detector on next call
                    global _detector
                    _detector = None
                    # Reset slots since a new model is selected
                    if os.path.exists(cfg.PARKING_SLOTS_FILE):
                        os.remove(cfg.PARKING_SLOTS_FILE)
                        global _parking_manager
                        _parking_manager = None

        return jsonify({"status": "success", "settings": global_settings})

    return jsonify(global_settings)


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return the latest frame processing metrics for Video mode."""
    return jsonify(latest_metrics)


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Handle video uploads and trigger auto-calibration if slots are empty."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    global_settings["video_path"] = filepath
    global_settings["is_playing"] = False

    # Check if we need to auto-calibrate
    slots_exist = os.path.exists(cfg.PARKING_SLOTS_FILE)

    return jsonify({
        "status": "success",
        "filename": filename,
        "needs_calibration": not slots_exist
    })


@app.route("/api/auto_detect", methods=["POST"])
def trigger_auto_detect():
    """Run auto-calibration on the first frame of the uploaded video."""
    video_path = global_settings["video_path"]
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "No video file uploaded yet"}), 400

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Could not read frame from video"}), 500

    detector = get_detector()
    slots = auto_detect_slots(frame, detector, conf=0.20)

    if not slots:
        return jsonify({"error": "No slots detected. Point at visible parked cars."}), 400

    save_slots(slots, cfg.PARKING_SLOTS_FILE)
    
    # Reload parking manager to pick up new slots
    global _parking_manager
    _parking_manager = ParkingManager()

    return jsonify({
        "status": "success",
        "slots_count": len(slots)
    })


@app.route("/api/reset_slots", methods=["POST"])
def reset_slots():
    """Delete current parking slots file."""
    if os.path.exists(cfg.PARKING_SLOTS_FILE):
        os.remove(cfg.PARKING_SLOTS_FILE)
    global _parking_manager
    _parking_manager = None
    return jsonify({"status": "success"})


@app.route("/api/control_playback", methods=["POST"])
def control_playback():
    """Start or stop video playback."""
    data = request.json
    action = data.get("action")
    if action == "start":
        global_settings["is_playing"] = True
    elif action == "stop":
        global_settings["is_playing"] = False
    return jsonify({"status": "success", "is_playing": global_settings["is_playing"]})


@app.route("/video_feed")
def video_feed():
    """Stream processed video frames as MJPEG."""
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/process_webcam", methods=["POST"])
def process_webcam():
    """Process a single base64 webcam frame sent from the browser."""
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    # Parse base64 image data
    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Process frame
    conf = global_settings["conf_threshold"]
    overlap = global_settings["overlap_threshold"]

    # Auto-detect slots on webcam if none exist
    parking_manager = get_parking_manager()
    if len(parking_manager.get_slots()) == 0:
        detector = get_detector()
        slots = auto_detect_slots(frame, detector, conf=0.20)
        if slots:
            save_slots(slots, cfg.PARKING_SLOTS_FILE)
            parking_manager = ParkingManager()

    # Process frame
    processed, total, occupied, available = process_single_frame(frame, conf, overlap)

    # Calculate FPS (simulated for webcam API client)
    fps = data.get("fps", 0.0)
    processed = draw_dashboard(processed, total, occupied, available, fps)

    # Encode back to JPEG base64
    ret, jpeg = cv2.imencode(".jpg", processed)
    if not ret:
        return jsonify({"error": "Failed to encode frame"}), 500

    processed_bytes = jpeg.tobytes()
    processed_base64 = base64.b64encode(processed_bytes).decode("utf-8")

    return jsonify({
        "image": f"data:image/jpeg;base64,{processed_base64}",
        "total": total,
        "occupied": occupied,
        "available": available
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
