"""
config.py - Configuration constants for Parking Slot Detection System.

Centralizes all tunable parameters: model settings, detection thresholds,
color palette, display options, and file paths.
"""

# ──────────────────────────────────────────────
# YOLOv8 Model Settings
# ──────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"          # YOLOv8 Nano — fast, auto-downloaded
CONFIDENCE_THRESHOLD = 0.4         # Minimum detection confidence
IOU_THRESHOLD = 0.45               # NMS IoU threshold

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ──────────────────────────────────────────────
# Parking Slot Settings
# ──────────────────────────────────────────────
PARKING_SLOTS_FILE = "parking_slots.json"
OVERLAP_THRESHOLD = 0.3            # Min overlap ratio to mark slot as occupied

# ──────────────────────────────────────────────
# Colors (BGR format for OpenCV)
# ──────────────────────────────────────────────
COLOR_GREEN = (0, 200, 0)          # Empty slot
COLOR_RED = (0, 0, 200)            # Occupied slot
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (0, 220, 255)       # Detection bounding box
COLOR_DASHBOARD_BG = (30, 30, 30)  # Dashboard background

# ──────────────────────────────────────────────
# Display Settings
# ──────────────────────────────────────────────
SLOT_OVERLAY_ALPHA = 0.35          # Transparency for slot overlays
DASHBOARD_HEIGHT = 60              # Height of info panel in pixels
FONT_SCALE = 0.6
FONT_THICKNESS = 2
WINDOW_NAME = "Parking Slot Detection"

# ──────────────────────────────────────────────
# Output Settings
# ──────────────────────────────────────────────
OUTPUT_VIDEO_PATH = "output.avi"
OUTPUT_CODEC = "XVID"
OUTPUT_FPS = 25.0
