"""
detector.py - Vehicle detection using YOLOv8.

Wraps the Ultralytics YOLOv8 model to detect cars, motorcycles,
buses, and trucks in video frames.
"""

from ultralytics import YOLO
from config import (
    YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    VEHICLE_CLASSES,
)


class VehicleDetector:
    """Detects vehicles in frames using a pretrained YOLOv8 model."""

    def __init__(self, model_path=YOLO_MODEL):
        """
        Load the YOLOv8 model.

        Args:
            model_path: Path to the YOLO weights file.
                        Defaults to yolov8n.pt (auto-downloaded).
        """
        print(f"[INFO] Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.vehicle_class_ids = set(VEHICLE_CLASSES.keys())
        print(f"[INFO] Model loaded. Detecting: {list(VEHICLE_CLASSES.values())}")

    def detect(self, frame, conf=None):
        """
        Run vehicle detection on a single frame.

        Args:
            frame: BGR image (numpy array).
            conf: Optional confidence threshold override.

        Returns:
            List of detections, each as a dict:
                {
                    "bbox": (x1, y1, x2, y2),   # top-left, bottom-right
                    "confidence": float,
                    "class_id": int,
                    "class_name": str,
                }
        """
        results = self.model(
            frame,
            conf=conf if conf is not None else CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                if class_id not in self.vehicle_class_ids:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": VEHICLE_CLASSES[class_id],
                })

        return detections
