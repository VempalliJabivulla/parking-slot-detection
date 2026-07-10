"""
parking.py - Parking slot management and occupancy detection.

Loads parking slot coordinates from a JSON file, checks each slot
against detected vehicle bounding boxes, and tracks occupancy status.
"""

import json
import numpy as np
import cv2
from config import PARKING_SLOTS_FILE, OVERLAP_THRESHOLD


class ParkingManager:
    """Manages parking slot definitions and occupancy state."""

    def __init__(self, slots_file=PARKING_SLOTS_FILE):
        """
        Initialize and load parking slots from a JSON file.

        Args:
            slots_file: Path to the JSON file containing slot coordinates.
        """
        self.slots = []
        self.load_slots(slots_file)

    def load_slots(self, json_path):
        """
        Load parking slot polygons from a JSON file.

        Expected JSON format:
        {
            "slots": [
                {
                    "id": 1,
                    "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                },
                ...
            ]
        }

        Args:
            json_path: Path to the parking slots JSON file.
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            self.slots = []
            self.history = {}  # Reset temporal state buffer: slot_id -> list of bools
            for slot_data in data["slots"]:
                slot = {
                    "id": slot_data["id"],
                    "points": np.array(slot_data["points"], dtype=np.int32),
                    "occupied": False,
                }
                self.slots.append(slot)
                self.history[slot_data["id"]] = []

            print(f"[INFO] Loaded {len(self.slots)} parking slots from '{json_path}'")

        except FileNotFoundError:
            print(f"[ERROR] Parking slots file not found: {json_path}")
            print("[INFO]  Create a 'parking_slots.json' with your slot coordinates.")
            self.slots = []
            self.history = {}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] Invalid parking slots file: {e}")
            self.slots = []
            self.history = {}

    def check_occupancy(self, detections):
        """
        Update occupancy status for each slot based on vehicle detections.

        A slot is marked occupied if a detected vehicle bounding box (bottom 45%)
        overlaps with the slot polygon area above the configured threshold.

        Args:
            detections: List of detection dicts from VehicleDetector.detect().
        """
        for slot in self.slots:
            instant_occupied = False

            # Compute the bounding rect and area of the slot polygon
            slot_mask = np.zeros((1080, 1920), dtype=np.uint8)
            cv2.fillPoly(slot_mask, [slot["points"]], 255)
            slot_area = cv2.countNonZero(slot_mask)

            if slot_area == 0:
                continue

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]

                # Ground contact area: use bottom 45% of the vehicle height to avoid slanting overlaps
                h_box = y2 - y1
                y1_bottom = int(y2 - h_box * 0.45)

                # Create a mask for the ground contact bounding box
                det_mask = np.zeros_like(slot_mask)
                cv2.rectangle(det_mask, (x1, y1_bottom), (x2, y2), 255, -1)

                # Compute intersection
                intersection = cv2.bitwise_and(slot_mask, det_mask)
                intersection_area = cv2.countNonZero(intersection)

                # Compute overlap ratio relative to the slot area
                overlap_ratio = intersection_area / slot_area

                if overlap_ratio >= OVERLAP_THRESHOLD:
                    instant_occupied = True
                    break

            # Update temporal buffer and perform majority voting
            self._update_temporal_state(slot, instant_occupied)

    def check_occupancy_fast(self, detections, frame_shape, overlap_thresh=None):
        """
        Fast occupancy check using bounding-box approximation.

        Uses the bounding rectangle of each slot polygon and bottom 45%
        of vehicle bounding boxes for robust, fast overlap check.

        Args:
            detections: List of detection dicts from VehicleDetector.detect().
            frame_shape: Tuple (height, width) of the current frame.
            overlap_thresh: Optional overlap threshold override.
        """
        threshold = overlap_thresh if overlap_thresh is not None else OVERLAP_THRESHOLD

        for slot in self.slots:
            instant_occupied = False

            # Get bounding rect of the slot polygon
            sx, sy, sw, sh = cv2.boundingRect(slot["points"])
            slot_area = sw * sh

            if slot_area == 0:
                continue

            for det in detections:
                dx1, dy1, dx2, dy2 = det["bbox"]

                # Perspective check: restrict to bottom 45% (wheels/chassis)
                h_box = dy2 - dy1
                dy1_bottom = int(dy2 - h_box * 0.45)

                # Compute intersection rectangle
                ix1 = max(sx, dx1)
                iy1 = max(sy, dy1_bottom)
                ix2 = min(sx + sw, dx2)
                iy2 = min(sy + sh, dy2)

                if ix1 < ix2 and iy1 < iy2:
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    overlap_ratio = intersection_area / slot_area

                    if overlap_ratio >= threshold:
                        instant_occupied = True
                        break

            # Update temporal buffer and perform majority voting
            self._update_temporal_state(slot, instant_occupied)

    def _update_temporal_state(self, slot, instant_occupied):
        """Update historical state and apply rolling-window smoothing."""
        slot_id = slot["id"]
        if slot_id not in self.history:
            self.history[slot_id] = []

        # Keep rolling window of last 15 frames
        self.history[slot_id].append(instant_occupied)
        if len(self.history[slot_id]) > 15:
            self.history[slot_id].pop(0)

        # Hysteresis: if occupied in >= 25% of recent frames, mark as occupied
        # This keeps the slot red during temporary YOLO misdetections / camera noise
        occupied_ratio = sum(self.history[slot_id]) / len(self.history[slot_id])
        slot["occupied"] = occupied_ratio >= 0.25

    def get_status(self):
        """
        Get current parking lot status.

        Returns:
            Tuple of (total_slots, occupied_count, available_count).
        """
        total = len(self.slots)
        occupied = sum(1 for s in self.slots if s["occupied"])
        available = total - occupied
        return total, occupied, available

    def get_slots(self):
        """
        Get the list of all parking slots with their current state.

        Returns:
            List of slot dicts with keys: id, points, occupied.
        """
        return self.slots
