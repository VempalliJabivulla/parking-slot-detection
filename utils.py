"""
utils.py - Drawing utilities and FPS counter for Parking Slot Detection.

Provides functions to render slot overlays, detection bounding boxes,
and an informational dashboard onto video frames.
"""

import time
import cv2
import numpy as np
from config import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_BLACK,
    COLOR_YELLOW,
    COLOR_DASHBOARD_BG,
    SLOT_OVERLAY_ALPHA,
    DASHBOARD_HEIGHT,
    FONT_SCALE,
    FONT_THICKNESS,
)


class FPSCounter:
    """Tracks frame timestamps and computes a rolling average FPS."""

    def __init__(self, avg_frames=30):
        """
        Args:
            avg_frames: Number of recent frames to average over.
        """
        self.avg_frames = avg_frames
        self.timestamps = []
        self.fps = 0.0

    def tick(self):
        """Record a new frame timestamp and update FPS."""
        now = time.time()
        self.timestamps.append(now)

        # Keep only the most recent timestamps
        if len(self.timestamps) > self.avg_frames:
            self.timestamps = self.timestamps[-self.avg_frames:]

        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self.fps = (len(self.timestamps) - 1) / elapsed

    def get_fps(self):
        """Return the current FPS value."""
        return self.fps


def draw_slots(frame, slots):
    """
    Draw semi-transparent colored overlays on parking slot polygons.

    Green = empty, Red = occupied.

    Args:
        frame: BGR image (numpy array), modified in-place.
        slots: List of slot dicts with 'points' and 'occupied' keys.

    Returns:
        The modified frame.
    """
    overlay = frame.copy()

    for slot in slots:
        color = COLOR_RED if slot["occupied"] else COLOR_GREEN
        pts = slot["points"]

        # Draw filled polygon on the overlay
        cv2.fillPoly(overlay, [pts], color)

        # Draw polygon border on the original frame
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        # Draw slot ID label
        centroid = pts.mean(axis=0).astype(int)
        slot_id = str(slot["id"])
        text_size = cv2.getTextSize(slot_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = centroid[0] - text_size[0] // 2
        text_y = centroid[1] + text_size[1] // 2
        cv2.putText(
            frame, slot_id, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA,
        )

    # Blend the overlay with the original frame
    cv2.addWeighted(overlay, SLOT_OVERLAY_ALPHA, frame, 1 - SLOT_OVERLAY_ALPHA, 0, frame)

    return frame


def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels around detected vehicles.

    Args:
        frame: BGR image (numpy array), modified in-place.
        detections: List of detection dicts from VehicleDetector.detect().

    Returns:
        The modified frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)

        # Label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 8),
            (x1 + text_size[0] + 4, y1),
            COLOR_YELLOW,
            -1,
        )

        # Label text
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLACK, 1, cv2.LINE_AA,
        )

    return frame


def draw_dashboard(frame, total, occupied, available, fps):
    """
    Render an informational dashboard bar at the top of the frame.

    Shows: Total Slots | Occupied | Available | FPS

    Args:
        frame: BGR image (numpy array), modified in-place.
        total: Total number of parking slots.
        occupied: Number of occupied slots.
        available: Number of available slots.
        fps: Current frames per second.

    Returns:
        The modified frame.
    """
    h, w = frame.shape[:2]
    bar_height = DASHBOARD_HEIGHT

    # Semi-transparent dark background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), COLOR_DASHBOARD_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Bottom border line
    cv2.line(frame, (0, bar_height), (w, bar_height), COLOR_GREEN, 2)

    # Text segments
    y_pos = bar_height - 18
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Total Slots
    text_total = f"Total: {total}"
    cv2.putText(frame, text_total, (20, y_pos), font, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS, cv2.LINE_AA)

    # Occupied (red)
    text_occ = f"Occupied: {occupied}"
    cv2.putText(frame, text_occ, (200, y_pos), font, FONT_SCALE, COLOR_RED, FONT_THICKNESS, cv2.LINE_AA)

    # Available (green)
    text_avail = f"Available: {available}"
    cv2.putText(frame, text_avail, (420, y_pos), font, FONT_SCALE, COLOR_GREEN, FONT_THICKNESS, cv2.LINE_AA)

    # FPS
    text_fps = f"FPS: {fps:.1f}"
    cv2.putText(frame, text_fps, (w - 160, y_pos), font, FONT_SCALE, COLOR_YELLOW, FONT_THICKNESS, cv2.LINE_AA)

    return frame
