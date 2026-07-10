"""
calibrate.py - Interactive Parking Slot Calibrator.

Opens a frame from your video and lets you click to define
parking slot boundaries. Saves coordinates to parking_slots.json.

Usage:
    python calibrate.py                     # Use webcam
    python calibrate.py path/to/video.mp4   # Use video file

Controls:
    Left Click  — Place a corner point (4 points per slot)
    Right Click — Undo last point
    N           — Finish current slot and start next
    S           — Save all slots to parking_slots.json
    R           — Reset all slots and start over
    Q / ESC     — Quit without saving
    Z           — Zoom in/out toggle
"""

import cv2
import numpy as np
import json
import sys
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
OUTPUT_FILE = "parking_slots.json"
WINDOW_NAME = "Parking Slot Calibrator"

# Colors (BGR)
COLOR_ACTIVE = (0, 255, 255)       # Yellow — current slot being drawn
COLOR_COMPLETE = (0, 200, 0)       # Green — completed slots
COLOR_POINT = (0, 140, 255)        # Orange — corner points
COLOR_TEXT_BG = (30, 30, 30)       # Dashboard background
COLOR_WHITE = (255, 255, 255)
COLOR_GUIDE = (100, 100, 100)      # Crosshair guide lines


class SlotCalibrator:
    """Interactive tool to define parking slot polygons by clicking."""

    def __init__(self, frame):
        """
        Args:
            frame: The reference BGR image to draw slots on.
        """
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.frame_h, self.frame_w = frame.shape[:2]

        self.slots = []            # List of completed slots: [{"id": int, "points": [[x,y],...]}]
        self.current_points = []   # Points for the slot currently being drawn
        self.slot_counter = 1      # Auto-incrementing slot ID

        self.mouse_pos = (0, 0)    # Current mouse position for crosshair
        self.show_crosshair = True

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add a corner point
            if len(self.current_points) < 4:
                self.current_points.append([x, y])
                print(f"  Point {len(self.current_points)}/4: ({x}, {y})")

                if len(self.current_points) == 4:
                    print(f"  ✓ Slot {self.slot_counter} complete! "
                          f"Press 'N' to confirm, Right-click to undo.")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Undo last point
            if self.current_points:
                removed = self.current_points.pop()
                print(f"  ✗ Removed point ({removed[0]}, {removed[1]})")

    def confirm_current_slot(self):
        """Save the current 4-point slot and start a new one."""
        if len(self.current_points) == 4:
            slot = {
                "id": self.slot_counter,
                "points": self.current_points.copy(),
            }
            self.slots.append(slot)
            print(f"\n  ✅ Slot {self.slot_counter} saved! "
                  f"({len(self.slots)} total)")
            self.slot_counter += 1
            self.current_points = []
        elif len(self.current_points) > 0:
            print(f"  ⚠ Need 4 points to complete a slot "
                  f"(have {len(self.current_points)})")
        else:
            print("  ⚠ No points placed yet. Click 4 corners.")

    def reset_all(self):
        """Clear all slots and start fresh."""
        self.slots = []
        self.current_points = []
        self.slot_counter = 1
        print("\n  🔄 All slots cleared. Starting fresh.")

    def save_to_json(self):
        """Save all completed slots to the JSON file."""
        if not self.slots:
            print("\n  ⚠ No slots to save!")
            return False

        data = {"slots": self.slots}

        # Backup existing file
        if os.path.exists(OUTPUT_FILE):
            backup = OUTPUT_FILE + ".backup"
            os.replace(OUTPUT_FILE, backup)
            print(f"  📋 Backed up existing file to {backup}")

        with open(OUTPUT_FILE, "w") as f:
            json.dump(data, f, indent=4)

        print(f"\n  💾 Saved {len(self.slots)} slots to '{OUTPUT_FILE}'")
        return True

    def render(self):
        """Draw everything on the display frame."""
        self.display_frame = self.original_frame.copy()
        overlay = self.display_frame.copy()

        # Draw completed slots (green, semi-transparent)
        for slot in self.slots:
            pts = np.array(slot["points"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], COLOR_COMPLETE)
            cv2.polylines(
                self.display_frame, [pts], True, COLOR_COMPLETE, 2,
            )
            # Slot ID at centroid
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(
                self.display_frame,
                str(slot["id"]),
                (centroid[0] - 8, centroid[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA,
            )

        # Blend overlay for completed slots
        cv2.addWeighted(overlay, 0.25, self.display_frame, 0.75, 0, self.display_frame)

        # Draw current slot points and lines (yellow)
        if self.current_points:
            pts = np.array(self.current_points, dtype=np.int32)

            # Draw lines connecting placed points
            for i in range(len(self.current_points) - 1):
                cv2.line(
                    self.display_frame,
                    tuple(self.current_points[i]),
                    tuple(self.current_points[i + 1]),
                    COLOR_ACTIVE, 2, cv2.LINE_AA,
                )

            # If 4 points, close the polygon
            if len(self.current_points) == 4:
                cv2.line(
                    self.display_frame,
                    tuple(self.current_points[3]),
                    tuple(self.current_points[0]),
                    COLOR_ACTIVE, 2, cv2.LINE_AA,
                )
                # Fill with transparent yellow
                overlay2 = self.display_frame.copy()
                cv2.fillPoly(overlay2, [pts], COLOR_ACTIVE)
                cv2.addWeighted(overlay2, 0.2, self.display_frame, 0.8, 0, self.display_frame)

            # Draw guide line from last point to mouse
            elif len(self.current_points) < 4:
                cv2.line(
                    self.display_frame,
                    tuple(self.current_points[-1]),
                    self.mouse_pos,
                    COLOR_ACTIVE, 1, cv2.LINE_AA,
                )

            # Draw corner points as circles
            for i, pt in enumerate(self.current_points):
                cv2.circle(self.display_frame, tuple(pt), 6, COLOR_POINT, -1, cv2.LINE_AA)
                cv2.circle(self.display_frame, tuple(pt), 6, COLOR_WHITE, 1, cv2.LINE_AA)

        # Draw crosshair at mouse position
        if self.show_crosshair:
            mx, my = self.mouse_pos
            cv2.line(self.display_frame, (mx, 0), (mx, self.frame_h), COLOR_GUIDE, 1)
            cv2.line(self.display_frame, (0, my), (self.frame_w, my), COLOR_GUIDE, 1)

        # Dashboard bar at top
        bar_h = 50
        cv2.rectangle(self.display_frame, (0, 0), (self.frame_w, bar_h), COLOR_TEXT_BG, -1)

        # Info text
        info_parts = [
            f"Slots: {len(self.slots)}",
            f"Current: {len(self.current_points)}/4 points",
            f"Pos: ({self.mouse_pos[0]}, {self.mouse_pos[1]})",
        ]
        info_text = "  |  ".join(info_parts)
        cv2.putText(
            self.display_frame, info_text, (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA,
        )

        # Controls hint (right side)
        controls = "L-Click: Point | R-Click: Undo | N: Next | S: Save | Q: Quit"
        text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(
            self.display_frame, controls,
            (self.frame_w - text_size[0] - 15, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA,
        )

        return self.display_frame


def get_frame(source):
    """
    Get a reference frame from the video source.

    Args:
        source: 0 for webcam, or path to video file.

    Returns:
        BGR frame (numpy array), or None on failure.
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Cannot read frame from source.")
        return None

    return frame


def main():
    """Main entry point for the calibration tool."""
    print("\n" + "=" * 55)
    print("  🅿️  PARKING SLOT CALIBRATOR")
    print("=" * 55)

    # Determine video source
    if len(sys.argv) > 1:
        source = sys.argv[1]
        if not os.path.isfile(source):
            print(f"[ERROR] File not found: {source}")
            sys.exit(1)
        print(f"  Source: {source}")
    else:
        source = 0
        print("  Source: Webcam")

    # Get reference frame
    print("  Loading frame...")
    frame = get_frame(source)

    if frame is None:
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"  Frame size: {w}x{h}")

    # Check for existing slots
    if os.path.exists(OUTPUT_FILE):
        print(f"\n  ⚠ Existing '{OUTPUT_FILE}' found.")
        print(f"    It will be backed up when you save new slots.")

    print(f"\n  INSTRUCTIONS:")
    print(f"  1. Left-click 4 corners of each parking slot (in order)")
    print(f"  2. Press 'N' to confirm the slot and start the next one")
    print(f"  3. Press 'S' to save all slots to {OUTPUT_FILE}")
    print(f"  4. Right-click to undo the last point")
    print(f"  5. Press 'R' to reset all slots")
    print(f"  6. Press 'Q' or ESC to quit")
    print(f"\n  Start clicking slot corners...\n")

    # Initialize calibrator
    calibrator = SlotCalibrator(frame)

    # Setup window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(w, 1400), min(h, 900))
    cv2.setMouseCallback(WINDOW_NAME, calibrator.mouse_callback)

    while True:
        display = calibrator.render()
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:  # Q or ESC
            if calibrator.slots:
                print("\n  ⚠ You have unsaved slots!")
                print("    Press 'S' to save or 'Q' again to quit.")
            break

        elif key == ord("n"):
            calibrator.confirm_current_slot()

        elif key == ord("s"):
            if calibrator.save_to_json():
                print("  You can now run: streamlit run app.py")

        elif key == ord("r"):
            calibrator.reset_all()

    cv2.destroyAllWindows()
    print("\n  Calibrator closed.\n")


if __name__ == "__main__":
    main()
