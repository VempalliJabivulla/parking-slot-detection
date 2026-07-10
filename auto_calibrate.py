"""
auto_calibrate.py - Automatic parking slot detection from video frames.

Analyzes the first frame using YOLOv8 vehicle detections to:
  1. Detect all parked vehicles
  2. Filter duplicate overlapping boxes (Non-Maximum Suppression)
  3. Cluster vehicles into slanted rows using line regression to handle perspective slants
  4. Generate and normalize slots along the line of each row
  5. Fill gaps between cars and extend borders dynamically
  6. Generate and save a clean parking_slots.json
"""

import numpy as np
import json
import os


def compute_iou(box1, box2):
    """Compute 2D Intersection-over-Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0, xi2 - xi1)
    inter_w_val = inter_w
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w_val * inter_h

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


def filter_duplicate_boxes(bboxes, iou_threshold=0.35):
    """Filter duplicate or highly overlapping bounding boxes."""
    sorted_boxes = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep = []

    for box in sorted_boxes:
        overlap = False
        for kept in keep:
            if compute_iou(box, kept) > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(box)

    return keep


def auto_detect_slots(frame, detector, conf=0.25, padding=4):
    """
    Automatically detect parking slot positions from a single frame using
    linear regression to fit slanted rows and fill gaps.
    """
    h, w = frame.shape[:2]

    # Step 1: Detect vehicles
    detections = detector.detect(frame, conf=conf)
    if not detections:
        print("[AUTO-CALIBRATE] No vehicles detected. Cannot auto-calibrate.")
        return []

    raw_bboxes = [d["bbox"] for d in detections]
    bboxes = filter_duplicate_boxes(raw_bboxes, iou_threshold=0.35)
    print(f"[AUTO-CALIBRATE] Detected {len(bboxes)} unique vehicles after duplicate filter.")

    # Compute bounding box centers and sizes
    cars = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        car_w = x2 - x1
        car_h = y2 - y1
        cars.append({
            "bbox": bbox,
            "cx": cx,
            "cy": cy,
            "w": car_w,
            "h": car_h
        })

    # Sort cars from left to right to build horizontal rows
    cars.sort(key=lambda c: c["cx"])

    # Step 2: Cluster cars into rows by chain-tracking
    rows = []
    for car in cars:
        placed = False
        # Find if this car fits into an existing row
        best_row_idx = -1
        min_y_dist = float("inf")

        for idx, row in enumerate(rows):
            last_car = row[-1]
            dx = car["cx"] - last_car["cx"]
            dy = abs(car["cy"] - last_car["cy"])

            # A car belongs to a row if Y-distance is small relative to height,
            # and the slant slope between successive cars is gentle (< 0.25)
            if dx > 0:
                slope = dy / dx
                if dy < last_car["h"] * 0.70 and slope < 0.25:
                    if dy < min_y_dist:
                        min_y_dist = dy
                        best_row_idx = idx

        if best_row_idx != -1:
            rows[best_row_idx].append(car)
        else:
            rows.append([car])

    # Filter out noise rows (require at least 2 cars per row to define a pattern)
    rows = [r for r in rows if len(r) >= 2]
    print(f"[AUTO-CALIBRATE] Found {len(rows)} parking rows after clustering.")

    all_slots = []
    slot_id = 1

    # Step 3: Fit slanted row lines and generate slots
    for row_idx, row_cars in enumerate(rows):
        # Sort row left-to-right
        row_cars.sort(key=lambda c: c["cx"])

        # Fit a line y = m*x + c representing the slanted row center
        cx_vals = [c["cx"] for c in row_cars]
        cy_vals = [c["cy"] for c in row_cars]
        
        if len(row_cars) >= 2:
            m, c_val = np.polyfit(cx_vals, cy_vals, 1)
        else:
            m, c_val = 0, cy_vals[0]

        # Median width and height of vehicles in this row
        row_med_w = np.median([c["w"] for c in row_cars])
        row_med_h = np.median([c["h"] for c in row_cars])

        # Helper to generate slot points at a given center X position
        def make_slot_at(cx):
            # Compute slanted Y center at this X position
            cy = m * cx + c_val
            # Half-dimensions with padding
            hw = (row_med_w / 2) + padding
            hh = (row_med_h / 2) + padding
            
            # Form rectangle vertices
            x1_s = int(cx - hw)
            x2_s = int(cx + hw)
            y1_s = int(cy - hh)
            y2_s = int(cy + hh)

            # Clamp coordinates to frame limits
            x1_s = max(0, min(w - 1, x1_s))
            x2_s = max(0, min(w - 1, x2_s))
            y1_s = max(0, min(h - 1, y1_s))
            y2_s = max(0, min(h - 1, y2_s))

            return [[x1_s, y1_s], [x2_s, y1_s], [x2_s, y2_s], [x1_s, y2_s]]

        # Generate slots for detected vehicles
        generated_x_centers = []
        for car in row_cars:
            all_slots.append({
                "id": slot_id,
                "points": make_slot_at(car["cx"])
            })
            generated_x_centers.append(car["cx"])
            slot_id += 1

        # Fill gaps between adjacent detected vehicles in the row
        for i in range(len(row_cars) - 1):
            cx1 = row_cars[i]["cx"]
            cx2 = row_cars[i+1]["cx"]
            gap_w = (cx2 - row_cars[i]["w"]/2) - (cx1 + row_cars[i+1]["w"]/2)

            # Check if an empty slot fits in the gap
            if gap_w > row_med_w * 0.75:
                num_slots = max(1, round(gap_w / (row_med_w + padding*2)))
                step = (cx2 - cx1) / (num_slots + 1)
                for j in range(1, num_slots + 1):
                    empty_cx = cx1 + j * step
                    all_slots.append({
                        "id": slot_id,
                        "points": make_slot_at(empty_cx)
                    })
                    slot_id += 1

        # Extend rows beyond the left and right edges only for long established rows (>= 5 cars)
        # This prevents short/accidental rows from extending empty slots into active roadways
        if len(row_cars) >= 5:
            # Left extension
            leftmost_cx = row_cars[0]["cx"]
            current_cx = leftmost_cx - (row_med_w + padding * 2)
            ext_count = 0
            while current_cx > (row_med_w / 2) and ext_count < 3:
                all_slots.append({
                    "id": slot_id,
                    "points": make_slot_at(current_cx)
                })
                slot_id += 1
                current_cx -= (row_med_w + padding * 2)
                ext_count += 1

            # Right extension
            rightmost_cx = row_cars[-1]["cx"]
            current_cx = rightmost_cx + (row_med_w + padding * 2)
            ext_count = 0
            while current_cx < w - (row_med_w / 2) and ext_count < 3:
                all_slots.append({
                    "id": slot_id,
                    "points": make_slot_at(current_cx)
                })
                slot_id += 1
                current_cx += (row_med_w + padding * 2)
                ext_count += 1

    # Renumber slots sequentially, sorting top-to-bottom and left-to-right
    all_slots.sort(key=lambda s: (s["points"][0][1], s["points"][0][0]))
    for i, slot in enumerate(all_slots):
        slot["id"] = i + 1

    print(f"[AUTO-CALIBRATE] Created {len(all_slots)} total slots with perspective fitting.")
    return all_slots


def save_slots(slots, filepath="parking_slots.json"):
    """Save auto-detected slots to a JSON file."""
    if os.path.exists(filepath):
        backup = filepath + ".backup"
        os.replace(filepath, backup)

    data = {"slots": slots}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[AUTO-CALIBRATE] Saved {len(slots)} slots to '{filepath}'")


def auto_calibrate_from_video(video_path, detector, output_file="parking_slots.json"):
    """Auto-detect slots from the first frame of a video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Cannot read frame from video.")
        return []

    slots = auto_detect_slots(frame, detector)
    if slots:
        save_slots(slots, output_file)

    return slots
