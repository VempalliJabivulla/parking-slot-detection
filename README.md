# Parking Slot Detection System

Real-time parking slot occupancy detection using **YOLOv8** and **OpenCV**.

Detects vehicles (cars, motorcycles, buses, trucks) and maps them to predefined parking slot coordinates to determine which slots are occupied or available.

---

## Features

- **Live Camera Mode** — real-time detection from webcam
- **Upload Video Mode** — process MP4 videos frame-by-frame with save option
- **Visual Overlays** — green (empty) and red (occupied) slot highlighting
- **Dashboard** — live display of Total, Occupied, Available slots + FPS
- **Modular Architecture** — clean separation of concerns across 5 modules

## Project Structure

```
parking-slot-detection/
├── app.py                 # Main application (entry point)
├── detector.py            # YOLOv8 vehicle detection
├── parking.py             # Parking slot management & occupancy
├── utils.py               # Drawing utilities & FPS counter
├── config.py              # Configuration constants
├── parking_slots.json     # Predefined slot coordinates
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, the YOLOv8 Nano model (`yolov8n.pt`, ~6MB) will be automatically downloaded by Ultralytics.

### 2. Configure Parking Slots

Edit `parking_slots.json` to match your camera's view of the parking lot. Each slot is defined by 4 corner points:

```json
{
    "slots": [
        {
            "id": 1,
            "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        }
    ]
}
```

**Tips for finding coordinates:**
- Open a frame from your video/camera in an image editor
- Note the pixel coordinates of each parking slot's four corners
- Points should be in clockwise or counter-clockwise order

### 3. Run the Application

```bash
python app.py
```

## Usage

### Live Camera Mode
1. Select `[1] Live Camera` from the menu
2. The webcam feed will open with slot overlays and detection boxes
3. Press `q` to quit

### Upload Video Mode
1. Select `[2] Upload Video` from the menu
2. Enter the path to your MP4 video file
3. Choose whether to save the processed output
4. Press `q` to stop, `p` to pause/resume

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model variant |
| `CONFIDENCE_THRESHOLD` | `0.4` | Min detection confidence |
| `OVERLAP_THRESHOLD` | `0.3` | Min overlap to mark slot occupied |
| `SLOT_OVERLAY_ALPHA` | `0.35` | Transparency of slot overlays |

## Requirements

- Python 3.8+
- OpenCV 4.8+
- Ultralytics 8.0+
- NumPy 1.24+
- Webcam (for Live Camera mode)
