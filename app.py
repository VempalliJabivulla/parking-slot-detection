"""
app.py - Streamlit UI for Parking Slot Detection System.

Provides a polished web interface with:
    1. Upload Video  — process an MP4 file with live preview
    2. Live Camera   — real-time detection from webcam

Auto-Calibration:
    On first upload, the system automatically detects parking slot positions
    from the video's first frame using YOLO vehicle detection + gap analysis.
    No manual coordinate entry needed.

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import json
import os
import time

from detector import VehicleDetector
from parking import ParkingManager
from utils import FPSCounter, draw_slots, draw_detections, draw_dashboard
from auto_calibrate import auto_detect_slots, save_slots
import config as cfg


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Parking Slot Detection",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .header-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 212, 170, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .header-banner h1 {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #00d4aa, #00b4d8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
    }
    .header-banner p { color: #94a3b8; font-size: 0.95rem; margin: 0; }

    .metric-card {
        background: linear-gradient(145deg, #1a1f2e, #151926);
        border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px); box-shadow: 0 6px 25px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.75rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.4rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-total .metric-label { color: #94a3b8; }
    .metric-total .metric-value { color: #e2e8f0; }
    .metric-occupied .metric-label { color: #fca5a5; }
    .metric-occupied .metric-value { color: #ef4444; }
    .metric-available .metric-label { color: #86efac; }
    .metric-available .metric-value { color: #22c55e; }
    .metric-fps .metric-label { color: #fde68a; }
    .metric-fps .metric-value { color: #f59e0b; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #141926 100%);
    }

    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 500;
    }
    .status-ready {
        background: rgba(34, 197, 94, 0.15); color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-processing {
        background: rgba(59, 130, 246, 0.15); color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .info-box {
        background: linear-gradient(145deg, #1a1f2e, #151926);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 12px; padding: 2rem; text-align: center; margin: 2rem 0;
    }
    .info-box .icon { font-size: 3rem; margin-bottom: 0.8rem; }
    .info-box p { color: #94a3b8; margin: 0.3rem 0; }

    .stButton > button {
        border-radius: 8px; font-weight: 600;
        padding: 0.5rem 1.5rem; transition: all 0.2s ease;
    }

    .calib-success {
        background: linear-gradient(145deg, #0d2818, #0f3520);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    }
    .calib-success p { color: #86efac; margin: 0; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cached Resources
# ─────────────────────────────────────────────
@st.cache_resource
def load_detector(model_name):
    """Load and cache the YOLOv8 model."""
    return VehicleDetector(model_path=model_name)


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
defaults = {
    "processing": False,
    "total": 0, "occupied": 0, "available": 0, "fps": 0.0,
    "slots_calibrated": False,
    "auto_slots": None,
    "current_model": "yolov8s.pt",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🅿️ Parking Detection")
    st.markdown("---")

    mode = st.radio("**Mode**", ["📹 Upload Video", "📷 Live Camera"], index=0)

    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")

    # Dropdown to choose YOLO model
    model_option = st.selectbox(
        "**YOLO Model**",
        ["Nano (Fastest)", "Small (Balanced)", "Medium (Accurate)"],
        index=1,
        help="Nano is fast but misses small/distant cars. Medium is highly accurate but slower.",
    )
    
    # Map selection to model path
    model_mapping = {
        "Nano (Fastest)": "yolov8n.pt",
        "Small (Balanced)": "yolov8s.pt",
        "Medium (Accurate)": "yolov8m.pt"
    }
    selected_model = model_mapping[model_option]

    # If the user changed the model, force re-calibration of slots
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.session_state.slots_calibrated = False
        st.session_state.auto_slots = None
        if os.path.exists(cfg.PARKING_SLOTS_FILE):
            os.remove(cfg.PARKING_SLOTS_FILE)
        st.toast(f"Model switched to {selected_model}. Slots cleared for re-calibration.", icon="🔄")
        st.rerun()

    conf_threshold = st.slider(
        "Confidence Threshold", 0.05, 1.00, 0.25, 0.05,
        help="Lower values detect more vehicles, higher values avoid false detections",
    )
    overlap_threshold = st.slider(
        "Overlap Threshold", 0.05, 1.00, 0.25, 0.05,
        help="Percentage overlap of vehicle bottom wheels area with slot to mark as occupied",
    )

    st.markdown("---")

    uploaded_video = None
    if "Upload" in mode:
        st.markdown("### 📁 Video Input")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
        )

    if "Upload" in mode or "Live" in mode:
        save_output = st.checkbox("💾 Save Processed Video", value=False)
    else:
        save_output = False

    # Slot status
    st.markdown("---")
    if os.path.exists(cfg.PARKING_SLOTS_FILE):
        with open(cfg.PARKING_SLOTS_FILE) as f:
            data = json.load(f)
        n = len(data.get("slots", []))
        st.success(f"✅ {n} parking slots loaded")
        if st.button("🔄 Re-calibrate Slots", use_container_width=True):
            st.session_state.slots_calibrated = False
            st.session_state.auto_slots = None
            if os.path.exists(cfg.PARKING_SLOTS_FILE):
                os.remove(cfg.PARKING_SLOTS_FILE)
            st.rerun()
    else:
        st.info("🎯 Slots will auto-detect on first video upload")

    st.markdown(
        "<div style='text-align:center;color:#64748b;font-size:0.75rem;margin-top:1rem;'>"
        "Powered by YOLOv8 + OpenCV</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🅿️ Parking Slot Detection System</h1>
    <p>Real-time vehicle detection &amp; parking occupancy monitoring powered by YOLOv8</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def render_metrics(total, occupied, available, fps):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card metric-total">'
                     f'<div class="metric-label">Total Slots</div>'
                     f'<div class="metric-value">{total}</div></div>',
                     unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card metric-occupied">'
                     f'<div class="metric-label">Occupied</div>'
                     f'<div class="metric-value">{occupied}</div></div>',
                     unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card metric-available">'
                     f'<div class="metric-label">Available</div>'
                     f'<div class="metric-value">{available}</div></div>',
                     unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card metric-fps">'
                     f'<div class="metric-label">FPS</div>'
                     f'<div class="metric-value">{fps:.1f}</div></div>',
                     unsafe_allow_html=True)


def process_frame(frame, detector, parking_manager, fps_counter,
                  conf_thresh, overlap_thresh):
    fps_counter.tick()
    detections = detector.detect(frame, conf=conf_thresh)
    h, w = frame.shape[:2]
    parking_manager.check_occupancy_fast(detections, (h, w), overlap_thresh=overlap_thresh)
    total, occupied, available = parking_manager.get_status()
    fps = fps_counter.get_fps()
    frame = draw_slots(frame, parking_manager.get_slots())
    frame = draw_detections(frame, detections)
    frame = draw_dashboard(frame, total, occupied, available, fps)
    return frame, total, occupied, available, fps


def save_temp_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    return tfile.name


def run_auto_calibration(temp_path, detector):
    """
    Run auto-calibration on the first frame of the video.
    Shows progress and preview in the Streamlit UI.
    """
    cap = cv2.VideoCapture(temp_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("❌ Cannot read frame from video.")
        return False

    with st.spinner("🎯 Auto-detecting parking slots from first frame..."):
        slots = auto_detect_slots(frame, detector, conf=0.20)

    if not slots:
        st.error("❌ Could not detect any parking slots. Try a video with more visible cars.")
        return False

    # Save to file
    save_slots(slots, cfg.PARKING_SLOTS_FILE)
    st.session_state.auto_slots = slots
    st.session_state.slots_calibrated = True

    # Show preview of detected slots
    preview = frame.copy()
    overlay = preview.copy()
    for slot in slots:
        pts = np.array(slot["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
        cv2.polylines(preview, [pts], True, (0, 255, 0), 2)
        centroid = pts.mean(axis=0).astype(int)
        cv2.putText(preview, str(slot["id"]),
                    (centroid[0] - 8, centroid[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.25, preview, 0.75, 0, preview)
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    st.markdown(
        f'<div class="calib-success">'
        f'<p>✅ Auto-detected <strong>{len(slots)}</strong> parking slots!</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.image(preview_rgb, caption="Detected parking slot positions", use_container_width=True)

    return True


# ─────────────────────────────────────────────
# Load detector based on selected model
# ─────────────────────────────────────────────
detector = load_detector(st.session_state.current_model)


# ═════════════════════════════════════════════
# MODE: Upload Video
# ═════════════════════════════════════════════
if "Upload" in mode:
    metrics_placeholder = st.empty()
    with metrics_placeholder.container():
        render_metrics(st.session_state.total, st.session_state.occupied,
                       st.session_state.available, st.session_state.fps)

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    download_placeholder = st.empty()

    if uploaded_video is not None:
        temp_path = save_temp_video(uploaded_video)

        # ── Auto-calibrate if no slots exist ──
        needs_calibration = not os.path.exists(cfg.PARKING_SLOTS_FILE)

        if needs_calibration:
            st.markdown("### 🎯 Automatic Slot Detection")
            st.markdown(
                "No parking slots defined yet. The system will analyze your video's "
                "first frame to automatically detect all parking slot positions."
            )

            if st.button("🎯 Auto-Detect Parking Slots", use_container_width=True, type="primary"):
                success = run_auto_calibration(temp_path, detector)
                if success:
                    st.info("👆 Slots detected! Click **Start Processing** below to begin.")
                    st.rerun()

        else:
            # Slots exist — show processing controls
            parking_manager = ParkingManager()

            col_start, col_stop = st.columns(2)
            start_btn = col_start.button("▶️  Start Processing", use_container_width=True, type="primary")
            stop_btn = col_stop.button("⏹️  Stop", use_container_width=True)

            if stop_btn:
                st.session_state.processing = False
            if start_btn:
                st.session_state.processing = True

            if st.session_state.processing:
                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    st.error("❌ Cannot open video file.")
                    st.session_state.processing = False
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps_counter = FPSCounter()

                    status_placeholder.markdown(
                        '<span class="status-badge status-processing">⏳ Processing...</span>',
                        unsafe_allow_html=True,
                    )

                    writer = None
                    output_path = None
                    if save_output:
                        output_path = os.path.join(tempfile.gettempdir(), "parking_output.avi")
                        fourcc = cv2.VideoWriter_fourcc(*cfg.OUTPUT_CODEC)
                        writer = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_w, frame_h))

                    prog_bar = progress_placeholder.progress(0, text="Starting...")
                    frame_num = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_num += 1
                        processed, total, occupied, available, fps = process_frame(
                            frame, detector, parking_manager, fps_counter,
                            conf_threshold, overlap_threshold,
                        )

                        if writer is not None:
                            writer.write(processed)

                        frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        with metrics_placeholder.container():
                            render_metrics(total, occupied, available, fps)

                        progress = frame_num / total_frames if total_frames > 0 else 0
                        prog_bar.progress(
                            min(progress, 1.0),
                            text=f"Frame {frame_num}/{total_frames} • {progress*100:.0f}%",
                        )

                    cap.release()
                    if writer:
                        writer.release()

                    st.session_state.processing = False
                    status_placeholder.markdown(
                        '<span class="status-badge status-ready">✅ Complete</span>',
                        unsafe_allow_html=True,
                    )
                    progress_placeholder.empty()

                    if output_path and os.path.exists(output_path):
                        with open(output_path, "rb") as f:
                            download_placeholder.download_button(
                                "📥  Download Processed Video", data=f,
                                file_name="parking_processed.avi", mime="video/avi",
                                use_container_width=True,
                            )

        try:
            os.unlink(temp_path)
        except (PermissionError, FileNotFoundError):
            pass
    else:
        st.markdown("""
        <div class="info-box">
            <div class="icon">📹</div>
            <p style="font-size:1.1rem; color:#e2e8f0; font-weight:500;">
                Upload a video to get started</p>
            <p>The system will automatically detect parking slots from the first frame</p>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# MODE: Live Camera
# ═════════════════════════════════════════════
elif "Live" in mode:
    # Ensure camera is released if we change modes (handled by session state)
    if "cap" in st.session_state and not st.session_state.processing:
        st.session_state.cap.release()
        del st.session_state.cap

    parking_manager = ParkingManager()

    metrics_placeholder = st.empty()
    with metrics_placeholder.container():
        render_metrics(st.session_state.total, st.session_state.occupied,
                       st.session_state.available, st.session_state.fps)

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    col_start, col_stop, col_calib = st.columns(3)
    start_btn = col_start.button("📷  Start Camera", use_container_width=True, type="primary")
    stop_btn = col_stop.button("⏹️  Stop Camera", use_container_width=True)
    calib_btn = col_calib.button("🎯  Auto-Detect Slots", use_container_width=True)

    if stop_btn:
        st.session_state.processing = False
        if "cap" in st.session_state:
            st.session_state.cap.release()
            del st.session_state.cap
        st.rerun()

    if start_btn:
        st.session_state.processing = True
        st.rerun()

    # Auto-calibrate from webcam
    if calib_btn:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Discard first 25 warm-up frames so camera exposure/focus settles
            with st.spinner("📷 Initializing camera sensor..."):
                for _ in range(25):
                    cap.read()
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                with st.spinner("🎯 Auto-detecting slots from camera..."):
                    slots = auto_detect_slots(frame, detector, conf=0.20)
                if slots:
                    save_slots(slots, cfg.PARKING_SLOTS_FILE)
                    st.success(f"✅ Detected {len(slots)} parking slots!")
                    parking_manager = ParkingManager()
                    st.rerun()
                else:
                    st.warning("⚠️ No vehicles detected. Adjust your camera angle or light.")
            else:
                st.error("❌ Failed to grab frame from webcam.")
        else:
            st.error("❌ Cannot open webcam.")

    # Frame-by-frame non-blocking rendering loop
    if st.session_state.processing:
        # Lazy initialize camera session
        if "cap" not in st.session_state:
            with st.spinner("📷 Connecting to camera..."):
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Cannot open webcam. Check connections and permissions.")
                    st.session_state.processing = False
                    st.rerun()
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    # Sensor warm-up
                    for _ in range(15):
                        cap.read()
                    st.session_state.cap = cap
                    st.session_state.fps_counter = FPSCounter()

        # Read and process single frame
        cap = st.session_state.cap
        ret, frame = cap.read()
        
        if ret:
            status_placeholder.markdown(
                '<span class="status-badge status-processing">🔴 Live Feed</span>',
                unsafe_allow_html=True,
            )
            processed, total, occupied, available, fps = process_frame(
                frame, detector, parking_manager, st.session_state.fps_counter,
                conf_threshold, overlap_threshold,
            )
            
            frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            with metrics_placeholder.container():
                render_metrics(total, occupied, available, fps)
            
            # Tiny sleep to yield control and limit CPU load
            time.sleep(0.01)
            # Rerun the script to grab the next frame (keeps Streamlit interactive)
            st.rerun()
        else:
            st.warning("⚠️ Connection to webcam lost.")
            st.session_state.processing = False
            if "cap" in st.session_state:
                st.session_state.cap.release()
                del st.session_state.cap
            st.rerun()

    elif not st.session_state.processing:
        if not os.path.exists(cfg.PARKING_SLOTS_FILE):
            st.markdown("""
            <div class="info-box">
                <div class="icon">📷</div>
                <p style="font-size:1.1rem; color:#e2e8f0; font-weight:500;">
                    No slots defined yet</p>
                <p>Click <strong>Auto-Detect Slots</strong> to scan from camera,
                   or upload a video first</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <div class="icon">📷</div>
                <p style="font-size:1.1rem; color:#e2e8f0; font-weight:500;">Ready to detect</p>
                <p>Click <strong>Start Camera</strong> to begin live detection</p>
            </div>""", unsafe_allow_html=True)
