"""
Streamlit demo for real-time multi-object tracking.

Run:
    streamlit run scripts/run_streamlit.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import YOLOv8Detector
from src.tracking.sort import SORTTracker
from src.tracking.bytetrack import ByteTracker
from src.visualization.draw import draw_tracks, draw_detections, resize_for_display

# Page config

st.set_page_config(
    page_title="Real-time Object Tracker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Real-time Multi-Object Tracker")
st.markdown(
    "Upload a video and watch **SORT** or **ByteTrack** track pedestrians frame-by-frame. "
    "Built with YOLOv8 detection + Kalman filter + Hungarian assignment."
)

 
# Sidebar controls
 
st.sidebar.header("Configuration")

tracker_name = st.sidebar.selectbox(
    "Tracker algorithm",
    ["bytetrack", "sort"],
    index=0,
    help="ByteTrack uses two-stage matching with low-confidence detections.",
)

yolo_model = st.sidebar.selectbox(
    "YOLOv8 model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
    help="Larger models = better accuracy, slower speed.",
)

conf_threshold = st.sidebar.slider(
    "Detection confidence threshold",
    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
)

max_age = st.sidebar.slider(
    "Track max age (frames)",
    min_value=1, max_value=60, value=30,
    help="How many frames a track survives without a match.",
)

min_hits = st.sidebar.slider(
    "Min hits to confirm track",
    min_value=1, max_value=10, value=3,
)

show_dets = st.sidebar.checkbox("Show raw detections", value=False)
max_frames = st.sidebar.number_input(
    "Max frames to process (0 = all)", min_value=0, value=300, step=50,
)
max_frames = int(max_frames) or None

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**ByteTrack params** (only when ByteTrack selected)"
)
high_thresh = st.sidebar.slider("High confidence threshold", 0.1, 0.95, 0.5, 0.05)
low_thresh  = st.sidebar.slider("Low confidence threshold",  0.01, 0.5,  0.1, 0.01)

 
# Video upload
 
uploaded = st.file_uploader(
    "Upload a video file", type=["mp4", "avi", "mov", "mkv"]
)

if uploaded is None:
    st.info("👆 Upload a video to start tracking.")
    st.stop()

# Save to temp file

tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
tmp.write(uploaded.read())
tmp.flush()
video_path = tmp.name

 
# Build detector & tracker (cached by params)
 
@st.cache_resource
def get_detector(model_path: str, conf: float) -> YOLOv8Detector:
    return YOLOv8Detector(
        model_path=model_path,
        confidence=conf,
        target_classes=[0],
        device="cpu",
    )


def build_tracker(name: str):
    if name == "sort":
        return SORTTracker(max_age=max_age, min_hits=min_hits, iou_threshold=0.3)
    return ByteTracker(
        high_thresh=high_thresh,
        low_thresh=low_thresh,
        match_thresh=0.8,
        max_age=max_age,
        min_hits=min_hits,
    )


detector = get_detector(yolo_model, conf_threshold)
tracker  = build_tracker(tracker_name)

 
# Run tracking
 
run_btn = st.button("▶ Run Tracking", type="primary", use_container_width=True)
if not run_btn:
    st.stop()

cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

# UI placeholders
col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    st.markdown("### Live stats")
    stat_frame   = st.metric("Frame",          "0")
    stat_tracks  = st.metric("Active tracks",  "0")
    stat_ids     = st.metric("Total IDs seen", "0")
    stat_fps     = st.metric("Processing FPS", "0")

progress_bar  = st.progress(0.0)
status_text   = st.empty()

frame_id   = 0
all_ids    = set()
frame_times: list = []
t_pipeline = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if max_frames and frame_id > max_frames:
            break

        t0   = time.perf_counter()
        dets = detector.detect(frame)
        inp  = dets[:, :5] if len(dets) else np.empty((0, 5), dtype=np.float32)
        tracks = tracker.update(inp)
        frame_times.append(time.perf_counter() - t0)

        for t in tracks:
            all_ids.add(t.track_id)

        # Annotate
        vis = frame.copy()
        if show_dets and len(dets):
            vis = draw_detections(vis, dets, color=(160, 160, 160),
                                  show_conf=False, thickness=1)
        vis = draw_tracks(vis, tracks, show_id=True)

        # FPS overlay
        if len(frame_times) >= 5:
            proc_fps = 1.0 / (sum(frame_times[-10:]) / min(10, len(frame_times)))
            cv2.putText(vis, f"FPS: {proc_fps:.1f}  Tracks: {len(tracks)}",
                        (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        vis_rgb = cv2.cvtColor(
            resize_for_display(vis, max_width=960), cv2.COLOR_BGR2RGB
        )
        frame_placeholder.image(vis_rgb, use_container_width=True)

        # Stats (update every 5 frames to reduce overhead)
        if frame_id % 5 == 0:
            proc_fps_now = (1.0 / (sum(frame_times[-10:]) / min(10, len(frame_times)))
                            if len(frame_times) >= 5 else 0.0)
            with col2:
                stat_frame   = st.metric("Frame",          str(frame_id))
                stat_tracks  = st.metric("Active tracks",  str(len(tracks)))
                stat_ids     = st.metric("Total IDs seen", str(len(all_ids)))
                stat_fps     = st.metric("Processing FPS", f"{proc_fps_now:.1f}")

        progress = frame_id / (max_frames or total) if (max_frames or total) > 0 else 0
        progress_bar.progress(min(progress, 1.0))
        status_text.text(
            f"Processing frame {frame_id}"
            + (f" / {max_frames or total}" if (max_frames or total) else "")
        )

finally:
    cap.release()

 
# Summary
 
elapsed = sum(frame_times)
st.success(
    f"✅ Done! Processed **{frame_id} frames** in **{elapsed:.1f}s** "
    f"({frame_id/elapsed:.1f} fps avg) · "
    f"Total unique track IDs: **{len(all_ids)}** · "
    f"Tracker: **{tracker_name}**"
)
