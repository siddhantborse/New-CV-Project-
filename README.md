# Real-Time ADAS Perception System

A portfolio-grade ADAS (Advanced Driver Assistance System) that performs lane detection,
vehicle detection/tracking, proximity estimation, and driver alert generation on dashcam video.

---

## Project Structure

```
adas_system/
├── main.py                      # Entry point: orchestrates pipeline
├── config/
│   └── settings.py              # All tuneable parameters in one place
├── modules/
│   ├── lane_detection.py        # Classical CV lane pipeline
│   ├── object_detection.py      # YOLOv8 wrapper + NMS
│   ├── tracker.py               # Simple IoU-based multi-object tracker
│   ├── proximity.py             # Distance banding + risk scoring
│   └── alert_engine.py          # Rule-based alert generation
├── utils/
│   ├── video_io.py              # VideoCapture / VideoWriter helpers
│   ├── drawing.py               # All overlay / HUD rendering
│   └── geometry.py              # Shared geometry helpers
├── assets/
│   └── (fonts, icons if needed)
└── output/
    └── (processed videos written here)
```

---

## Setup

```bash
pip install ultralytics opencv-python-headless numpy scipy
```

YOLOv8 weights are auto-downloaded by `ultralytics` on first run.

---

## Usage

```bash
# Process a local video file
python main.py --input path/to/dashcam.mp4 --output output/result.mp4

# Use webcam
python main.py --input 0 --output output/webcam_result.mp4

# Tune confidence threshold
python main.py --input video.mp4 --conf 0.45 --iou 0.5
```

---

## Dataset Sources
- **BDD100K** – https://bdd-data.berkeley.edu/
- **KITTI Vision** – https://www.cvlibs.net/datasets/kitti/
- **Comma2k19** – https://github.com/commaai/comma2k19
- Any front-facing dashcam MP4 works out of the box.

---

## Module Overview

| Module | Technique |
|--------|-----------|
| Lane Detection | Grayscale → Gaussian blur → Canny → ROI mask → Hough lines → polynomial fit |
| Object Detection | YOLOv8n (nano) pretrained on COCO – classes: car, truck, bus, motorcycle |
| Tracking | IoU-based greedy assignment across frames with track lifecycle management |
| Proximity | Bounding-box-height heuristic mapped to Near / Medium / Far bands |
| Alerts | Stateful rule engine with cooldown timers to avoid alert flooding |
