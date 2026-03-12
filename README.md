# Smart Co-Pilot: Real-Time ADAS Perception System

**🌟 Status: First Working Prototype (Milestone 1)**

A portfolio-grade ADAS pipeline that takes any front-facing dashcam video and outputs
a processed video with lane overlays, vehicle bounding boxes, distance bands, driver alerts,
and a post-run session metrics report.

---

## Why This Matters

Modern ADAS systems (Tesla Autopilot, Waymo, Mobileye) are built on layered perception
stacks: raw sensor input → lane understanding → object detection → risk assessment →
driver intervention. This project mirrors that architecture in a clean, modular Python
codebase — demonstrating that the engineering concerns (state management, evaluation,
environment robustness, ablation testing) matter as much as the ML component itself.

Key engineering decisions that distinguish this from tutorial code:
- Every module is independently testable with no GPU or video file required
- All parameters live in typed dataclasses — no magic numbers scattered in code
- The evaluation layer treats the pipeline as a black box, measuring it the same way
  a production team would: detection rates, false positive rates, alert lead times
- Environment modes make the system honest about what it can and can't do in the dark

---

## Tech Highlights

- **Modular perception stack** — lane, detection, tracking, proximity, alerts are fully
  decoupled and independently replaceable
- **Real-time processing** — YOLOv8n runs at 25-35 FPS on a modern laptop GPU;
  CPU-only is viable at reduced resolution
- **Environment modes** — adaptive Canny thresholds and wider ROI for night/rainy
  conditions, controlled by a single `--env-mode` flag
- **Metrics & evaluation** — per-frame logging → detection rate, FP rate, alert lead
  time, cut-in event detection, written to JSON after every run
- **Ablation experiments** — single script runs 5 configurations, prints a comparison
  table, exports CSV/JSON
- **Driver session summary** — human-readable narrative printed to console after processing

---

## Project Structure

```
adas_system/
├── main.py                      # CLI entry point + pipeline orchestrator
├── actions.md                   # Change log / rollback history
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   └── settings.py              # All tuneable params (VideoConfig, LaneConfig,
│                                #   DetectionConfig, TrackerConfig, ProximityConfig,
│                                #   AlertConfig, EnvironmentConfig, ADASConfig)
├── modules/
│   ├── lane_detection.py        # Classical CV pipeline (env-mode aware)
│   ├── object_detection.py      # YOLOv8 wrapper
│   ├── tracker.py               # IoU-based multi-object tracker
│   ├── proximity.py             # Distance banding + lateral zone classifier
│   └── alert_engine.py          # Rule-based alert engine with cooldowns
├── utils/
│   ├── geometry.py              # IoU, slope, ROI, polynomial helpers
│   ├── video_io.py              # VideoReader / VideoWriter
│   └── drawing.py               # All HUD / overlay rendering
├── metrics/
│   ├── eval_events.py           # FrameLog, VideoMetrics, evaluate_video_log()
│   └── driver_profile.py        # DriverSummary, build_driver_summary()
├── experiments/
│   └── ablation.py              # Multi-config comparative experiment runner
├── results/                     # Auto-created; metrics JSON written here
└── test_modules.py              # 30+ smoke tests; no GPU or video file needed
```

---

## Setup

```bash
pip install -r requirements.txt
# YOLOv8 weights (yolov8n.pt) are auto-downloaded on first run
```

---

## Usage

### Basic run
```bash
python main.py --input dashcam.mp4 --output output/result.mp4
```

### With environment mode
```bash
python main.py --input night_drive.mp4 --env-mode night --output output/night_result.mp4
python main.py --input rain_footage.mp4 --env-mode rainy --no-display
```

### Generate metrics JSON
```bash
# Metrics are auto-saved to results/<video_stem>_metrics.json
python main.py --input dashcam.mp4 --output-metrics results/session1_metrics.json
```

### Run ablation experiments
```bash
# Compare 5 configurations on a validation clip
python experiments/ablation.py --input validation.mp4 --max-frames 500

# Save results
python experiments/ablation.py --input validation.mp4 \
    --output-csv results/ablation.csv \
    --output-json results/ablation.json

# Run specific configs only
python experiments/ablation.py --input video.mp4 --configs baseline near_35pct night_mode
```

### Run tests
```bash
python test_modules.py
```

---

## Metrics Output Format

After each run, `results/<video>_metrics.json` contains:

```json
{
  "video_metrics": {
    "total_frames": 1800,
    "danger_detection_rate": 0.82,
    "danger_fp_rate": 0.004,
    "danger_avg_lead_frames": 3.2,
    "cut_in_event_count": 2,
    "lane_detected_pct": 94.1
  },
  "driver_summary": {
    "risk_level": "MODERATE",
    "danger_ahead_pct": 6.3,
    "lane_detected_pct": 94.1,
    "narrative": "6.3% of frames had a dangerously close vehicle ..."
  }
}
```

---

## Dataset Sources

| Dataset | URL | Notes |
|---------|-----|-------|
| BDD100K | https://bdd-data.berkeley.edu/ | 100K diverse driving videos |
| KITTI | https://www.cvlibs.net/datasets/kitti/ | Benchmark with GT annotations |
| Comma2k19 | https://github.com/commaai/comma2k19 | Highway driving, clean lanes |
| Any dashcam MP4 | — | Works out of the box |
