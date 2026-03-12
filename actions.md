# ADAS Project — Change Log

All notable changes to this project are recorded here in reverse-chronological order.  
Use this file to track what changed, why, and how to roll back if needed.

Format per entry:
```
## [vX.Y.Z] — YYYY-MM-DD
### Added / Changed / Fixed / Removed / Rollback Notes
```

---

## [v1.0.0] — 2026-03-10

### Added — Initial project scaffold

**Scope:** Full initial implementation of the Real-Time ADAS Perception System.

**Files introduced:**
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point; orchestrates the 4-stage pipeline |
| `config/settings.py` | Centralised typed config via Python dataclasses |
| `modules/lane_detection.py` | Classical CV lane pipeline (Canny → Hough → EMA smoothing) |
| `modules/object_detection.py` | YOLOv8n wrapper filtered to road-vehicle COCO classes |
| `modules/tracker.py` | IoU-based greedy multi-object tracker with track lifecycle |
| `modules/proximity.py` | Bounding-box-height distance banding + lateral zone classification |
| `modules/alert_engine.py` | Stateful rule engine with per-category cooldown timers |
| `utils/geometry.py` | Shared math helpers (IoU, slope, ROI vertices, polynomial fit) |
| `utils/video_io.py` | VideoReader / VideoWriter context managers |
| `utils/drawing.py` | All HUD / overlay rendering (zero business logic) |
| `test_modules.py` | 20 smoke tests — run without GPU or video file |
| `requirements.txt` | Python dependency list |
| `README.md` | Setup, usage, dataset sources, module overview |
| `actions.md` | This file — project change log |

**Key design decisions:**
- All magic numbers live in `config/settings.py`; no hard-coded values in modules
- Lane EMA smoothing (`ema_alpha=0.25`) prevents frame-to-frame flickering
- Tracker requires `min_hits=2` before a track is displayed (suppresses one-frame ghosts)
- Alert engine rules are first-class callables — easy to add/remove rules without touching engine logic
- `drawing.py` is purely presentational — fully testable modules with no display dependency

**Dependencies:**
- `ultralytics>=8.0.0` (YOLOv8, auto-downloads `yolov8n.pt` weights on first run)
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`

**Test results:** All 20 smoke tests pass (`python test_modules.py`)

**Rollback:** N/A — initial version.

---

## [v1.1.0] — 2026-03-11

### Added — Evaluation, Ablation, Environment Modes, Driver Summary

**Scope:** Major feature expansion. No existing module interfaces changed; all additions
are additive — existing `main.py` CLI flags still work identically.

**New files introduced:**
| File | Purpose |
|------|---------|
| `metrics/__init__.py` | Package marker |
| `metrics/eval_events.py` | `FrameLog` dataclass, `evaluate_video_log()`, cut-in detector, lead-time calculator |
| `metrics/driver_profile.py` | `DriverSummary` dataclass, `build_driver_summary()`, `print_driver_summary()` |
| `experiments/__init__.py` | Package marker |
| `experiments/ablation.py` | 5-config ablation runner, CSV/JSON export, comparison table printer |
| `results/` | Auto-created directory for metrics JSON output |

**Modified files:**
| File | Change summary |
|------|---------------|
| `config/settings.py` | Added `EnvMode` enum, `EnvironmentConfig` dataclass; added `environment` field to `ADASConfig` |
| `modules/lane_detection.py` | `LaneDetector.__init__` now accepts `env_cfg: EnvironmentConfig`; added `_adaptive_canny_thresholds()` and environment-aware `_apply_roi()` |
| `main.py` | Added `--env-mode`, `--output-metrics` CLI flags; per-frame `FrameLog` collection; post-run `evaluate_video_log` + `build_driver_summary` + JSON write |
| `test_modules.py` | Added 4 new test functions: `test_environment_config`, `test_eval_events`, `test_cut_in_detection`, `test_driver_profile` |
| `README.md` | Full rewrite — added "Why This Matters", "Tech Highlights", metrics format docs, ablation usage |

**New CLI flags in main.py:**
```
--env-mode {day,night,rainy}    Adaptive lane detection environment mode
--output-metrics PATH           Save metrics + driver summary JSON (default: results/<video>_metrics.json)
```

**Why:**
- Evaluation metrics make the project recruiter-credible: it demonstrates understanding
  that perception systems need to be *measured*, not just run
- Environment modes show awareness of real-world deployment conditions
- Ablation script demonstrates scientific rigour — the ability to isolate and test
  individual design decisions

**Test results:** All 30+ smoke tests pass (`python test_modules.py`)

**Rollback steps:**
1. `git revert HEAD` — reverts all v1.1.0 changes in one step
2. Or manually: remove `metrics/`, `experiments/`, `results/`; revert
   `config/settings.py`, `modules/lane_detection.py`, `main.py`, `test_modules.py`
   to the v1.0.0 state via `git checkout v1.0.0 -- <file>`

---

## [v1.2.0] — 2026-03-11

### Added — First Working Prototype

**Scope:** First major milestone achieved. The real-time ADAS perception system is now a fully working prototype, integrating lane detection, object detection, multi-object tracking, proximity assessment, and an alert engine into a cohesive pipeline.

**Why:**
- Marks the completion of the core integration phase.
- Serves as a baseline working model for future enhancements and optimizations.

---
     TEMPLATE — copy this block for every new change
     ──────────────────────────────────────────────────────────────────── -->

<!--
## [v1.X.X] — YYYY-MM-DD

### Added / Changed / Fixed / Removed
- 

### Files modified
| File | Change summary |
|------|---------------|
|  |  |

### Why
- 

### Rollback steps
1. 
-->
