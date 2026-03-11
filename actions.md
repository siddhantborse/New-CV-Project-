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

<!-- ─────────────────────────────────────────────────────────────────────
     TEMPLATE — copy this block for every new change
     ──────────────────────────────────────────────────────────────────── -->

## [v1.1.0] — 2026-03-10

### Added / Changed / Fixed / Removed
- **Changed**: Refactored the flat project structure into a modular one (`config/`, `modules/`, `utils/`).
- **Changed**: Updated `settings.py` implementation to sit under `config/settings.py` as a centralized dataclass configuration.

### Files modified
| File | Change summary |
|------|---------------|
| `config/settings.py` | Added centralized `ADASConfig` parameters |
| `config/__init__.py` | Package init for config |
| `modules/__init__.py` | Package init for modules |
| `*.py` (root) | Moved all module (.py) files from the project root into `modules/`, `utils/`, or `config/` |

### Why
- Enhances code maintainability, separation of concerns, and clearer project architecture as the ADAS system grows.

### Rollback steps
1. `git checkout main`
2. `git revert <commit-hash>` -> Revert the structural modifications.

<!-- ─────────────────────────────────────────────────────────────────────
     TEMPLATE — copy this block for every new change
     ──────────────────────────────────────────────────────────────────── -->

<!--
## [v1.2.0] — YYYY-MM-DD

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
