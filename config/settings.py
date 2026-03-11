"""
config/settings.py
==================
Single source of truth for all tuneable ADAS parameters.
Change values here rather than hunting through module code.
"""

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Video / Pipeline
# ---------------------------------------------------------------------------
@dataclass
class VideoConfig:
    target_width: int = 1280
    target_height: int = 720
    fps_override: float | None = None        # None → use source FPS
    fourcc: str = "mp4v"


# ---------------------------------------------------------------------------
# Lane Detection
# ---------------------------------------------------------------------------
@dataclass
class LaneConfig:
    # Pre-processing
    blur_kernel: Tuple[int, int] = (5, 5)
    canny_low: int = 50
    canny_high: int = 150

    # Region of interest (fraction of frame height from top)
    roi_top_fraction: float = 0.58          # ROI starts at 58% down
    roi_bottom_fraction: float = 0.95       # ROI ends at 95% down
    roi_apex_width_fraction: float = 0.08   # trapezoidal apex half-width

    # Hough transform
    hough_rho: int = 1
    hough_theta_deg: float = 1.0
    hough_threshold: int = 40
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 100

    # Line classification (slope thresholds)
    slope_min_abs: float = 0.4              # ignore near-horizontal lines
    slope_max_abs: float = 3.5              # ignore near-vertical noise

    # Exponential moving average for lane smoothing across frames
    ema_alpha: float = 0.25                 # 0 = no update, 1 = no smoothing

    # Visualisation
    lane_fill_alpha: float = 0.25           # filled corridor transparency
    lane_line_thickness: int = 4


# ---------------------------------------------------------------------------
# Object Detection (YOLOv8)
# ---------------------------------------------------------------------------
@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"          # nano – fast; swap for yolov8s.pt
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.45

    # COCO class IDs we care about: car=2, motorcycle=3, bus=5, truck=7
    target_classes: Tuple[int, ...] = (2, 3, 5, 7)

    # Human-readable labels (keyed by COCO class id)
    class_labels: dict = field(default_factory=lambda: {
        2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
    })

    # Minimum detection area (px²) – filters tiny false positives
    min_box_area: int = 1500


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
@dataclass
class TrackerConfig:
    iou_threshold: float = 0.35             # minimum IoU to match track→det
    max_missed_frames: int = 8              # frames before track is dropped
    min_hits_to_show: int = 2               # frames before track is displayed


# ---------------------------------------------------------------------------
# Proximity & Risk
# ---------------------------------------------------------------------------
@dataclass
class ProximityConfig:
    # Distance bands keyed by minimum bounding-box height fraction of frame
    # (box_h / frame_h).  Thresholds are empirical for typical dashcam FOV.
    band_thresholds: Tuple[Tuple[float, str], ...] = (
        (0.40, "NEAR"),
        (0.20, "MEDIUM"),
        (0.00, "FAR"),
    )

    # Lateral lane zones (fraction of frame width)
    left_lane_x_max: float = 0.38
    right_lane_x_min: float = 0.62
    ego_lane_x_range: Tuple[float, float] = (0.30, 0.70)

    # "Danger zone" – ahead + NEAR band
    danger_box_height_fraction: float = 0.38
    danger_x_range: Tuple[float, float] = (0.28, 0.72)


# ---------------------------------------------------------------------------
# Alert Engine
# ---------------------------------------------------------------------------
@dataclass
class AlertConfig:
    # Minimum seconds between re-issuing the same alert category
    cooldown_seconds: float = 3.0

    # How many frames an alert banner stays on screen
    banner_display_frames: int = 90         # ~3 s at 30 fps

    # Priority levels mapped to colours (BGR)
    priority_colors: dict = field(default_factory=lambda: {
        "CRITICAL": (0,   0,   220),
        "WARNING":  (0,  140,  255),
        "INFO":     (0,  200,  80),
    })


# ---------------------------------------------------------------------------
# Composite config (passed around the pipeline)
# ---------------------------------------------------------------------------
@dataclass
class ADASConfig:
    video:     VideoConfig     = field(default_factory=VideoConfig)
    lane:      LaneConfig      = field(default_factory=LaneConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker:   TrackerConfig   = field(default_factory=TrackerConfig)
    proximity: ProximityConfig = field(default_factory=ProximityConfig)
    alert:     AlertConfig     = field(default_factory=AlertConfig)


# Module-level singleton – import and use directly, or override in tests
DEFAULT_CONFIG = ADASConfig()
