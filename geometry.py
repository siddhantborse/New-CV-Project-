"""
utils/geometry.py
=================
Shared geometric helpers used across multiple modules.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Sequence


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point   = Tuple[int, int]           # (x, y) pixel coordinate
BBox    = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
LineSeg = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------
def slope_intercept(x1: float, y1: float,
                    x2: float, y2: float) -> Tuple[float, float]:
    """Return (slope, intercept) for a line through two points."""
    if abs(x2 - x1) < 1e-6:
        return float('inf'), float('inf')
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def x_at_y(slope: float, intercept: float, y: float) -> float:
    """Return x coordinate for a given y on the line y = mx + b."""
    if slope == 0:
        return float('inf')
    return (y - intercept) / slope


def line_segment_from_slope(slope: float, intercept: float,
                             y_bottom: int, y_top: int) -> LineSeg:
    """Project a (slope, intercept) line to a pixel segment between two y values."""
    x_bottom = int(x_at_y(slope, intercept, y_bottom))
    x_top    = int(x_at_y(slope, intercept, y_top))
    return x_bottom, y_bottom, x_top, y_top


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------
def box_area(box: BBox) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box: BBox) -> Point:
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def box_bottom_center(box: BBox) -> Point:
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, y2


def iou(box_a: BBox, box_b: BBox) -> float:
    """Intersection-over-Union for two axis-aligned bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    union_area = box_area(box_a) + box_area(box_b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# ---------------------------------------------------------------------------
# ROI mask
# ---------------------------------------------------------------------------
def trapezoidal_roi_vertices(frame_h: int, frame_w: int,
                              top_frac: float, bottom_frac: float,
                              apex_half_width_frac: float) -> np.ndarray:
    """
    Build a trapezoid ROI for lane detection.

    Returns shape (4, 2) array of (x, y) vertices ordered:
        bottom-left, top-left, top-right, bottom-right
    """
    cx = frame_w / 2
    y_top    = int(frame_h * top_frac)
    y_bottom = int(frame_h * bottom_frac)
    half_w   = int(frame_w * apex_half_width_frac)

    return np.array([
        [0,            y_bottom],
        [cx - half_w,  y_top],
        [cx + half_w,  y_top],
        [frame_w,      y_bottom],
    ], dtype=np.int32)


# ---------------------------------------------------------------------------
# Polynomial lane helpers
# ---------------------------------------------------------------------------
def fit_polynomial(xs: Sequence[float], ys: Sequence[float],
                   degree: int = 1) -> Optional[np.ndarray]:
    """Fit a polynomial to (x, y) points.  Returns coefficients or None."""
    if len(xs) < degree + 1:
        return None
    try:
        return np.polyfit(ys, xs, degree)   # fit x = f(y) – more stable for lanes
    except (np.linalg.LinAlgError, ValueError):
        return None


def eval_polynomial_at_y(coeffs: np.ndarray, y: float) -> float:
    """Evaluate x = poly(y) at a single y coordinate."""
    return float(np.polyval(coeffs, y))
