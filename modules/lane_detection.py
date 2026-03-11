"""
modules/lane_detection.py
==========================
Classical computer-vision lane pipeline.

Pipeline:
    BGR frame
    → grayscale + Gaussian blur
    → Canny edge detection
    → trapezoidal ROI mask
    → probabilistic Hough lines
    → classify left / right by slope
    → robust line averaging per side
    → exponential moving average smoothing across frames
    → pixel segments + lane metadata (offset, curvature)
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from config.settings import LaneConfig, EnvironmentConfig, EnvMode
from utils.geometry import (
    trapezoidal_roi_vertices,
    slope_intercept,
    line_segment_from_slope,
    fit_polynomial,
    eval_polynomial_at_y,
    LineSeg,
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class LaneResult:
    left_seg:       Optional[LineSeg] = None   # (x1,y1,x2,y2) bottom→top
    right_seg:      Optional[LineSeg] = None
    lane_offset_px: Optional[float]  = None   # positive = car is right of centre
    curvature_m:    Optional[float]  = None   # very rough estimate
    lane_detected:  bool             = False
    left_raw_lines:  List = field(default_factory=list)
    right_raw_lines: List = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lane detector
# ---------------------------------------------------------------------------
class LaneDetector:
    """
    Stateful lane detector.  Call `process(frame)` each frame.
    State holds smoothed lane parameters across frames via EMA.

    Pass an EnvironmentConfig to enable adaptive Canny thresholds and
    wider ROI for night / rainy conditions.
    """

    def __init__(self, cfg: LaneConfig = LaneConfig(),
                 env_cfg: EnvironmentConfig = EnvironmentConfig()) -> None:
        self.cfg     = cfg
        self.env_cfg = env_cfg
        # Smoothed (slope, intercept) for left and right lanes
        self._left_smooth:  Optional[Tuple[float, float]] = None
        self._right_smooth: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> LaneResult:
        h, w = frame.shape[:2]
        edges = self._preprocess(frame)
        masked = self._apply_roi(edges, h, w)
        raw_lines = self._hough(masked)

        left_lines, right_lines = self._classify_lines(raw_lines, h, w)
        left_si  = self._average_lines(left_lines)
        right_si = self._average_lines(right_lines)

        # Smooth via EMA
        left_si  = self._ema(left_si,  "_left_smooth")
        right_si = self._ema(right_si, "_right_smooth")

        result = LaneResult(
            left_raw_lines  = left_lines,
            right_raw_lines = right_lines,
        )

        y_bot = int(h * self.cfg.roi_bottom_fraction)
        y_top = int(h * self.cfg.roi_top_fraction)

        if left_si:
            result.left_seg  = line_segment_from_slope(*left_si, y_bot, y_top)
        if right_si:
            result.right_seg = line_segment_from_slope(*right_si, y_bot, y_top)

        result.lane_detected = (result.left_seg is not None and
                                result.right_seg is not None)

        if result.lane_detected:
            result.lane_offset_px = self._compute_offset(
                result.left_seg, result.right_seg, w)
            result.curvature_m = self._estimate_curvature(
                result.left_seg, result.right_seg, h)

        return result

    def reset(self) -> None:
        self._left_smooth  = None
        self._right_smooth = None

    # ------------------------------------------------------------------
    # Pre-processing  (environment-aware)
    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.cfg.blur_kernel, 0)

        low, high = self._adaptive_canny_thresholds(blurred)
        edges = cv2.Canny(blurred, low, high)
        return edges

    def _adaptive_canny_thresholds(self, gray: np.ndarray) -> Tuple[int, int]:
        """Return Canny (low, high) adjusted for the current environment mode."""
        low  = self.cfg.canny_low
        high = self.cfg.canny_high
        mode = self.env_cfg.mode

        if mode == EnvMode.NIGHT:
            brightness = float(np.mean(gray))
            if brightness < self.env_cfg.night_brightness_target:
                # Scale thresholds down proportionally to how dark the frame is
                scale = max(0.35, brightness / self.env_cfg.night_brightness_target)
                low   = int(low  * scale)
                high  = int(high * scale)

        elif mode == EnvMode.RAINY:
            scale = self.env_cfg.rainy_canny_scale
            low   = int(low  * scale)
            high  = int(high * scale)

        return max(10, low), max(20, high)

    # ------------------------------------------------------------------
    # ROI  (environment-aware – wider apex for night/rainy)
    # ------------------------------------------------------------------
    def _apply_roi(self, edges: np.ndarray, h: int, w: int) -> np.ndarray:
        apex_w = self.cfg.roi_apex_width_fraction
        if self.env_cfg.mode in (EnvMode.NIGHT, EnvMode.RAINY):
            apex_w += self.env_cfg.roi_apex_extra

        vertices = trapezoidal_roi_vertices(
            h, w,
            self.cfg.roi_top_fraction,
            self.cfg.roi_bottom_fraction,
            apex_w,
        )
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [vertices], 255)
        return cv2.bitwise_and(edges, mask)

    # ------------------------------------------------------------------
    # Hough
    # ------------------------------------------------------------------
    def _hough(self, masked: np.ndarray) -> Optional[np.ndarray]:
        theta = np.deg2rad(self.cfg.hough_theta_deg)
        return cv2.HoughLinesP(
            masked,
            rho        = self.cfg.hough_rho,
            theta      = theta,
            threshold  = self.cfg.hough_threshold,
            minLineLength = self.cfg.hough_min_line_len,
            maxLineGap    = self.cfg.hough_max_line_gap,
        )

    # ------------------------------------------------------------------
    # Line classification
    # ------------------------------------------------------------------
    def _classify_lines(self,
                         raw_lines: Optional[np.ndarray],
                         h: int, w: int
                         ) -> Tuple[List, List]:
        left_lines:  List[Tuple[float, float]] = []
        right_lines: List[Tuple[float, float]] = []

        if raw_lines is None:
            return left_lines, right_lines

        cx = w / 2
        slope_min = self.cfg.slope_min_abs
        slope_max = self.cfg.slope_max_abs

        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            m, b = slope_intercept(x1, y1, x2, y2)

            if not (slope_min <= abs(m) <= slope_max):
                continue

            if m < 0 and x1 < cx and x2 < cx:    # left lane: negative slope, left half
                left_lines.append((m, b))
            elif m > 0 and x1 > cx and x2 > cx:  # right lane: positive slope, right half
                right_lines.append((m, b))

        return left_lines, right_lines

    # ------------------------------------------------------------------
    # Robust averaging
    # ------------------------------------------------------------------
    @staticmethod
    def _average_lines(lines: List[Tuple[float, float]]
                        ) -> Optional[Tuple[float, float]]:
        """Median slope & intercept – robust to outliers."""
        if not lines:
            return None
        slopes     = [l[0] for l in lines]
        intercepts = [l[1] for l in lines]
        return float(np.median(slopes)), float(np.median(intercepts))

    # ------------------------------------------------------------------
    # EMA smoothing
    # ------------------------------------------------------------------
    def _ema(self, new_si: Optional[Tuple[float, float]],
              attr: str) -> Optional[Tuple[float, float]]:
        prev = getattr(self, attr)
        a = self.cfg.ema_alpha

        if new_si is None:
            return prev  # keep previous estimate

        if prev is None:
            setattr(self, attr, new_si)
            return new_si

        smoothed = (
            a * new_si[0] + (1 - a) * prev[0],
            a * new_si[1] + (1 - a) * prev[1],
        )
        setattr(self, attr, smoothed)
        return smoothed

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_offset(left_seg: LineSeg, right_seg: LineSeg,
                         frame_w: int) -> float:
        """
        Lane centre offset in pixels.
        Positive → vehicle is to the right of lane centre.
        """
        left_x_bot  = left_seg[0]
        right_x_bot = right_seg[0]
        lane_cx     = (left_x_bot + right_x_bot) / 2.0
        return (frame_w / 2.0) - lane_cx      # positive = car is right of lane centre

    @staticmethod
    def _estimate_curvature(left_seg: LineSeg, right_seg: LineSeg,
                             frame_h: int) -> float:
        """
        Very rough curvature estimate based on the angle the lane makes.
        Real curvature needs a perspective-corrected bird's-eye projection;
        this approximation is intentionally lightweight for a portfolio demo.
        Returns radius in metres (higher = straighter road).
        """
        # Average the top x positions of left and right segments
        # A perfectly straight road gives the same x at top and bottom
        avg_dx_left  = left_seg[2]  - left_seg[0]   # top_x - bottom_x
        avg_dx_right = right_seg[2] - right_seg[0]
        avg_dx = (avg_dx_left + avg_dx_right) / 2.0

        # Heuristic: treat as arc approximation
        px_per_metre = 30.0
        dy_px = abs(left_seg[1] - left_seg[3])      # vertical span in px
        dy_m  = dy_px / px_per_metre
        dx_m  = abs(avg_dx) / px_per_metre

        if dx_m < 1e-3:
            return 9999.0    # essentially straight

        # Approximate circle radius: R ≈ (dy² + dx²) / (2·dx)
        r = (dy_m ** 2 + dx_m ** 2) / (2.0 * dx_m)
        return min(r, 9999.0)
