"""
utils/drawing.py
================
All rendering logic – lane overlays, bounding boxes, HUD panels, alert banners.
Kept separate from business logic so modules stay clean.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from utils.geometry import BBox, Point


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
CLR_LANE_LEFT    = (0,   255, 127)   # spring green
CLR_LANE_RIGHT   = (0,   200, 255)   # gold-ish
CLR_LANE_CENTER  = (255, 255,   0)   # cyan
CLR_LANE_FILL    = (0,   180, 255)   # amber fill
CLR_BOX_DEFAULT  = (50,  220, 255)
CLR_BOX_NEAR     = (0,    40, 230)   # red
CLR_BOX_MEDIUM   = (0,   165, 255)   # orange
CLR_BOX_FAR      = (0,   200,  80)   # green
CLR_HUD_BG       = (20,   20,  20)
CLR_WHITE        = (255, 255, 255)
CLR_YELLOW       = (0,   220, 220)
CLR_RED          = (0,    50, 220)

BAND_COLORS: Dict[str, Tuple[int, int, int]] = {
    "NEAR":   CLR_BOX_NEAR,
    "MEDIUM": CLR_BOX_MEDIUM,
    "FAR":    CLR_BOX_FAR,
}

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Lane overlay
# ---------------------------------------------------------------------------
def draw_lane_lines(canvas: np.ndarray,
                    left_seg:  Optional[Tuple[int,int,int,int]],
                    right_seg: Optional[Tuple[int,int,int,int]],
                    thickness: int = 4) -> None:
    if left_seg:
        cv2.line(canvas, left_seg[:2], left_seg[2:], CLR_LANE_LEFT, thickness, cv2.LINE_AA)
    if right_seg:
        cv2.line(canvas, right_seg[:2], right_seg[2:], CLR_LANE_RIGHT, thickness, cv2.LINE_AA)


def draw_lane_corridor(canvas: np.ndarray,
                        left_seg:  Tuple[int,int,int,int],
                        right_seg: Tuple[int,int,int,int],
                        alpha: float = 0.20) -> None:
    """Fill the drivable corridor between left/right lane segments."""
    overlay = canvas.copy()
    pts = np.array([
        left_seg[:2],    # bottom-left
        left_seg[2:],    # top-left
        right_seg[2:],   # top-right
        right_seg[:2],   # bottom-right
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], CLR_LANE_FILL)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def draw_center_line(canvas: np.ndarray,
                      left_seg:  Tuple[int,int,int,int],
                      right_seg: Tuple[int,int,int,int]) -> None:
    """Draw a dashed center reference line between the two lanes."""
    lx1, ly1, lx2, ly2 = left_seg
    rx1, ry1, rx2, ry2 = right_seg
    cx_bot = (lx1 + rx1) // 2
    cx_top = (lx2 + rx2) // 2
    cy_bot = (ly1 + ry1) // 2
    cy_top = (ly2 + ry2) // 2
    _draw_dashed_line(canvas, (cx_bot, cy_bot), (cx_top, cy_top),
                      CLR_LANE_CENTER, thickness=2)


def _draw_dashed_line(canvas: np.ndarray, pt1: Point, pt2: Point,
                       color: Tuple, thickness: int = 2,
                       dash_len: int = 18, gap_len: int = 12) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    pos = 0.0
    draw = True
    while pos < dist:
        seg_len = dash_len if draw else gap_len
        end_pos = min(pos + seg_len, dist)
        if draw:
            sx = int(x1 + dx * pos);   sy = int(y1 + dy * pos)
            ex = int(x1 + dx * end_pos); ey = int(y1 + dy * end_pos)
            cv2.line(canvas, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos += seg_len
        draw = not draw


# ---------------------------------------------------------------------------
# Detection / tracking boxes
# ---------------------------------------------------------------------------
def draw_detection(canvas: np.ndarray,
                   box: BBox,
                   label: str,
                   band: str,
                   track_id: Optional[int] = None,
                   conf: float = 0.0) -> None:
    x1, y1, x2, y2 = box
    color = BAND_COLORS.get(band, CLR_BOX_DEFAULT)
    bw    = max(2, int((y2 - y1) / 60))          # scale thickness with box size

    # Main rectangle
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, bw)

    # Corner accents – cleaner look than full rectangle
    _draw_corner_accents(canvas, box, color, bw + 1)

    # Label background
    id_str   = f"#{track_id}" if track_id is not None else ""
    top_text = f"{label}{id_str}  {band}"
    (tw, th), _ = cv2.getTextSize(top_text, FONT_SMALL, 0.50, 1)
    label_y = max(y1 - 6, th + 4)
    cv2.rectangle(canvas, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 2),
                  color, cv2.FILLED)
    cv2.putText(canvas, top_text, (x1 + 3, label_y - 1),
                FONT_SMALL, 0.50, (15, 15, 15), 1, cv2.LINE_AA)


def _draw_corner_accents(canvas: np.ndarray, box: BBox,
                          color: Tuple, thickness: int,
                          length: int = 14) -> None:
    x1, y1, x2, y2 = box
    corners = [
        ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
    ]
    for apex, h_end, v_end in corners:
        cv2.line(canvas, apex, h_end, color, thickness, cv2.LINE_AA)
        cv2.line(canvas, apex, v_end, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HUD (heads-up display) panel
# ---------------------------------------------------------------------------
def draw_hud(canvas: np.ndarray,
             frame_idx: int,
             fps: float,
             lane_offset_px: Optional[float],
             curvature_m: Optional[float],
             n_vehicles: int) -> None:
    """Render a semi-transparent info panel in the top-left corner."""
    h, w = canvas.shape[:2]
    panel_w, panel_h = 270, 115
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), CLR_HUD_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    lines = [
        f"Frame:    {frame_idx:>6}",
        f"FPS:      {fps:>5.1f}",
        f"Vehicles: {n_vehicles:>3}",
    ]
    if lane_offset_px is not None:
        direction = "L" if lane_offset_px < 0 else "R"
        lines.append(f"Offset:   {abs(lane_offset_px):>4.0f}px {direction}")
    else:
        lines.append("Offset:   N/A")

    if curvature_m is not None:
        lines.append(f"Curv:     {curvature_m:>6.0f} m")

    for i, txt in enumerate(lines):
        cv2.putText(canvas, txt, (16, 30 + i * 18),
                    FONT_SMALL, 0.48, CLR_WHITE, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Alert banner
# ---------------------------------------------------------------------------
def draw_alert_banner(canvas: np.ndarray,
                       message: str,
                       priority: str,
                       priority_colors: Dict[str, Tuple]) -> None:
    """
    Render a full-width alert banner near the bottom of the frame.
    Priority drives the banner colour.
    """
    h, w = canvas.shape[:2]
    color = priority_colors.get(priority, (0, 140, 255))

    banner_h = 46
    y_top    = h - banner_h - 12

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, y_top), (w, y_top + banner_h), color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    # Left accent bar
    cv2.rectangle(canvas, (0, y_top), (6, y_top + banner_h), CLR_WHITE, cv2.FILLED)

    # Icon prefix
    icon = {"CRITICAL": "⚠  ", "WARNING": "▲  ", "INFO": "ℹ  "}.get(priority, "• ")
    full_msg = f"{icon}{message}"

    (tw, th), _ = cv2.getTextSize(full_msg, FONT, 0.70, 2)
    tx = max(16, (w - tw) // 2)
    ty = y_top + (banner_h + th) // 2 - 2

    # Drop shadow
    cv2.putText(canvas, full_msg, (tx + 2, ty + 2),
                FONT, 0.70, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, full_msg, (tx, ty),
                FONT, 0.70, CLR_WHITE, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Mini-map / bird's eye view placeholder
# ---------------------------------------------------------------------------
def draw_minimap(canvas: np.ndarray,
                  detections: List[Dict],
                  frame_w: int, frame_h: int,
                  size: int = 130) -> None:
    """
    Render a tiny schematic bird's-eye view in the top-right corner showing
    detected vehicle positions relative to the ego vehicle.
    """
    margin   = 10
    x_off    = frame_w - size - margin
    y_off    = margin
    bg_color = (30, 30, 30)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x_off, y_off),
                  (x_off + size, y_off + size), bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)

    # Border
    cv2.rectangle(canvas, (x_off, y_off),
                  (x_off + size, y_off + size), (80, 80, 80), 1)

    # Ego car dot at bottom-centre
    ego_x = x_off + size // 2
    ego_y = y_off + size - 10
    cv2.circle(canvas, (ego_x, ego_y), 5, (0, 255, 200), cv2.FILLED)

    # Map each detection to a minimap dot
    for det in detections:
        box = det["box"]
        cx_norm = ((box[0] + box[2]) / 2) / frame_w   # 0..1 horizontal
        cy_norm = box[3] / frame_h                      # 0..1 vertical (bottom)

        mx = int(x_off + cx_norm * size)
        my = int(y_off + (1.0 - cy_norm) * (size - 20) + 5)  # invert y, leave ego gap

        dot_color = BAND_COLORS.get(det.get("band", "FAR"), CLR_BOX_DEFAULT)
        cv2.circle(canvas, (mx, my), 4, dot_color, cv2.FILLED)

    # Label
    cv2.putText(canvas, "MAP", (x_off + 4, y_off + 12),
                FONT_SMALL, 0.36, (160, 160, 160), 1, cv2.LINE_AA)
