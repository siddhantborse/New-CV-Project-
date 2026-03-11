"""
modules/proximity.py
=====================
Assigns distance bands (NEAR / MEDIUM / FAR) and lateral zone labels
(EGO_LANE / LEFT_LANE / RIGHT_LANE) to each tracked vehicle.

Distance estimation approach
─────────────────────────────
True monocular distance is underdetermined without calibration data.
This module uses a bounding-box-height heuristic which is well-established
for forward-facing cameras on a road surface:

    "As a vehicle gets closer, its bounding box grows taller."

We normalise box_height / frame_height to get a scale-invariant ratio,
then map it to three bands using empirically tuned thresholds (see config).

For a production system, replace this with:
  - Known vehicle heights + focal length → metric distance
  - Depth estimation model (MiDaS, DepthAnything)
  - Stereo/LiDAR fusion
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from config.settings import ProximityConfig
from modules.tracker import Track


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class RichTrack:
    """Track enriched with spatial metadata."""
    track:       Track
    band:        str    # "NEAR" | "MEDIUM" | "FAR"
    zone:        str    # "EGO_LANE" | "LEFT_LANE" | "RIGHT_LANE" | "UNKNOWN"
    danger:      bool   # True if requires immediate alert
    box_h_frac:  float  # normalised box height (diagnostic)
    cx_frac:     float  # normalised centre-x   (diagnostic)


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------
class ProximityAnalyser:

    def __init__(self, cfg: ProximityConfig = ProximityConfig()) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyse(self, tracks: List[Track],
                 frame_w: int, frame_h: int) -> List[RichTrack]:
        results: List[RichTrack] = []
        for track in tracks:
            x1, y1, x2, y2 = track.box
            box_h    = y2 - y1
            box_h_frac = box_h / frame_h
            cx_frac    = ((x1 + x2) / 2.0) / frame_w

            band   = self._distance_band(box_h_frac)
            zone   = self._lateral_zone(cx_frac)
            danger = self._is_danger(box_h_frac, cx_frac)

            results.append(RichTrack(
                track      = track,
                band       = band,
                zone       = zone,
                danger     = danger,
                box_h_frac = box_h_frac,
                cx_frac    = cx_frac,
            ))
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _distance_band(self, box_h_frac: float) -> str:
        for threshold, band_name in self.cfg.band_thresholds:
            if box_h_frac >= threshold:
                return band_name
        return "FAR"

    def _lateral_zone(self, cx_frac: float) -> str:
        if cx_frac <= self.cfg.left_lane_x_max:
            return "LEFT_LANE"
        if cx_frac >= self.cfg.right_lane_x_min:
            return "RIGHT_LANE"
        lo, hi = self.cfg.ego_lane_x_range
        if lo <= cx_frac <= hi:
            return "EGO_LANE"
        return "UNKNOWN"

    def _is_danger(self, box_h_frac: float, cx_frac: float) -> bool:
        in_danger_zone = (
            box_h_frac >= self.cfg.danger_box_height_fraction and
            self.cfg.danger_x_range[0] <= cx_frac <= self.cfg.danger_x_range[1]
        )
        return in_danger_zone
