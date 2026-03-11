"""
modules/tracker.py
==================
Simple IoU-based multi-object tracker (no Kalman filter).

Design goals:
  - Zero external dependencies beyond NumPy
  - Stable track IDs across frames for alert de-duplication
  - Track lifecycle: tentative (new) → confirmed → lost → deleted

Algorithm (greedy IoU assignment):
  Each frame:
    1. Compute pairwise IoU between all active tracks and new detections.
    2. Greedy match: pick highest-IoU pair above threshold, repeat.
    3. Update matched tracks; age unmatched tracks.
    4. Promote new detections to tentative tracks.
    5. Delete tracks missing for > max_missed_frames.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from config.settings import TrackerConfig
from modules.object_detection import Detection
from utils.geometry import iou, BBox


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------
@dataclass
class Track:
    track_id:   int
    box:        BBox
    class_id:   int
    label:      str
    confidence: float
    age:        int   = 0     # total frames since creation
    hits:       int   = 0     # frames with a matched detection
    missed:     int   = 0     # consecutive frames without match

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 2   # becomes visible after first confirmation

    @property
    def box_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class IoUTracker:
    """
    Stateful IoU tracker.  Call `update(detections)` each frame.
    Returns only confirmed, currently-visible tracks.
    """

    def __init__(self, cfg: TrackerConfig = TrackerConfig()) -> None:
        self.cfg       = cfg
        self._tracks:  List[Track] = []
        self._next_id: int         = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Match detections to existing tracks, update state, return active tracks.
        """
        # --- Build cost matrix (we maximise IoU) -----------------------
        n_tracks = len(self._tracks)
        n_dets   = len(detections)

        matched_track_ids: set  = set()
        matched_det_ids:   set  = set()

        if n_tracks > 0 and n_dets > 0:
            iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
            for ti, track in enumerate(self._tracks):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = iou(track.box, det.box)

            # Greedy assignment
            while True:
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                best_iou = iou_matrix[idx]
                if best_iou < self.cfg.iou_threshold:
                    break
                ti, di = idx
                self._update_track(self._tracks[ti], detections[di])
                matched_track_ids.add(ti)
                matched_det_ids.add(di)
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1

        # --- Age unmatched tracks --------------------------------------
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_ids:
                track.missed += 1
                track.age    += 1

        # --- Spawn new tracks for unmatched detections -----------------
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                self._tracks.append(Track(
                    track_id   = self._next_id,
                    box        = det.box,
                    class_id   = det.class_id,
                    label      = det.label,
                    confidence = det.confidence,
                    hits       = 1,
                ))
                self._next_id += 1

        # --- Prune dead tracks -----------------------------------------
        self._tracks = [
            t for t in self._tracks
            if t.missed <= self.cfg.max_missed_frames
        ]

        # --- Return confirmed, visible tracks --------------------------
        return [
            t for t in self._tracks
            if t.hits >= self.cfg.min_hits_to_show and t.missed == 0
        ]

    def reset(self) -> None:
        self._tracks  = []
        self._next_id = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _update_track(track: Track, det: Detection) -> None:
        track.box        = det.box
        track.confidence = det.confidence
        track.hits      += 1
        track.missed     = 0
        track.age       += 1
