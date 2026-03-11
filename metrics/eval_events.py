"""
metrics/eval_events.py
======================
Evaluation layer that operates on a list of per-frame log entries produced
by main.py's pipeline loop.

Concepts
--------
Each frame is logged as a FrameLog (lightweight snapshot of pipeline outputs).
Events are defined as conditions that are either:
  - TRUE EVENTS  – the dangerous/noteworthy condition is actually present
  - ALERT FIRED  – the alert engine issued the corresponding alert

From the set of true events vs fired alerts we compute:
  - Detection rate  =  TP / total true-event frames
  - False positive rate  =  FP / non-event frames
  - Average alert lead time (frames between condition onset and alert firing)

Cut-in events are identified by tracking lateral zone transitions across frames.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Per-frame log (populated by main.py each frame)
# ---------------------------------------------------------------------------
@dataclass
class FrameLog:
    frame_idx:      int
    lane_detected:  bool
    lane_offset_px: Optional[float]

    # Simplified rich-track snapshot (avoid storing full objects)
    rich_tracks: List[Dict]     # each: {track_id, band, zone, danger, label}

    # Alert category that fired this frame (or None)
    alert_category: Optional[str]
    alert_priority: Optional[str]


def snapshot_rich_tracks(rich_tracks) -> List[Dict]:
    """Convert RichTrack objects to lightweight dicts for logging."""
    return [
        {
            "track_id": rt.track.track_id,
            "label":    rt.track.label,
            "band":     rt.band,
            "zone":     rt.zone,
            "danger":   rt.danger,
        }
        for rt in rich_tracks
    ]


# ---------------------------------------------------------------------------
# Event detectors (operate on a single FrameLog)
# ---------------------------------------------------------------------------
def _has_danger_condition(log: FrameLog) -> bool:
    """True if a NEAR vehicle is in EGO_LANE this frame."""
    return any(
        t["danger"] and t["zone"] == "EGO_LANE"
        for t in log.rich_tracks
    )


def _danger_alert_fired(log: FrameLog) -> bool:
    return log.alert_category == "danger_ahead"


def _lane_lost_condition(log: FrameLog) -> bool:
    return not log.lane_detected


def _lane_lost_alert_fired(log: FrameLog) -> bool:
    return log.alert_category == "lane_lost"


# ---------------------------------------------------------------------------
# Cut-in detector
# ---------------------------------------------------------------------------
def detect_cut_in_events(logs: List[FrameLog],
                           window: int = 30) -> List[int]:
    """
    A cut-in event occurs when a tracked vehicle transitions from
    LEFT_LANE or RIGHT_LANE into EGO_LANE and reaches NEAR band
    within `window` frames of the zone transition.

    Returns a list of frame indices where cut-ins were detected.
    """
    # track_id → last known zone
    last_zone: Dict[int, str] = {}
    # track_id → frame index when it entered EGO_LANE
    entered_ego: Dict[int, int] = {}
    cut_in_frames: List[int] = []

    for log in logs:
        for t in log.rich_tracks:
            tid  = t["track_id"]
            zone = t["zone"]
            band = t["band"]
            prev = last_zone.get(tid)

            # Zone transition: side lane → ego lane
            if prev in ("LEFT_LANE", "RIGHT_LANE") and zone == "EGO_LANE":
                entered_ego[tid] = log.frame_idx

            # Reached NEAR band in ego lane within window frames
            if (zone == "EGO_LANE" and band == "NEAR"
                    and tid in entered_ego
                    and (log.frame_idx - entered_ego[tid]) <= window):
                cut_in_frames.append(log.frame_idx)
                del entered_ego[tid]   # count once per cut-in

            last_zone[tid] = zone

    return cut_in_frames


# ---------------------------------------------------------------------------
# Alert lead time
# ---------------------------------------------------------------------------
def _compute_lead_time(logs: List[FrameLog],
                        condition_fn,
                        alert_fn) -> Optional[float]:
    """
    For each run of frames where condition_fn is True, find the first frame
    where alert_fn fires.  Lead time = alert_frame - condition_start_frame.
    Negative lead time means alert fired before condition (alert cooldown artefact).
    Returns mean lead time across all events, or None if no events.
    """
    lead_times: List[int] = []
    in_event       = False
    condition_start = 0

    for log in logs:
        cond  = condition_fn(log)
        fired = alert_fn(log)

        if cond and not in_event:
            in_event = True
            condition_start = log.frame_idx

        if fired and in_event:
            lead_times.append(log.frame_idx - condition_start)
            in_event = False

        if not cond:
            in_event = False

    return float(sum(lead_times) / len(lead_times)) if lead_times else None


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
@dataclass
class VideoMetrics:
    total_frames:             int
    # Danger ahead
    danger_condition_frames:  int
    danger_alert_frames:      int
    danger_tp:                int
    danger_fp:                int
    danger_detection_rate:    float   # TP / condition_frames  (0–1)
    danger_fp_rate:           float   # FP / non-condition frames  (0–1)
    danger_avg_lead_frames:   Optional[float]
    # Lane lost
    lane_lost_frames:         int
    lane_lost_alert_frames:   int
    lane_lost_detection_rate: float
    # Cut-ins
    cut_in_event_count:       int
    cut_in_frames:            List[int]
    # Lane
    lane_detected_pct:        float   # % frames lane was found


def evaluate_video_log(logs: List[FrameLog]) -> VideoMetrics:
    """
    Given the full list of per-frame logs for one video, compute all metrics.
    """
    n = len(logs)
    if n == 0:
        raise ValueError("Cannot evaluate an empty log.")

    # ── Danger ahead ───────────────────────────────────────────────────
    danger_condition = [l for l in logs if _has_danger_condition(l)]
    danger_alert     = [l for l in logs if _danger_alert_fired(l)]
    non_danger       = [l for l in logs if not _has_danger_condition(l)]

    # TP: alert fired AND condition was true that frame
    danger_tp = sum(1 for l in logs if _has_danger_condition(l) and _danger_alert_fired(l))
    # FP: alert fired but condition was NOT true
    danger_fp = sum(1 for l in logs if not _has_danger_condition(l) and _danger_alert_fired(l))

    danger_detection_rate = danger_tp / len(danger_condition) if danger_condition else 0.0
    danger_fp_rate        = danger_fp / len(non_danger)        if non_danger       else 0.0
    danger_lead           = _compute_lead_time(logs, _has_danger_condition, _danger_alert_fired)

    # ── Lane lost ──────────────────────────────────────────────────────
    lane_lost_cond  = [l for l in logs if _lane_lost_condition(l)]
    lane_lost_alert = [l for l in logs if _lane_lost_alert_fired(l)]
    lane_lost_tp    = sum(1 for l in logs if _lane_lost_condition(l) and _lane_lost_alert_fired(l))
    lane_lost_rate  = lane_lost_tp / len(lane_lost_cond) if lane_lost_cond else 1.0

    # ── Cut-ins ────────────────────────────────────────────────────────
    cut_in_frames = detect_cut_in_events(logs)

    # ── Lane detection coverage ────────────────────────────────────────
    lane_detected_pct = sum(1 for l in logs if l.lane_detected) / n * 100.0

    return VideoMetrics(
        total_frames             = n,
        danger_condition_frames  = len(danger_condition),
        danger_alert_frames      = len(danger_alert),
        danger_tp                = danger_tp,
        danger_fp                = danger_fp,
        danger_detection_rate    = round(danger_detection_rate, 4),
        danger_fp_rate           = round(danger_fp_rate, 4),
        danger_avg_lead_frames   = round(danger_lead, 2) if danger_lead is not None else None,
        lane_lost_frames         = len(lane_lost_cond),
        lane_lost_alert_frames   = len(lane_lost_alert),
        lane_lost_detection_rate = round(lane_lost_rate, 4),
        cut_in_event_count       = len(cut_in_frames),
        cut_in_frames            = cut_in_frames,
        lane_detected_pct        = round(lane_detected_pct, 2),
    )


def metrics_to_dict(m: VideoMetrics) -> Dict[str, Any]:
    return asdict(m)


def write_metrics_json(m: VideoMetrics, path: str) -> None:
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_to_dict(m), f, indent=2)
    print(f"[Metrics] Written → {path}")
