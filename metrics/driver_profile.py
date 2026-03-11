"""
metrics/driver_profile.py
==========================
Computes a human-readable session summary from the per-frame logs.

This is intentionally presentation-layer code: it takes the raw metrics
produced by eval_events.py and formats them into a DriverSummary that can
be printed to the console or embedded in the metrics JSON.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from metrics.eval_events import FrameLog, VideoMetrics, evaluate_video_log


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------
@dataclass
class DriverSummary:
    # Raw stats
    total_frames:               int
    danger_ahead_pct:           float   # % frames with NEAR vehicle in ego lane
    danger_alert_count:         int     # total danger_ahead alerts fired
    lane_detected_pct:          float   # % frames lane was found
    cut_in_count:               int
    avg_danger_lead_frames:     Optional[float]

    # Derived text
    risk_level:                 str     # "LOW" | "MODERATE" | "HIGH"
    narrative:                  str     # human-readable paragraph


def _risk_level(danger_pct: float) -> str:
    if danger_pct >= 10.0:
        return "HIGH"
    if danger_pct >= 3.0:
        return "MODERATE"
    return "LOW"


def build_driver_summary(logs: List[FrameLog],
                          metrics: Optional[VideoMetrics] = None) -> DriverSummary:
    """
    Build a DriverSummary from per-frame logs.
    Optionally accepts a pre-computed VideoMetrics to avoid re-computation.
    """
    if metrics is None:
        metrics = evaluate_video_log(logs)

    n = metrics.total_frames
    danger_pct = (metrics.danger_condition_frames / n * 100.0) if n > 0 else 0.0
    risk       = _risk_level(danger_pct)

    # ── Narrative ─────────────────────────────────────────────────────
    parts: List[str] = []

    # Danger ahead sentence
    if danger_pct < 1.0:
        parts.append("No significant close-following situations were detected.")
    else:
        parts.append(
            f"{danger_pct:.1f}% of frames had a dangerously close vehicle directly ahead."
        )

    # Lane detection sentence
    if metrics.lane_detected_pct >= 95.0:
        parts.append(f"Lane markings were clearly visible throughout the session ({metrics.lane_detected_pct:.0f}% detection rate).")
    elif metrics.lane_detected_pct >= 70.0:
        parts.append(f"Lane markings were detected in {metrics.lane_detected_pct:.0f}% of frames — some degraded visibility periods observed.")
    else:
        parts.append(f"Lane detection was unreliable ({metrics.lane_detected_pct:.0f}% detection rate) — likely poor road markings or lighting conditions.")

    # Cut-ins
    if metrics.cut_in_event_count == 0:
        parts.append("No cut-in events were observed.")
    elif metrics.cut_in_event_count == 1:
        parts.append("1 potential cut-in event was observed.")
    else:
        parts.append(f"{metrics.cut_in_event_count} potential cut-in events were observed.")

    # Alert lead time
    if metrics.danger_avg_lead_frames is not None:
        parts.append(
            f"Average alert lead time for danger-ahead conditions: "
            f"{metrics.danger_avg_lead_frames:.1f} frames."
        )

    # Risk summary
    parts.append(f"Overall session risk assessment: {risk}.")

    narrative = "  ".join(parts)

    return DriverSummary(
        total_frames             = n,
        danger_ahead_pct         = round(danger_pct, 2),
        danger_alert_count       = metrics.danger_alert_frames,
        lane_detected_pct        = metrics.lane_detected_pct,
        cut_in_count             = metrics.cut_in_event_count,
        avg_danger_lead_frames   = metrics.danger_avg_lead_frames,
        risk_level               = risk,
        narrative                = narrative,
    )


def print_driver_summary(summary: DriverSummary) -> None:
    width = 70
    print()
    print("╔" + "═" * width + "╗")
    print("║  SESSION SUMMARY" + " " * (width - 17) + "║")
    print("╠" + "═" * width + "╣")
    rows = [
        ("Total frames processed", str(summary.total_frames)),
        ("Danger-ahead (% frames)", f"{summary.danger_ahead_pct:.1f}%"),
        ("Danger-ahead alerts fired", str(summary.danger_alert_count)),
        ("Lane detected (% frames)", f"{summary.lane_detected_pct:.1f}%"),
        ("Cut-in events", str(summary.cut_in_count)),
        ("Avg alert lead time",
         f"{summary.avg_danger_lead_frames:.1f} frames"
         if summary.avg_danger_lead_frames else "N/A"),
        ("Risk level", summary.risk_level),
    ]
    for label, value in rows:
        pad = width - len(label) - len(value) - 4
        print(f"║  {label}{'.' * pad}{value}  ║")
    print("╠" + "═" * width + "╣")
    # Narrative wrapped at ~66 chars
    words  = summary.narrative.split()
    line   = ""
    for word in words:
        if len(line) + len(word) + 1 > 66:
            print(f"║  {line:<{width - 2}}║")
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        print(f"║  {line:<{width - 2}}║")
    print("╚" + "═" * width + "╝")
    print()


def summary_to_dict(s: DriverSummary) -> Dict[str, Any]:
    return asdict(s)
