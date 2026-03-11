"""
modules/alert_engine.py
========================
Stateful, rule-based alert generator.

Design principles:
  - Rules are explicit and ordered by priority
  - Per-category cooldown timers prevent alert flooding
  - A single "active alert" is surfaced to the HUD at any time
    (highest priority wins; ties use most recent)
  - Alert state is decoupled from rendering
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from config.settings import AlertConfig
from modules.proximity import RichTrack
from modules.lane_detection import LaneResult


# ---------------------------------------------------------------------------
# Alert data model
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    category: str           # unique key for cooldown tracking
    message:  str           # displayed text
    priority: str           # "CRITICAL" | "WARNING" | "INFO"
    frames_remaining: int   # decremented each frame until banner disappears


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
@dataclass
class AlertRule:
    category:    str
    priority:    str
    message_fn:  object   # callable(rich_tracks, lane_result) → str | None


def _rule_danger_ahead(rich_tracks: List[RichTrack], lane: LaneResult
                        ) -> Optional[str]:
    for rt in rich_tracks:
        if rt.danger and rt.zone == "EGO_LANE":
            return f"{rt.track.label.upper()} DANGEROUSLY CLOSE AHEAD"
    return None


def _rule_near_ego(rich_tracks: List[RichTrack], lane: LaneResult
                    ) -> Optional[str]:
    for rt in rich_tracks:
        if rt.band == "NEAR" and rt.zone == "EGO_LANE" and not rt.danger:
            return f"CLOSE {rt.track.label.upper()} AHEAD – REDUCE SPEED"
    return None


def _rule_vehicle_left(rich_tracks: List[RichTrack], lane: LaneResult
                        ) -> Optional[str]:
    for rt in rich_tracks:
        if rt.zone == "LEFT_LANE" and rt.band in ("NEAR", "MEDIUM"):
            return f"VEHICLE APPROACHING FROM LEFT"
    return None


def _rule_vehicle_right(rich_tracks: List[RichTrack], lane: LaneResult
                         ) -> Optional[str]:
    for rt in rich_tracks:
        if rt.zone == "RIGHT_LANE" and rt.band in ("NEAR", "MEDIUM"):
            return f"VEHICLE APPROACHING FROM RIGHT"
    return None


def _rule_lane_lost(rich_tracks: List[RichTrack], lane: LaneResult
                     ) -> Optional[str]:
    if not lane.lane_detected:
        return "LANE DETECTION LOST – VISION DEGRADED"
    return None


def _rule_lane_departure(rich_tracks: List[RichTrack], lane: LaneResult
                          ) -> Optional[str]:
    if lane.lane_offset_px is not None and abs(lane.lane_offset_px) > 120:
        direction = "RIGHT" if lane.lane_offset_px < 0 else "LEFT"
        return f"LANE DEPARTURE WARNING – DRIFTING {direction}"
    return None


RULES: List[AlertRule] = [
    AlertRule("danger_ahead",    "CRITICAL", _rule_danger_ahead),
    AlertRule("near_ego",        "WARNING",  _rule_near_ego),
    AlertRule("vehicle_left",    "WARNING",  _rule_vehicle_left),
    AlertRule("vehicle_right",   "WARNING",  _rule_vehicle_right),
    AlertRule("lane_departure",  "WARNING",  _rule_lane_departure),
    AlertRule("lane_lost",       "INFO",     _rule_lane_lost),
]

PRIORITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------
class AlertEngine:

    def __init__(self, cfg: AlertConfig = AlertConfig()) -> None:
        self.cfg = cfg
        # category → last trigger timestamp
        self._cooldowns: Dict[str, float] = {}
        # Currently active displayed alert (may outlive the trigger condition)
        self._active: Optional[Alert] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self,
                  rich_tracks: List[RichTrack],
                  lane_result: LaneResult) -> Optional[Alert]:
        """
        Run all rules, apply cooldowns, update the active alert.
        Returns the alert to render this frame (or None).
        """
        now = time.monotonic()
        candidates: List[Alert] = []

        for rule in RULES:
            msg = rule.message_fn(rich_tracks, lane_result)
            if msg is None:
                continue

            last = self._cooldowns.get(rule.category, 0.0)
            if (now - last) < self.cfg.cooldown_seconds:
                continue  # still on cooldown

            candidates.append(Alert(
                category         = rule.category,
                message          = msg,
                priority         = rule.priority,
                frames_remaining = self.cfg.banner_display_frames,
            ))

        # Pick highest-priority candidate
        if candidates:
            best = min(candidates, key=lambda a: PRIORITY_ORDER[a.priority])
            self._cooldowns[best.category] = now
            self._active = best
        elif self._active is not None:
            self._active.frames_remaining -= 1
            if self._active.frames_remaining <= 0:
                self._active = None

        return self._active

    def reset(self) -> None:
        self._cooldowns.clear()
        self._active = None
