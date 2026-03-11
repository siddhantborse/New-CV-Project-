"""
experiments/ablation.py
========================
Runs the ADAS pipeline against a validation video under several
configurations and reports comparative metrics.

Ablation configurations
-----------------------
1. baseline          – default settings
2. near_threshold_35 – proximity NEAR band at bbox-height ≥ 35% (more sensitive)
3. near_threshold_45 – proximity NEAR band at bbox-height ≥ 45% (less sensitive)
4. no_tracker        – tracker disabled (detections fed directly, no track IDs)
5. night_mode        – EnvironmentConfig.mode = NIGHT

Usage
-----
    python experiments/ablation.py --input path/to/validation.mp4
    python experiments/ablation.py --input video.mp4 --output-csv results/ablation.csv
    python experiments/ablation.py --input video.mp4 --max-frames 300
"""

from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import ADASConfig, EnvironmentConfig, EnvMode
from modules.lane_detection   import LaneDetector
from modules.object_detection import VehicleDetector
from modules.tracker          import IoUTracker
from modules.proximity        import ProximityAnalyser
from modules.alert_engine     import AlertEngine
from utils.video_io           import VideoReader
from metrics.eval_events      import (
    FrameLog, snapshot_rich_tracks,
    evaluate_video_log, VideoMetrics,
)


# ---------------------------------------------------------------------------
# Ablation configuration descriptor
# ---------------------------------------------------------------------------
@dataclass
class AblationConfig:
    name:        str
    description: str
    cfg:         ADASConfig
    use_tracker: bool = True   # set False to test tracker-disabled mode


def _make_configs() -> List[AblationConfig]:
    """Build the set of configurations to compare."""

    def base() -> ADASConfig:
        return ADASConfig()   # fresh copy each time

    # 1. Baseline
    c_base = base()

    # 2. NEAR threshold = 35%
    c_near35 = base()
    c_near35.proximity.band_thresholds = ((0.35, "NEAR"), (0.20, "MEDIUM"), (0.00, "FAR"))
    c_near35.proximity.danger_box_height_fraction = 0.33

    # 3. NEAR threshold = 45%
    c_near45 = base()
    c_near45.proximity.band_thresholds = ((0.45, "NEAR"), (0.20, "MEDIUM"), (0.00, "FAR"))
    c_near45.proximity.danger_box_height_fraction = 0.43

    # 4. No tracker (min_hits = 1 → immediate show; max_missed = 0 → no memory)
    c_no_tracker = base()
    c_no_tracker.tracker.min_hits_to_show = 1
    c_no_tracker.tracker.max_missed_frames = 0

    # 5. Night mode
    c_night = base()
    c_night.environment = EnvironmentConfig(mode=EnvMode.NIGHT)

    return [
        AblationConfig("baseline",          "Default settings",                       c_base),
        AblationConfig("near_35pct",        "NEAR threshold lowered to 35% bbox-h",   c_near35),
        AblationConfig("near_45pct",        "NEAR threshold raised to 45% bbox-h",    c_near45),
        AblationConfig("no_tracker",        "Tracker memory disabled (min_hits=1)",   c_no_tracker),
        AblationConfig("night_mode",        "EnvMode.NIGHT adaptive Canny",           c_night),
    ]


# ---------------------------------------------------------------------------
# Lightweight pipeline runner (no rendering, no display)
# ---------------------------------------------------------------------------
def run_ablation_pipeline(
    input_source: str,
    ablation:     AblationConfig,
    max_frames:   Optional[int] = None,
) -> tuple[VideoMetrics, float]:
    """
    Run the full pipeline for one ablation config.
    Returns (VideoMetrics, avg_fps).
    """
    cfg = ablation.cfg
    lane_detector    = LaneDetector(cfg.lane, cfg.environment)
    vehicle_detector = VehicleDetector(cfg.detection)
    tracker          = IoUTracker(cfg.tracker)
    proximity        = ProximityAnalyser(cfg.proximity)
    alert_engine     = AlertEngine(cfg.alert)

    frame_logs: List[FrameLog] = []
    frame_times: List[float]   = []

    with VideoReader(input_source, cfg.video) as reader:
        for frame_idx, frame in reader:
            if max_frames and frame_idx >= max_frames:
                break

            t0 = time.perf_counter()

            lane_result     = lane_detector.process(frame)
            raw_detections  = vehicle_detector.detect(frame)
            tracks          = tracker.update(raw_detections)
            rich_tracks     = proximity.analyse(tracks, reader.meta.width, reader.meta.height)
            active_alert    = alert_engine.evaluate(rich_tracks, lane_result)

            frame_times.append(time.perf_counter() - t0)

            frame_logs.append(FrameLog(
                frame_idx      = frame_idx,
                lane_detected  = lane_result.lane_detected,
                lane_offset_px = lane_result.lane_offset_px,
                rich_tracks    = snapshot_rich_tracks(rich_tracks),
                alert_category = active_alert.category if active_alert else None,
                alert_priority = active_alert.priority if active_alert else None,
            ))

    metrics = evaluate_video_log(frame_logs) if frame_logs else None
    avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

    return metrics, avg_fps


# ---------------------------------------------------------------------------
# Result container + reporting
# ---------------------------------------------------------------------------
@dataclass
class AblationResult:
    config_name:            str
    description:            str
    avg_fps:                float
    total_frames:           int
    danger_detection_rate:  float
    danger_fp_rate:         float
    danger_alert_count:     int
    cut_in_count:           int
    lane_detected_pct:      float
    avg_lead_frames:        Optional[float]


def _result_from(name: str, desc: str, m: VideoMetrics, fps: float) -> AblationResult:
    return AblationResult(
        config_name           = name,
        description           = desc,
        avg_fps               = round(fps, 2),
        total_frames          = m.total_frames,
        danger_detection_rate = m.danger_detection_rate,
        danger_fp_rate        = m.danger_fp_rate,
        danger_alert_count    = m.danger_alert_frames,
        cut_in_count          = m.cut_in_event_count,
        lane_detected_pct     = m.lane_detected_pct,
        avg_lead_frames       = m.danger_avg_lead_frames,
    )


def print_ablation_table(results: List[AblationResult]) -> None:
    col_w = [22, 8, 8, 10, 8, 8, 8, 8]
    headers = ["Config", "FPS", "Frames", "DangerDR%", "FPR%", "Alerts", "CutIns", "Lane%"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    def row(vals):
        cells = [str(v)[:w-1].ljust(w-1) for v, w in zip(vals, col_w)]
        return "|" + "|".join(f" {c}" for c in cells) + "|"

    print()
    print("  ABLATION RESULTS")
    print(sep)
    print(row(headers))
    print(sep)
    for r in results:
        print(row([
            r.config_name,
            f"{r.avg_fps:.1f}",
            r.total_frames,
            f"{r.danger_detection_rate*100:.1f}",
            f"{r.danger_fp_rate*100:.1f}",
            r.danger_alert_count,
            r.cut_in_count,
            f"{r.lane_detected_pct:.1f}",
        ]))
    print(sep)
    print()


def write_ablation_csv(results: List[AblationResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fi.name for fi in fields(AblationResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"[Ablation] CSV written → {path}")


def write_ablation_json(results: List[AblationResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"[Ablation] JSON written → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ADAS ablation experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",       required=True, help="Validation video path")
    p.add_argument("--output-csv",  default=None,  help="Save results CSV here")
    p.add_argument("--output-json", default=None,  help="Save results JSON here")
    p.add_argument("--max-frames",  type=int, default=None,
                   help="Limit frames per config (for quick runs)")
    p.add_argument("--configs",     nargs="+", default=None,
                   help="Run only these config names (default: all)")
    return p


def main() -> None:
    args   = build_parser().parse_args()
    configs = _make_configs()

    if args.configs:
        configs = [c for c in configs if c.name in args.configs]
        if not configs:
            print("[Ablation] No matching configs found.", file=sys.stderr)
            sys.exit(1)

    print(f"\n[Ablation] Running {len(configs)} configuration(s) on: {args.input}")
    print(f"[Ablation] Max frames per config: {args.max_frames or 'all'}\n")

    results: List[AblationResult] = []

    for abl in configs:
        print(f"  → {abl.name}: {abl.description}")
        try:
            metrics, avg_fps = run_ablation_pipeline(
                args.input, abl, max_frames=args.max_frames
            )
            if metrics:
                results.append(_result_from(abl.name, abl.description, metrics, avg_fps))
                print(f"     fps={avg_fps:.1f}  "
                      f"detect_rate={metrics.danger_detection_rate:.2f}  "
                      f"fp_rate={metrics.danger_fp_rate:.3f}")
        except Exception as exc:
            print(f"     [ERROR] {exc}", file=sys.stderr)

    if results:
        print_ablation_table(results)
        if args.output_csv:
            write_ablation_csv(results, args.output_csv)
        if args.output_json:
            write_ablation_json(results, args.output_json)


if __name__ == "__main__":
    main()
