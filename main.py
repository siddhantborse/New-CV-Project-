"""
main.py
=======
ADAS Pipeline Entry Point
─────────────────────────
Orchestrates the five-stage perception + evaluation pipeline:

    VideoReader
        ├─→ LaneDetector       (classical CV, env-mode aware)
        ├─→ VehicleDetector    (YOLOv8)
        │       └─→ IoUTracker → ProximityAnalyser → AlertEngine
        ├─→ FrameLog collector
        └─→ Drawing layer → VideoWriter + optional display

    Post-processing:
        FrameLogs → VideoMetrics → DriverSummary → JSON output

Usage:
    python main.py --input video.mp4
    python main.py --input video.mp4 --output output/result.mp4 --output-metrics results/metrics.json
    python main.py --input video.mp4 --env-mode night --no-display
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from config.settings import ADASConfig, EnvironmentConfig, EnvMode
from modules.lane_detection   import LaneDetector
from modules.object_detection import VehicleDetector
from modules.tracker          import IoUTracker
from modules.proximity        import ProximityAnalyser
from modules.alert_engine     import AlertEngine
from utils.video_io           import VideoReader, VideoWriter
from utils.drawing import (
    draw_lane_lines, draw_lane_corridor, draw_center_line,
    draw_detection, draw_hud, draw_alert_banner, draw_minimap,
)
from metrics.eval_events import (
    FrameLog, snapshot_rich_tracks,
    evaluate_video_log, metrics_to_dict,
)
from metrics.driver_profile import (
    build_driver_summary, print_driver_summary, summary_to_dict,
)


class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: list = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def _write_metrics_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Metrics] Written → {path}")


def run_pipeline(
    input_source:   str | int,
    output_path:    Optional[str],
    cfg:            ADASConfig,
    display:        bool = True,
    max_frames:     Optional[int] = None,
    output_metrics: Optional[str] = None,
) -> List[FrameLog]:

    lane_detector    = LaneDetector(cfg.lane, cfg.environment)
    vehicle_detector = VehicleDetector(cfg.detection)
    tracker          = IoUTracker(cfg.tracker)
    proximity        = ProximityAnalyser(cfg.proximity)
    alert_engine     = AlertEngine(cfg.alert)
    fps_counter      = FPSCounter()
    frame_logs: List[FrameLog] = []

    print(f"\n[Pipeline] Environment mode: {cfg.environment.mode.value.upper()}")

    with VideoReader(input_source, cfg.video) as reader:
        meta = reader.meta
        writer = VideoWriter(output_path, meta, cfg.video) if output_path else None

        print(f"[Pipeline] Starting  →  press Q to quit\n")
        last_frame_idx = 0

        for frame_idx, frame in reader:
            if max_frames and frame_idx >= max_frames:
                break
            last_frame_idx = frame_idx

            # ── Perception pipeline ────────────────────────────────────
            lane_result    = lane_detector.process(frame)
            raw_dets       = vehicle_detector.detect(frame)
            tracks         = tracker.update(raw_dets)
            rich_tracks    = proximity.analyse(tracks, meta.width, meta.height)
            active_alert   = alert_engine.evaluate(rich_tracks, lane_result)

            # ── Log ────────────────────────────────────────────────────
            frame_logs.append(FrameLog(
                frame_idx      = frame_idx,
                lane_detected  = lane_result.lane_detected,
                lane_offset_px = lane_result.lane_offset_px,
                rich_tracks    = snapshot_rich_tracks(rich_tracks),
                alert_category = active_alert.category if active_alert else None,
                alert_priority = active_alert.priority if active_alert else None,
            ))

            # ── Render ─────────────────────────────────────────────────
            canvas = frame.copy()
            if lane_result.lane_detected:
                draw_lane_corridor(canvas, lane_result.left_seg, lane_result.right_seg,
                                   alpha=cfg.lane.lane_fill_alpha)
                draw_lane_lines(canvas, lane_result.left_seg, lane_result.right_seg,
                                thickness=cfg.lane.lane_line_thickness)
                draw_center_line(canvas, lane_result.left_seg, lane_result.right_seg)

            for rt in rich_tracks:
                draw_detection(canvas, box=rt.track.box, label=rt.track.label,
                               band=rt.band, track_id=rt.track.track_id,
                               conf=rt.track.confidence)

            measured_fps = fps_counter.tick()
            draw_hud(canvas, frame_idx=frame_idx, fps=measured_fps,
                     lane_offset_px=lane_result.lane_offset_px,
                     curvature_m=lane_result.curvature_m, n_vehicles=len(rich_tracks))
            draw_minimap(canvas, [{"box": rt.track.box, "band": rt.band} for rt in rich_tracks],
                         meta.width, meta.height)
            if active_alert:
                draw_alert_banner(canvas, message=active_alert.message,
                                  priority=active_alert.priority,
                                  priority_colors=cfg.alert.priority_colors)

            if writer:
                writer.write(canvas)
            if display:
                cv2.imshow("ADAS Perception", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Pipeline] Quit requested.")
                    break

            if frame_idx % 100 == 0:
                total = meta.total_frames
                pct = f"{100*frame_idx/total:.1f}%" if total > 0 else "?"
                print(f"  frame {frame_idx:>6}  ({pct})  fps={measured_fps:.1f}  "
                      f"tracks={len(tracks)}  alert={'YES' if active_alert else 'no'}")

        print(f"\n[Pipeline] Done.  Processed {last_frame_idx + 1} frames.")

    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    # ── Post-processing ────────────────────────────────────────────────
    if frame_logs:
        metrics = evaluate_video_log(frame_logs)
        summary = build_driver_summary(frame_logs, metrics)
        print_driver_summary(summary)
        if output_metrics:
            _write_metrics_json(
                {"video_metrics": metrics_to_dict(metrics),
                 "driver_summary": summary_to_dict(summary)},
                output_metrics,
            )

    return frame_logs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-Time ADAS Perception – Lane, Vehicle, Proximity, Metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",          required=True)
    p.add_argument("--output",         default=None)
    p.add_argument("--output-metrics", default=None,
                   help="Save metrics JSON (default: results/<video>_metrics.json)")
    p.add_argument("--conf",           type=float, default=0.40)
    p.add_argument("--iou",            type=float, default=0.45)
    p.add_argument("--model",          default="yolov8n.pt")
    p.add_argument("--width",          type=int,   default=1280)
    p.add_argument("--height",         type=int,   default=720)
    p.add_argument("--max-frames",     type=int,   default=None)
    p.add_argument("--no-display",     action="store_true")
    p.add_argument("--env-mode",       choices=["day", "night", "rainy"], default="day",
                   help="Adaptive lane detection environment mode")
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        source = int(args.input)
    except ValueError:
        source = args.input
        if not Path(source).exists():
            print(f"[ERROR] Input file not found: {source!r}", file=sys.stderr)
            sys.exit(1)

    cfg = ADASConfig()
    cfg.video.target_width             = args.width
    cfg.video.target_height            = args.height
    cfg.detection.confidence_threshold = args.conf
    cfg.detection.iou_threshold        = args.iou
    cfg.detection.model_name           = args.model
    cfg.environment                    = EnvironmentConfig(mode=EnvMode(args.env_mode))

    metrics_path = args.output_metrics
    if metrics_path is None and isinstance(source, str):
        metrics_path = f"results/{Path(source).stem}_metrics.json"

    run_pipeline(
        input_source   = source,
        output_path    = args.output,
        cfg            = cfg,
        display        = not args.no_display,
        max_frames     = args.max_frames,
        output_metrics = metrics_path,
    )


if __name__ == "__main__":
    main()
