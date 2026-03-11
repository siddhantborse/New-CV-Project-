"""
main.py
=======
ADAS Pipeline Entry Point
─────────────────────────
Orchestrates the four-stage perception pipeline:

    VideoReader
        │
        ├─→ LaneDetector       (classical CV)
        ├─→ VehicleDetector    (YOLOv8)
        │       └─→ IoUTracker (stateful)
        │               └─→ ProximityAnalyser
        │                       └─→ AlertEngine
        │
        └─→ Drawing layer → VideoWriter + optional display

Usage:
    python main.py --input path/to/video.mp4
    python main.py --input path/to/video.mp4 --output output/result.mp4
    python main.py --input 0                              # webcam
    python main.py --input video.mp4 --conf 0.45 --no-display
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import ADASConfig
from modules.lane_detection  import LaneDetector
from modules.object_detection import VehicleDetector
from modules.tracker          import IoUTracker
from modules.proximity        import ProximityAnalyser
from modules.alert_engine     import AlertEngine
from utils.video_io           import VideoReader, VideoWriter
from utils.drawing import (
    draw_lane_lines,
    draw_lane_corridor,
    draw_center_line,
    draw_detection,
    draw_hud,
    draw_alert_banner,
    draw_minimap,
)


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    input_source: str | int,
    output_path:  Optional[str],
    cfg:          ADASConfig,
    display:      bool = True,
    max_frames:   Optional[int] = None,
) -> None:

    # ── Instantiate modules ────────────────────────────────────────────
    lane_detector   = LaneDetector(cfg.lane)
    vehicle_detector = VehicleDetector(cfg.detection)
    tracker          = IoUTracker(cfg.tracker)
    proximity        = ProximityAnalyser(cfg.proximity)
    alert_engine     = AlertEngine(cfg.alert)
    fps_counter      = FPSCounter()

    # ── Open video ─────────────────────────────────────────────────────
    with VideoReader(input_source, cfg.video) as reader:
        meta = reader.meta

        writer = None
        if output_path:
            writer = VideoWriter(output_path, meta, cfg.video)

        print(f"\n[Pipeline] Starting  →  press Q to quit\n")

        for frame_idx, frame in reader:
            if max_frames and frame_idx >= max_frames:
                break

            # ── 1. Lane detection ──────────────────────────────────────
            lane_result = lane_detector.process(frame)

            # ── 2. Vehicle detection + tracking ───────────────────────
            raw_detections = vehicle_detector.detect(frame)
            tracks         = tracker.update(raw_detections)

            # ── 3. Proximity & risk ────────────────────────────────────
            rich_tracks = proximity.analyse(tracks, meta.width, meta.height)

            # ── 4. Alerts ─────────────────────────────────────────────
            active_alert = alert_engine.evaluate(rich_tracks, lane_result)

            # ── 5. Render ─────────────────────────────────────────────
            canvas = frame.copy()

            # Lane overlays
            if lane_result.lane_detected:
                draw_lane_corridor(canvas,
                                    lane_result.left_seg,
                                    lane_result.right_seg,
                                    alpha=cfg.lane.lane_fill_alpha)
                draw_lane_lines(canvas,
                                lane_result.left_seg,
                                lane_result.right_seg,
                                thickness=cfg.lane.lane_line_thickness)
                draw_center_line(canvas,
                                  lane_result.left_seg,
                                  lane_result.right_seg)

            # Vehicle boxes
            for rt in rich_tracks:
                draw_detection(
                    canvas,
                    box      = rt.track.box,
                    label    = rt.track.label,
                    band     = rt.band,
                    track_id = rt.track.track_id,
                    conf     = rt.track.confidence,
                )

            # HUD
            measured_fps = fps_counter.tick()
            draw_hud(
                canvas,
                frame_idx       = frame_idx,
                fps             = measured_fps,
                lane_offset_px  = lane_result.lane_offset_px,
                curvature_m     = lane_result.curvature_m,
                n_vehicles      = len(rich_tracks),
            )

            # Mini-map
            det_dicts = [
                {"box": rt.track.box, "band": rt.band}
                for rt in rich_tracks
            ]
            draw_minimap(canvas, det_dicts, meta.width, meta.height)

            # Alert banner
            if active_alert:
                draw_alert_banner(
                    canvas,
                    message        = active_alert.message,
                    priority       = active_alert.priority,
                    priority_colors = cfg.alert.priority_colors,
                )

            # ── 6. Output ─────────────────────────────────────────────
            if writer:
                writer.write(canvas)

            if display:
                cv2.imshow("ADAS Perception", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[Pipeline] Quit requested.")
                    break

            # Progress log every 100 frames
            if frame_idx % 100 == 0:
                total = meta.total_frames
                pct   = f"{100*frame_idx/total:.1f}%" if total > 0 else "?"
                print(f"  frame {frame_idx:>6}  ({pct})  "
                      f"fps={measured_fps:.1f}  "
                      f"tracks={len(tracks)}  "
                      f"alert={'YES' if active_alert else 'no'}")

        print(f"\n[Pipeline] Done.  Processed {frame_idx + 1} frames.")

    if writer:
        writer.release()

    if display:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-Time ADAS Perception – Lane, Vehicle, Proximity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True,
                   help="Path to input video file or camera index (e.g. 0)")
    p.add_argument("--output",     default=None,
                   help="Path to save output video (optional)")
    p.add_argument("--conf",       type=float, default=0.40,
                   help="YOLO detection confidence threshold")
    p.add_argument("--iou",        type=float, default=0.45,
                   help="YOLO NMS IoU threshold")
    p.add_argument("--model",      default="yolov8n.pt",
                   help="YOLOv8 model weights filename")
    p.add_argument("--width",      type=int,   default=1280)
    p.add_argument("--height",     type=int,   default=720)
    p.add_argument("--max-frames", type=int,   default=None,
                   help="Stop after this many frames (useful for testing)")
    p.add_argument("--no-display", action="store_true",
                   help="Disable live preview window (headless mode)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Resolve source (int for webcam)
    try:
        source = int(args.input)
    except ValueError:
        source = args.input
        if not Path(source).exists():
            print(f"[ERROR] Input file not found: {source!r}", file=sys.stderr)
            sys.exit(1)

    # Build config, applying CLI overrides
    cfg = ADASConfig()
    cfg.video.target_width        = args.width
    cfg.video.target_height       = args.height
    cfg.detection.confidence_threshold = args.conf
    cfg.detection.iou_threshold   = args.iou
    cfg.detection.model_name      = args.model

    run_pipeline(
        input_source = source,
        output_path  = args.output,
        cfg          = cfg,
        display      = not args.no_display,
        max_frames   = args.max_frames,
    )


if __name__ == "__main__":
    main()
