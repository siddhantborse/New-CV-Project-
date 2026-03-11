"""
utils/video_io.py
=================
Thin wrappers around cv2.VideoCapture / VideoWriter with helpful metadata.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple
from pathlib import Path

from config.settings import VideoConfig


@dataclass
class VideoMeta:
    width:  int
    height: int
    fps:    float
    total_frames: int
    source: str


class VideoReader:
    """Context manager around cv2.VideoCapture that resizes frames on read."""

    def __init__(self, source: str | int, cfg: VideoConfig = VideoConfig()) -> None:
        self._source = source
        self._cfg    = cfg
        self._cap    = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {source!r}")

        raw_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps  = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.meta = VideoMeta(
            width  = cfg.target_width,
            height = cfg.target_height,
            fps    = cfg.fps_override or raw_fps,
            total_frames = n_frames,
            source = str(source),
        )

        print(f"[VideoReader] {source!r}  "
              f"{raw_w}x{raw_h}@{raw_fps:.1f}fps  "
              f"→ resized to {cfg.target_width}x{cfg.target_height}")

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        frame_idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            resized = cv2.resize(
                frame,
                (self._cfg.target_width, self._cfg.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            yield frame_idx, resized
            frame_idx += 1

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def release(self) -> None:
        self._cap.release()


class VideoWriter:
    """Context manager that writes processed frames to an output file."""

    def __init__(self, output_path: str, meta: VideoMeta,
                 cfg: VideoConfig = VideoConfig()) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)
        self._writer = cv2.VideoWriter(
            output_path, fourcc, meta.fps,
            (meta.width, meta.height),
        )
        self._path = output_path
        print(f"[VideoWriter] → {output_path}  "
              f"{meta.width}x{meta.height}@{meta.fps:.1f}fps")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()
        print(f"[VideoWriter] Saved: {self._path}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()
