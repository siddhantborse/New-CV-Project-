"""
modules/object_detection.py
============================
YOLOv8 wrapper that filters to road-vehicle classes and returns
normalised Detection objects ready for the rest of the pipeline.

The ultralytics package auto-downloads model weights on first run.
Swap `model_name` in settings.py for a heavier model (yolov8s, yolov8m)
to trade speed for accuracy.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import DetectionConfig
from utils.geometry import BBox, box_area


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    box:        BBox            # (x1, y1, x2, y2) in pixel coords
    class_id:   int
    label:      str
    confidence: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class VehicleDetector:
    """
    Thin YOLOv8 wrapper.

    Usage:
        detector = VehicleDetector()
        detections = detector.detect(bgr_frame)
    """

    def __init__(self, cfg: DetectionConfig = DetectionConfig()) -> None:
        self.cfg = cfg
        self._model = self._load_model(cfg.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference and return filtered vehicle detections."""
        results = self._model(
            frame,
            conf    = self.cfg.confidence_threshold,
            iou     = self.cfg.iou_threshold,
            classes = list(self.cfg.target_classes),
            verbose = False,
        )

        detections: List[Detection] = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box_data in result.boxes:
                x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())

                # Clamp to frame bounds
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)

                cls_id = int(box_data.cls[0])
                conf   = float(box_data.conf[0])
                area   = box_area((x1, y1, x2, y2))

                if area < self.cfg.min_box_area:
                    continue

                label = self.cfg.class_labels.get(cls_id, f"class_{cls_id}")
                detections.append(Detection(
                    box        = (x1, y1, x2, y2),
                    class_id   = cls_id,
                    label      = label,
                    confidence = conf,
                ))

        return detections

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _load_model(model_name: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for vehicle detection.\n"
                "Install with:  pip install ultralytics"
            ) from exc

        print(f"[VehicleDetector] Loading {model_name} …")
        model = YOLO(model_name)
        print(f"[VehicleDetector] Model ready.")
        return model
