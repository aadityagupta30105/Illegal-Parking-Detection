"""
detectors/yolo.py — Optional YOLOv8 vehicle detector.

Install:  pip install ultralytics
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from core.config import MIN_DETECTION_AREA, VEHICLE_CLASSES, BBox

log = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLOv8-based detector with the same interface as ClassicalDetector.
    Raises ImportError at construction time if ultralytics is absent.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLODetector.\n"
                "Install with:  pip install ultralytics"
            ) from exc
        self._model = YOLO(model_path)
        self._conf  = conf

    def detect(self, frame: np.ndarray, gray: np.ndarray | None = None) -> List[BBox]:
        # `gray` accepted for API compatibility; unused by YOLO.
        results = self._model(frame, verbose=False, conf=self._conf)[0]
        out: List[BBox] = []
        for box in results.boxes:
            if int(box.cls[0]) not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            b = BBox(x1, y1, x2, y2)
            if b.area >= MIN_DETECTION_AREA:
                out.append(b)
        return out


def build_detector(use_yolo: bool, model: str = "yolov8n.pt", conf: float = 0.35):
    """
    Factory: returns YOLODetector when requested and available,
    otherwise falls back to ClassicalDetector with a warning.
    """
    if use_yolo:
        try:
            det = YOLODetector(model, conf)
            log.info("Using YOLOv8 detector.")
            return det
        except ImportError as exc:
            log.warning("YOLO unavailable (%s); falling back to classical detector.", exc)

    from detectors.classical import ClassicalDetector
    log.info("Using classical MOG2 detector.")
    return ClassicalDetector()