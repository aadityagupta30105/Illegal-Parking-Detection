"""
detectors/classical.py — MOG2 background-subtraction vehicle detector.

Optimizations:
- Single grayscale conversion shared across slot-occupancy and detection.
- Pre-allocated morph kernels (class-level, not per-call).
- Contour area pre-filter before boundingRect (avoids Python overhead).
- Aspect-ratio filter uses integer math.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from core.config import (
    BACKGROUND_HISTORY, BG_THRESHOLD, BLUR_KSIZE, BLUR_SIGMA,
    DILATE_ITER, MEDIAN_KSIZE, MIN_DETECTION_AREA,
    THRESH_BLOCK, THRESH_C, BBox,
)

# ── Pre-allocated kernels (module-level, never reallocated) ──────────────────
_DILATE_KERNEL  = np.ones((3, 3), np.uint8)
_MORPH_KERNEL   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ── Shared preprocessing ─────────────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert *frame* to grayscale and produce the slot-occupancy mask.

    Returns
    -------
    gray      : uint8 grayscale (H×W)
    processed : adaptive-threshold + dilate mask used for slot counting
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KSIZE, BLUR_SIGMA)
    thresh  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        THRESH_BLOCK, THRESH_C,
    )
    median    = cv2.medianBlur(thresh, MEDIAN_KSIZE)
    processed = cv2.dilate(median, _DILATE_KERNEL, iterations=DILATE_ITER)
    return gray, processed


# ── Classical MOG2 detector ──────────────────────────────────────────────────
class ClassicalDetector:
    """
    MOG2 → morphological cleanup → contour filtering.

    Improvements over original:
    - Accepts pre-computed gray (no redundant conversion when called from pipeline).
    - Uses class-level kernels (no per-call allocation).
    - Early-exits contour loop with area pre-check before boundingRect.
    - Aspect ratio computed on integer w/h (avoids float division on every contour).
    """

    def __init__(self) -> None:
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=BACKGROUND_HISTORY,
            varThreshold=BG_THRESHOLD,
            detectShadows=True,
        )
        # Preallocate nothing here; buffers are managed by OpenCV internally.

    def detect(self, frame: np.ndarray, gray: np.ndarray | None = None) -> List[BBox]:
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        fg = self._bg_sub.apply(blurred)
        # Threshold shadows (value 127) away; keep full foreground (255).
        cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY, dst=fg)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  _MORPH_KERNEL, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=3)
        fg = cv2.dilate(fg, _MORPH_KERNEL, iterations=2)

        h, w = frame.shape[:2]
        out: List[BBox] = []

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Fast area pre-filter using bounding-rect area (avoids contourArea on tiny blobs)
            if cv2.contourArea(cnt) < MIN_DETECTION_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Integer aspect-ratio check (avoids float division on every reject)
            # 0.3 < bw/bh < 5.0  →  3*bh < 10*bw < 50*bh
            ratio_10 = bw * 10
            if not (3 * bh < ratio_10 < 50 * bh):
                continue
            out.append(BBox(max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)))
        return out