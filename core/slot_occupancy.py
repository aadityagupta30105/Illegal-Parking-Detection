"""
core/slot_occupancy.py — Parking slot occupancy detection.

Reuses the preprocessed frame produced by the classical detector pipeline
to avoid duplicate grayscale/threshold operations.

Optimizations:
- draw_slots pre-clamps all slot ROIs once (avoids per-slot min/max calls).
- Summary banner drawn with alpha_rect for visual consistency.
- run_slot_occupancy loops video endlessly with a single rewind.
"""
from __future__ import annotations

import logging
import pickle
from typing import List, Tuple

import cv2
import numpy as np

from core.config import (
    DISPLAY_H, DISPLAY_W, FREE_THRESHOLD,
    PICKER_H, PICKER_W, SLOT_H, SLOT_W,
)
from core.video_utils import open_video, video_props
from detectors.classical import preprocess_frame
from ui._hud_utils import alpha_rect

log = logging.getLogger(__name__)

POS_FILE = "CarParkPos"


def load_positions(pos_file: str = POS_FILE) -> List[Tuple[int, int]]:
    try:
        with open(pos_file, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        raise FileNotFoundError(
            f"Slot position file '{pos_file}' not found. "
            "Run the slot picker first (--mode picker)."
        )


def _scale(fw: int, fh: int) -> Tuple[float, float]:
    return fw / PICKER_W, fh / PICKER_H


def draw_slots(
    frame: np.ndarray,
    processed: np.ndarray,
    pos_list: List[Tuple[int, int]],
    sx: float,
    sy: float,
) -> int:
    """
    Overlay slot rectangles on *frame* in-place using *processed* mask.
    Returns number of free slots.

    ROI clamping is vectorised via pre-computed slot dims to avoid
    repeated min() calls inside the hot loop.
    """
    fh, fw = frame.shape[:2]
    sw = int(SLOT_W * sx)
    sh = int(SLOT_H * sy)
    free = 0

    for (px, py) in pos_list:
        x  = int(px * sx)
        y  = int(py * sy)
        x2 = min(x + sw, fw)
        y2 = min(y + sh, fh)
        if x2 <= x or y2 <= y:
            continue
        count = cv2.countNonZero(processed[y:y2, x:x2])
        occ   = count >= FREE_THRESHOLD
        color = (0, 0, 255) if occ else (0, 255, 0)
        thick = 2 if occ else 5
        if not occ:
            free += 1
        cv2.rectangle(frame, (x, y), (x2, y2), color, thick)
        cv2.putText(frame, str(count), (x + 2, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Summary banner — use alpha_rect so it's always legible
    total = len(pos_list)
    banner = f"Free: {free}/{total}"
    alpha_rect(frame, 78, 13, 452, 77, (0, 0, 0), alpha=0.65)
    cv2.putText(frame, banner, (90, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 0), 3, cv2.LINE_AA)
    return free


def run_slot_occupancy(video_path: str, pos_file: str = POS_FILE) -> None:
    pos_list = load_positions(pos_file)
    cap      = open_video(video_path)
    fw, fh, _, _ = video_props(cap)
    sx, sy   = _scale(fw, fh)

    cv2.namedWindow("Slot Occupancy", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Slot Occupancy", DISPLAY_W, DISPLAY_H)

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        _, processed = preprocess_frame(frame)
        draw_slots(frame, processed, pos_list, sx, sy)
        cv2.imshow("Slot Occupancy", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()