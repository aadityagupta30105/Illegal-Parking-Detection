"""
core/video_utils.py — Video I/O helpers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: '{path}'")
    return cap


def read_first_frame(cap: cv2.VideoCapture, rewind: bool = True) -> np.ndarray:
    """Read the first frame; optionally rewind the capture afterwards."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Cannot read first frame from video.")
    if rewind:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame


def video_props(cap: cv2.VideoCapture) -> Tuple[int, int, float, int]:
    """Return (width, height, fps, total_frames)."""
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fp = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fp, n


def extract_first_frame_image(
    video_path: str,
    out_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """
    Extract the first frame from *video_path*, save it to *out_path*,
    and return it (optionally resized to width×height).
    """
    cap   = open_video(video_path)
    frame = read_first_frame(cap, rewind=False)
    cap.release()

    if width and height:
        frame = cv2.resize(frame, (width, height))

    cv2.imwrite(out_path, frame)
    log.info("Saved picker image → %s", out_path)
    return frame


def make_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(path, fourcc, fps, size)
    if not w.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter at '{path}'")
    return w