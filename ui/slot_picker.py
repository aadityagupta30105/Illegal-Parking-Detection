"""
ui/slot_picker.py — Interactive rectangular slot annotator.

Automatically extracts the first video frame when the picker image is absent.

Controls
────────
Left-click   Place a slot rectangle
Right-click  Remove nearest slot
Q            Quit and save
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from core.config import PICKER_H, PICKER_W, SLOT_H, SLOT_W
from ui._hud_utils import draw_panel

log = logging.getLogger(__name__)

POS_FILE      = "CarParkPos"
PICKER_IMG    = "clipimage.png"
DEFAULT_VIDEO = "clip.mp4"


# ── Persistence ───────────────────────────────────────────────────────────────
def _load(path: str) -> List[Tuple[int, int]]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return []


def _save(path: str, pos: List[Tuple[int, int]]) -> None:
    with open(path, "wb") as f:
        pickle.dump(pos, f)


# ── Picker image ──────────────────────────────────────────────────────────────
def _ensure_picker_image(img_path: str, video_path: str, w: int, h: int) -> np.ndarray:
    """Return picker image; auto-extract from video if file is missing."""
    p = Path(img_path)
    if p.exists():
        img = cv2.imread(str(p))
        if img is not None:
            return cv2.resize(img, (w, h))
        log.warning("Could not read %s; re-extracting from video.", p)

    log.info("Picker image not found — extracting first frame from '%s'.", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video '{video_path}'. "
            "Provide a video file or place 'clipimage.png' manually."
        )
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read first frame from video.")
    frame_resized = cv2.resize(frame, (w, h))
    cv2.imwrite(str(p), frame_resized)
    log.info("Saved picker image → %s", p)
    return frame_resized


# ── Mouse callback ────────────────────────────────────────────────────────────
def _mouse_cb(event, x: int, y: int, flags, pos_list: List[Tuple[int, int]]) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        pos_list.append((x, y))
        _save(POS_FILE, pos_list)
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, (x1, y1) in enumerate(pos_list):
            if x1 < x < x1 + SLOT_W and y1 < y < y1 + SLOT_H:
                pos_list.pop(i)
                _save(POS_FILE, pos_list)
                break


# ── Render ─────────────────────────────────────────────────────────────────────
def _draw(base: np.ndarray, pos_list: List[Tuple[int, int]]) -> np.ndarray:
    """Overlay slot rectangles and a clearly readable control hint panel."""
    canvas = base.copy()

    # Draw placed slots
    for (x, y) in pos_list:
        cv2.rectangle(canvas, (x, y), (x + SLOT_W, y + SLOT_H), (255, 0, 255), 2)

    # Control hint panel — always visible via draw_panel
    hints: List[Tuple[str, Tuple[int, int, int]]] = [
        ("Left-click  : place slot",     (220, 220,  80)),
        ("Right-click : remove slot",    (255, 120,  40)),
        ("Q           : save & quit",    (80,  220,  80)),
        (f"Slots placed : {len(pos_list)}", (80, 255, 200)),
    ]
    draw_panel(
        canvas, hints, x=10, y=10,
        font_scale=0.65, thickness=1, line_height=28,
        bg_color=(10, 10, 10), bg_alpha=0.80,
        border_color=(200, 200, 200),
    )
    return canvas


# ── Public entry point ────────────────────────────────────────────────────────
def run_picker(
    video_path: str = DEFAULT_VIDEO,
    pos_file:   str = POS_FILE,
    img_path:   str = PICKER_IMG,
) -> None:
    base     = _ensure_picker_image(img_path, video_path, PICKER_W, PICKER_H)
    pos_list = _load(pos_file)

    cv2.namedWindow("ParkingSpacePicker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ParkingSpacePicker", PICKER_W, PICKER_H)
    cv2.setMouseCallback("ParkingSpacePicker", _mouse_cb, pos_list)

    while True:
        cv2.imshow("ParkingSpacePicker", _draw(base, pos_list))
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    _save(pos_file, pos_list)
    log.info("Saved %d slot(s) → %s", len(pos_list), pos_file)