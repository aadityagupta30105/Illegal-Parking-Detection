"""
ui/_hud_utils.py — Shared HUD visibility helpers.

All drawing helpers guarantee legibility via semi-transparent background panels.
Import:
    from ui._hud_utils import alpha_rect, draw_text_box, draw_panel
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def alpha_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    alpha: float = 0.72,
) -> None:
    """Blend a filled rectangle onto *img* in-place (no full-frame copy)."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def draw_text_box(
    img: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.65,
    thickness: int = 1,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (10, 10, 10),
    bg_alpha: float = 0.75,
    pad: int = 6,
) -> None:
    """Single text string with a semi-transparent background box."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x, y = origin
    alpha_rect(img, x - pad, y - th - pad, x + tw + pad, y + baseline + pad,
               bg_color, bg_alpha)
    cv2.putText(img, text, (x, y), _FONT, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_panel(
    img: np.ndarray,
    lines: List[Tuple[str, Tuple[int, int, int]]],
    x: int, y: int,
    font_scale: float = 0.65,
    thickness: int = 1,
    line_height: int = 28,
    bg_color: Tuple[int, int, int] = (10, 10, 10),
    bg_alpha: float = 0.80,
    border_color: Tuple[int, int, int] = (180, 180, 180),
    pad: int = 8,
) -> None:
    """
    Draw a multi-line panel with a single semi-transparent background rectangle.
    lines : list of (text, bgr_color) tuples.
    """
    if not lines:
        return

    max_w = max(
        cv2.getTextSize(t, _FONT, font_scale, thickness)[0][0]
        for t, _ in lines
    )
    panel_w = max_w + pad * 2
    panel_h = len(lines) * line_height + pad * 2

    alpha_rect(img, x, y, x + panel_w, y + panel_h, bg_color, bg_alpha)
    cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), border_color, 1)

    for i, (text, color) in enumerate(lines):
        ty = y + pad + (i + 1) * line_height - 4
        cv2.putText(img, text, (x + pad, ty), _FONT, font_scale, color, thickness, cv2.LINE_AA)