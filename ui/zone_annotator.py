"""
ui/zone_annotator.py
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from core.config import CLR_BLACK, CLR_GREEN, CLR_WHITE, CLR_YELLOW, ZONE_PALETTE
from ui._hud_utils import draw_panel, draw_text_box

log = logging.getLogger(__name__)

_WIN = "ANNOTATE NO-PARKING ZONES  |  F or ESC when done"

_CONTROLS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("Left-Click   : add vertex",          (220, 220,  80)),
    ("Enter/Space  : finish zone",          (220, 220,  80)),
    ("Z            : undo",                 (255, 160,  40)),
    ("Y            : redo",                 (255, 160,  40)),
    ("D            : delete nearest zone",  ( 80, 120, 255)),
    ("R            : reset in-progress",    (255, 160,  40)),
    ("F / ESC      : done",                 ( 80, 220,  80)),
]


class ZoneAnnotator:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame       = frame.copy()
        self.zones:   List[List[Tuple[int, int]]] = []
        self._redo:   List[List[Tuple[int, int]]] = []
        self.current: List[Tuple[int, int]]       = []
        self._mx = self._my = 0
        self._hover_idx = -1


    @staticmethod
    def _zone_color(idx: int) -> Tuple[int, int, int]:
        return ZONE_PALETTE[idx % len(ZONE_PALETTE)]

    @staticmethod
    def _centroid(pts: List[Tuple[int, int]]) -> Tuple[int, int]:
        a = np.array(pts, dtype=np.float32)
        return int(a[:, 0].mean()), int(a[:, 1].mean())

    def _nearest_zone(self) -> int:
        best, best_d = -1, float("inf")
        for i, z in enumerate(self.zones):
            cx, cy = self._centroid(z)
            d = (cx - self._mx) ** 2 + (cy - self._my) ** 2
            if d < best_d:
                best_d, best = d, i
        return best

    def _finish(self) -> None:
        if len(self.current) >= 3:
            self.zones.append(list(self.current))
            self._redo.clear()
            self.current = []
        else:
            log.warning("Need ≥ 3 points to close a zone.")

    def _undo(self) -> None:
        if self.current:
            self.current.pop()
        elif self.zones:
            self._redo.append(self.zones.pop())

    def _redo_last(self) -> None:
        if self._redo:
            self.zones.append(self._redo.pop())

    def _delete_nearest(self) -> None:
        idx = self._nearest_zone()
        if idx >= 0:
            self._redo.append(self.zones.pop(idx))

    def _mouse_cb(self, event, x: int, y: int, flags, param) -> None:
        self._mx, self._my = x, y
        self._hover_idx = self._nearest_zone()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y))
            self._redo.clear()

    def _render(self) -> np.ndarray:
        d    = self._frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = d.shape[:2]
        
        for i, zone in enumerate(self.zones):
            clr   = ZONE_PALETTE[i % len(ZONE_PALETTE)]
            pts   = np.array(zone, dtype=np.int32)
            alpha = 0.55 if i == self._hover_idx else 0.35
            over  = d.copy()
            cv2.fillPoly(over, [pts], clr)
            cv2.addWeighted(over, alpha, d, 1 - alpha, 0, d)
            cv2.polylines(d, [pts], True, clr, 3 if i == self._hover_idx else 2)
            for pt in zone:
                cv2.circle(d, pt, 5, CLR_WHITE, -1)
                cv2.circle(d, pt, 5, clr, 2)
            cx, cy = self._centroid(zone)
            lbl = f"Zone {i}"
            (tw, th), _ = cv2.getTextSize(lbl, font, 0.85, 2)
            cv2.rectangle(d, (cx - tw // 2 - 6, cy - th - 5),
                             (cx + tw // 2 + 6, cy + 5), CLR_BLACK, -1)
            cv2.rectangle(d, (cx - tw // 2 - 6, cy - th - 5),
                             (cx + tw // 2 + 6, cy + 5), clr, 2)
            cv2.putText(d, lbl, (cx - tw // 2, cy), font, 0.85, clr, 2, cv2.LINE_AA)

        if self.current:
            clr_c = ZONE_PALETTE[len(self.zones) % len(ZONE_PALETTE)]
            arr   = np.array(self.current, dtype=np.int32)
            if len(self.current) >= 2:
                cv2.polylines(d, [arr], False, clr_c, 3)
            if len(self.current) >= 3:
                over2 = d.copy()
                cv2.fillPoly(over2, [arr], clr_c)
                cv2.addWeighted(over2, 0.18, d, 0.82, 0, d)
                cv2.polylines(d, [arr], True, clr_c, 2)
            for pt in self.current:
                cv2.circle(d, pt, 6, clr_c, -1)
                cv2.circle(d, pt, 7, CLR_WHITE, 2)
            cv2.line(d, self.current[-1], (self._mx, self._my), clr_c, 2, cv2.LINE_AA)

        draw_panel(
            d, _CONTROLS, x=10, y=10,
            font_scale=0.62, thickness=1, line_height=27,
            bg_color=(10, 10, 10), bg_alpha=0.82,
            border_color=(200, 200, 200),
        )

        status_lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (f"Zones saved : {len(self.zones)}",       ( 80, 255, 200)),
            (f"In-progress : {len(self.current)} pts",
             CLR_YELLOW if self.current else (140, 140, 140)),
            (f"Redo stack  : {len(self._redo)}",
             CLR_GREEN if self._redo else (140, 140, 140)),
        ]
        panel_w_est = 270
        draw_panel(
            d, status_lines, x=w - panel_w_est - 10, y=10,
            font_scale=0.65, thickness=1, line_height=28,
            bg_color=(10, 10, 10), bg_alpha=0.82,
            border_color=(200, 200, 200),
        )
        
        if not self.zones and not self.current:
            hint = "Click to place the first vertex of a no-parking zone"
            (hw, _), _ = cv2.getTextSize(hint, font, 0.75, 2)
            hx = w // 2 - hw // 2
            hy = h - 20
            draw_text_box(
                d, hint, (hx, hy),
                font_scale=0.75, thickness=2,
                text_color=CLR_YELLOW, bg_color=(0, 0, 0), bg_alpha=0.72,
            )

        return d


    def run(self) -> List[np.ndarray]:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WIN, 1280, 720)
        cv2.setMouseCallback(_WIN, self._mouse_cb)

        while True:
            cv2.imshow(_WIN, self._render())
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if   key in (13, 32):                      self._finish()
            elif key in (26, ord("z"), ord("Z")):      self._undo()
            elif key in (25, ord("y"), ord("Y")):      self._redo_last()
            elif key in (4,  ord("d"), ord("D")):      self._delete_nearest()
            elif key in (ord("r"), ord("R")):          self.current.clear()
            elif key in (ord("f"), ord("F"), 27):
                if len(self.current) >= 3:
                    self._finish()
                break

        cv2.destroyAllWindows()
        result = [np.array(z, dtype=np.int32) for z in self.zones]
        log.info("Annotation done: %d zone(s).", len(result))
        return result