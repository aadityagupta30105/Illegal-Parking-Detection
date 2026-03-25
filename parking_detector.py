"""
=============================================================
  REAL-TIME ILLEGAL PARKING DETECTION SYSTEM
  Course: BCSE403L - Digital Image Processing
  Author: Senior CV Engineer Implementation
=============================================================

Architecture:
  1. Video ingestion  →  Frame extraction
  2. Interactive ROI annotation (polygon) via OpenCV window
  3. Background Subtraction + Contour detection for motion
  4. SORT-lite tracker (IoU-based centroid tracking, no deep deps)
  5. Dwell-time tracking per vehicle ID
  6. Alert system when dwell > T minutes
  7. Parking-spot occupancy counter (outside ROI)

Dependencies: opencv-python, numpy, scipy, Pillow
"""

import cv2
import numpy as np
import time
import json
import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment

# ──────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ──────────────────────────────────────────────
ILLEGAL_DWELL_MINUTES: float = 10.0   # ← change this to adjust alert threshold
PARKING_SPOT_GRID_ROWS: int = 2       # rows of parking spots to synthesise outside ROI
PARKING_SPOT_GRID_COLS: int = 5       # cols of parking spots
IOU_THRESHOLD: float = 0.25           # min IoU to associate detection → track
MAX_MISSED_FRAMES: int = 30           # frames before a track is removed
MIN_DETECTION_AREA: int = 1500        # px² — ignore tiny blobs
BACKGROUND_HISTORY: int = 300         # frames for BG model
BG_THRESHOLD: float = 40.0            # sensitivity

# Colours (BGR)
CLR_GREEN    = (50, 205, 50)
CLR_RED      = (0, 0, 220)
CLR_YELLOW   = (0, 215, 255)
CLR_ORANGE   = (0, 140, 255)
CLR_WHITE    = (255, 255, 255)
CLR_BLACK    = (0, 0, 0)
CLR_CYAN     = (255, 220, 0)
CLR_ZONE     = (0, 80, 255)          # no-parking zone overlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("parking_alerts.log", mode="w"),
    ],
)
log = logging.getLogger("ParkingDetector")


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────
@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int

    @property
    def cx(self): return (self.x1 + self.x2) // 2
    @property
    def cy(self): return (self.y1 + self.y2) // 2
    @property
    def area(self): return max(0, self.x2-self.x1) * max(0, self.y2-self.y1)
    @property
    def as_xywh(self): return (self.x1, self.y1, self.x2-self.x1, self.y2-self.y1)

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1); iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2); iy2 = min(self.y2, other.y2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: BBox
    missed: int = 0
    first_seen_in_zone: Optional[float] = None   # wall-clock time
    alert_fired: bool = False
    color: Tuple[int,int,int] = CLR_GREEN

    def enter_zone(self):
        if self.first_seen_in_zone is None:
            self.first_seen_in_zone = time.time()

    def leave_zone(self):
        self.first_seen_in_zone = None
        self.alert_fired = False

    @property
    def dwell_seconds(self) -> float:
        return (time.time() - self.first_seen_in_zone) if self.first_seen_in_zone else 0.0


@dataclass
class ParkingSpot:
    spot_id: int
    bbox: BBox
    occupied: bool = False


# ──────────────────────────────────────────────
# SORT-LITE TRACKER  (IoU-based, no Kalman)
# ──────────────────────────────────────────────
class SortLiteTracker:
    def __init__(self, iou_thresh=IOU_THRESHOLD, max_missed=MAX_MISSED_FRAMES):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[BBox]) -> List[Track]:
        if not self.tracks:
            for d in detections:
                t = Track(self._next_id, d)
                t.color = self._random_color(self._next_id)
                self.tracks[self._next_id] = t
                self._next_id += 1
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]

        # Build IoU cost matrix
        if detections and track_bboxes:
            cost = np.zeros((len(track_bboxes), len(detections)), dtype=np.float32)
            for r, tb in enumerate(track_bboxes):
                for c, db in enumerate(detections):
                    cost[r, c] = 1.0 - tb.iou(db)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_tracks = set(); matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - self.iou_thresh):
                    tid = track_ids[r]
                    self.tracks[tid].bbox = detections[c]
                    self.tracks[tid].missed = 0
                    matched_tracks.add(tid); matched_dets.add(c)
        else:
            matched_tracks = set(); matched_dets = set()

        # Unmatched tracks → increment missed
        for i, tid in enumerate(track_ids):
            if tid not in matched_tracks:
                self.tracks[tid].missed += 1

        # New detections → new tracks
        for c, d in enumerate(detections):
            if c not in matched_dets:
                t = Track(self._next_id, d)
                t.color = self._random_color(self._next_id)
                self.tracks[self._next_id] = t
                self._next_id += 1

        # Prune lost tracks
        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if t.missed <= self.max_missed}

        return list(self.tracks.values())

    @staticmethod
    def _random_color(seed: int) -> Tuple[int,int,int]:
        rng = np.random.default_rng(seed * 137 + 42)
        return tuple(int(x) for x in rng.integers(80, 220, 3))


# ──────────────────────────────────────────────
# MULTI-ZONE POLYGON ANNOTATOR  (with undo/redo)
# ──────────────────────────────────────────────

# One distinct BGR colour per zone (cycles if >8 zones)
ZONE_PALETTE: List[Tuple[int,int,int]] = [
    (0,  80,  255),   # zone 0 – red-orange
    (0,  200, 100),   # zone 1 – green
    (255, 140,  0),   # zone 2 – cyan-blue
    (180,  0, 180),   # zone 3 – magenta
    (0,  220, 220),   # zone 4 – yellow
    (255,  80,  80),  # zone 5 – blue
    (30,  180, 255),  # zone 6 – amber
    (100, 255, 150),  # zone 7 – lime
]


class MultiZoneAnnotator:
    """
    Interactive multi-zone polygon annotator.

    Controls
    --------
    Left-click          Add vertex to the current polygon being drawn
    ENTER / SPACE       Finish current polygon → save it, start a new one
    Ctrl+Z  / Z         Undo last vertex (or the whole last finished zone)
    Ctrl+Y  / Y         Redo last undone action
    Ctrl+D  / D         Delete the zone whose centre is closest to the mouse
    R                   Reset the vertex currently being drawn (discard in-progress)
    F / DONE            Finalise all zones and exit
    ESC                 Exit (returns whatever zones are complete so far)
    """

    def __init__(self, frame: np.ndarray):
        self.frame        = frame.copy()
        self.window       = "ANNOTATE NO-PARKING ZONES  |  See console for controls"

        # Completed zones  →  List of np.ndarray, each shape (N,2)
        self.zones: List[List[Tuple[int,int]]]  = []

        # Redo stack  →  holds zones that were undone
        self._redo_stack: List[List[Tuple[int,int]]] = []

        # Points of the polygon currently being drawn
        self.current: List[Tuple[int,int]] = []

        # Mouse position (for nearest-zone delete highlight)
        self._mx: int = 0
        self._my: int = 0

        # Which zone is being hovered for delete (index or -1)
        self._hover_delete: int = -1

    # ── helpers ──────────────────────────────
    def _zone_color(self, idx: int) -> Tuple[int,int,int]:
        return ZONE_PALETTE[idx % len(ZONE_PALETTE)]

    def _poly_centroid(self, pts: List[Tuple[int,int]]) -> Tuple[int,int]:
        arr = np.array(pts, dtype=np.float32)
        return int(arr[:,0].mean()), int(arr[:,1].mean())

    def _nearest_zone_to_mouse(self) -> int:
        """Return index of zone whose centroid is nearest to current mouse pos."""
        best_idx, best_d = -1, float('inf')
        for i, z in enumerate(self.zones):
            cx, cy = self._poly_centroid(z)
            d = (cx - self._mx)**2 + (cy - self._my)**2
            if d < best_d:
                best_d, best_idx = d, i
        return best_idx

    # ── undo / redo ──────────────────────────
    def _undo(self):
        if self.current:
            # Undo last vertex of in-progress polygon
            removed = self.current.pop()
            # We don't push partial-vertex undos onto the redo stack
        elif self.zones:
            # Undo the last completed zone
            zone = self.zones.pop()
            self._redo_stack.append(zone)
            log.info(f"Undo: removed zone {len(self.zones)} → {len(self.zones)} zones remain.")

    def _redo(self):
        if self._redo_stack:
            zone = self._redo_stack.pop()
            self.zones.append(zone)
            log.info(f"Redo: restored zone → {len(self.zones)} zones total.")

    def _delete_nearest(self):
        idx = self._nearest_zone_to_mouse()
        if idx >= 0:
            removed = self.zones.pop(idx)
            self._redo_stack.append(removed)
            log.info(f"Deleted zone {idx} → {len(self.zones)} zones remain.")

    def _finish_current(self):
        """Close the current in-progress polygon and save it."""
        if len(self.current) >= 3:
            self.zones.append(list(self.current))
            self._redo_stack.clear()   # new action clears redo
            log.info(f"Zone {len(self.zones)-1} confirmed with {len(self.current)} vertices.")
            self.current = []
        else:
            log.warning("Need ≥ 3 points to close a zone — keep clicking.")

    # ── mouse callback ────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        self._mx, self._my = x, y
        # Update hover highlight
        self._hover_delete = self._nearest_zone_to_mouse()

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y))
            self._redo_stack.clear()   # new click clears redo

    # ── low-level drawing helpers ─────────────
    @staticmethod
    def _filled_rect(img, x1, y1, x2, y2, color, alpha=0.78):
        """Semi-transparent filled rectangle."""
        sub = img[y1:y2, x1:x2]
        bg  = np.full_like(sub, color, dtype=np.uint8)
        cv2.addWeighted(bg, alpha, sub, 1-alpha, 0, sub)
        img[y1:y2, x1:x2] = sub

    @staticmethod
    def _outline_rect(img, x1, y1, x2, y2, color, thickness=1):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def _key_badge(img, x, y, label, badge_clr, text_clr=CLR_BLACK):
        """Draw a small rounded pill badge for a key name, return right edge x."""
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad_x, pad_y = 6, 3
        bx1, by1 = x, y - th - pad_y
        bx2, by2 = x + tw + pad_x*2, y + pad_y
        # Fill + outline
        cv2.rectangle(img, (bx1, by1), (bx2, by2), badge_clr, -1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), CLR_WHITE,  1)
        cv2.putText(img, label, (bx1 + pad_x, y), font, font_scale, text_clr, thickness, cv2.LINE_AA)
        return bx2 + 5   # next x after badge

    # ── render one frame ──────────────────────
    def _render(self) -> np.ndarray:
        display = self.frame.copy()
        h, w = display.shape[:2]

        # ════════════════════════════════════════
        # 1. Draw all completed zones
        # ════════════════════════════════════════
        for i, zone in enumerate(self.zones):
            clr  = self._zone_color(i)
            pts  = np.array(zone, dtype=np.int32)
            over = display.copy()
            cv2.fillPoly(over, [pts], clr)
            alpha = 0.40 if i != self._hover_delete else 0.58
            cv2.addWeighted(over, alpha, display, 1 - alpha, 0, display)
            # Border: thicker on hover
            border_t = 3 if i == self._hover_delete else 2
            cv2.polylines(display, [pts], isClosed=True, color=clr, thickness=border_t)
            # Dot on each vertex
            for pt in zone:
                cv2.circle(display, tuple(pt), 4, CLR_WHITE, -1)
                cv2.circle(display, tuple(pt), 4, clr,       1)
            # Centroid label
            cx, cy = self._poly_centroid(zone)
            lbl  = f"Zone {i}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(lbl, font, 0.65, 2)
            cv2.rectangle(display, (cx-tw//2-6, cy-th-5), (cx+tw//2+6, cy+5), CLR_BLACK, -1)
            cv2.rectangle(display, (cx-tw//2-6, cy-th-5), (cx+tw//2+6, cy+5), clr, 1)
            cv2.putText(display, lbl, (cx-tw//2, cy), font, 0.65, clr, 2, cv2.LINE_AA)
            # Hover delete hint below label
            if i == self._hover_delete:
                hint = "[ D ] delete"
                (hw, _), _ = cv2.getTextSize(hint, font, 0.42, 1)
                cv2.putText(display, hint, (cx-hw//2, cy+18),
                            font, 0.42, (0, 80, 255), 1, cv2.LINE_AA)

        # ════════════════════════════════════════
        # 2. Draw in-progress polygon
        # ════════════════════════════════════════
        next_idx = len(self.zones)
        clr_cur  = self._zone_color(next_idx)
        if len(self.current) >= 2:
            pts_cur = np.array(self.current, dtype=np.int32)
            cv2.polylines(display, [pts_cur], isClosed=False, color=clr_cur, thickness=2)
        if len(self.current) >= 3:
            pts_cur = np.array(self.current, dtype=np.int32)
            over2   = display.copy()
            cv2.fillPoly(over2, [pts_cur], clr_cur)
            cv2.addWeighted(over2, 0.18, display, 0.82, 0, display)
            cv2.polylines(display, [pts_cur], isClosed=True, color=clr_cur, thickness=1)
        for pt in self.current:
            cv2.circle(display, pt, 5, clr_cur,  -1)
            cv2.circle(display, pt, 6, CLR_WHITE,  1)
        # Rubber-band line to cursor
        if self.current:
            cv2.line(display, self.current[-1], (self._mx, self._my),
                     clr_cur, 1, cv2.LINE_AA)

        # ════════════════════════════════════════
        # 3. Controls panel — top-left
        # ════════════════════════════════════════
        #
        # Layout:
        #   ┌─────────────────────────────────────┐
        #   │  NO-PARKING ZONE ANNOTATOR          │  ← title bar
        #   ├─────────────────────────────────────┤
        #   │  [LClick]  Add vertex               │  ← draw group
        #   │  [Enter]   Finish zone              │
        #   ├·····································┤
        #   │  [Z]       Undo vertex / zone       │  ← edit group
        #   │  [Y]       Redo                     │
        #   │  [D]       Delete nearest zone      │
        #   │  [R]       Reset in-progress        │
        #   ├·····································┤
        #   │  [F] / [Esc]  Done → run detection  │  ← exit (green)
        #   └─────────────────────────────────────┘

        font     = cv2.FONT_HERSHEY_SIMPLEX
        fs_title = 0.52
        fs_row   = 0.46
        lh       = 24          # line height px
        pad_x    = 10          # left text margin inside panel
        panel_x  = 10          # panel left edge on frame
        panel_y  = 10          # panel top edge on frame
        panel_w  = 330

        # ── measure total height ──
        n_rows       = 8       # title + 7 action rows
        n_dividers   = 2       # thin lines
        title_h      = 28
        total_h      = title_h + n_rows * lh + n_dividers * 6 + 14

        # ── dark semi-transparent background ──
        px1, py1 = panel_x,           panel_y
        px2, py2 = panel_x + panel_w, panel_y + total_h
        self._filled_rect(display, px1, py1, px2, py2,
                          color=(15, 15, 15), alpha=0.80)
        self._outline_rect(display, px1, py1, px2, py2,
                           color=(200, 200, 200), thickness=1)

        # ── title bar ──
        ty = py1 + title_h - 6
        cv2.rectangle(display, (px1, py1), (px2, py1 + title_h),
                      (40, 40, 40), -1)
        cv2.rectangle(display, (px1, py1), (px2, py1 + title_h),
                      (200, 200, 200), 1)
        cv2.putText(display, "NO-PARKING ZONE ANNOTATOR",
                    (px1 + pad_x, ty), font, fs_title,
                    CLR_YELLOW, 1, cv2.LINE_AA)

        # Helper to draw one action row
        def _row(cursor_y, badge_label, badge_color, description, desc_color=CLR_WHITE):
            rx = self._key_badge(display,
                                 px1 + pad_x, cursor_y,
                                 badge_label, badge_color)
            cv2.putText(display, description,
                        (rx, cursor_y), font, fs_row,
                        desc_color, 1, cv2.LINE_AA)
            return cursor_y + lh

        # Helper for a thin divider
        def _divider(cursor_y, gap=6):
            mid = cursor_y + gap // 2
            cv2.line(display,
                     (px1 + pad_x, mid), (px2 - pad_x, mid),
                     (120, 120, 120), 1)
            return cursor_y + gap

        # ── row content ──
        # Badges: draw group = cyan; edit group = orange; exit = green
        CYAN   = (200, 180,  20)   # BGR ≈ goldenrod (looks cyan on dark bg)
        ORANGE = (0,  140, 255)
        GREEN  = (40, 160,  40)
        RED_B  = (40,  40, 200)

        ry = py1 + title_h + 6    # start just below title

        ry = _row(ry, "Left-Click", CYAN,   "Add a vertex to current polygon")
        ry = _row(ry, "Enter/Space", CYAN,  "Finish zone  ->  start new one")

        ry = _divider(ry)

        ry = _row(ry, "Z / Ctrl+Z", ORANGE, "Undo last vertex or whole zone")
        ry = _row(ry, "Y / Ctrl+Y", ORANGE, "Redo last undone zone")
        ry = _row(ry, "D / Ctrl+D", RED_B,  "Delete zone nearest to cursor")
        ry = _row(ry, "R",          ORANGE, "Reset in-progress polygon")

        ry = _divider(ry)

        _row(ry, "F  /  Esc",  GREEN,  "Done  ->  start detection", (80, 220, 80))

        # ════════════════════════════════════════
        # 4. Live status card — top-right
        # ════════════════════════════════════════
        status_lines = [
            (f"Zones saved   : {len(self.zones)}",        CLR_CYAN),
            (f"Current pts   : {len(self.current)}",
             CLR_YELLOW if len(self.current) > 0 else (160,160,160)),
            (f"Redo available: {len(self._redo_stack)}",
             CLR_GREEN if self._redo_stack else (160,160,160)),
        ]
        sw   = 230
        sh   = len(status_lines) * lh + 16
        sx1  = w - sw - 10
        sy1  = 10
        sx2  = w - 10
        sy2  = sy1 + sh
        self._filled_rect(display, sx1, sy1, sx2, sy2,
                          color=(15, 15, 15), alpha=0.78)
        self._outline_rect(display, sx1, sy1, sx2, sy2,
                           color=(200, 200, 200), thickness=1)
        cv2.putText(display, "STATUS", (sx1 + 8, sy1 + 16),
                    font, 0.48, CLR_YELLOW, 1, cv2.LINE_AA)
        for k, (txt, clr) in enumerate(status_lines):
            cv2.putText(display, txt, (sx1 + 8, sy1 + 16 + (k+1)*lh),
                        font, 0.44, clr, 1, cv2.LINE_AA)

        # ════════════════════════════════════════
        # 5. Zone colour legend — bottom-left
        # ════════════════════════════════════════
        if self.zones:
            leg_x, leg_y = 10, h - 14 - len(self.zones)*20 - 10
            lw = 200
            lh2 = len(self.zones) * 20 + 14
            self._filled_rect(display, leg_x, leg_y, leg_x+lw, leg_y+lh2,
                              color=(15,15,15), alpha=0.78)
            self._outline_rect(display, leg_x, leg_y, leg_x+lw, leg_y+lh2,
                               color=(200,200,200), thickness=1)
            cv2.putText(display, "Zone legend", (leg_x+8, leg_y+12),
                        font, 0.42, (200,200,200), 1, cv2.LINE_AA)
            for zi, zone in enumerate(self.zones):
                zclr = self._zone_color(zi)
                zy = leg_y + 12 + (zi+1)*20
                cv2.rectangle(display, (leg_x+8, zy-10), (leg_x+22, zy+2), zclr, -1)
                cv2.putText(display, f"Zone {zi}  ({len(zone)} pts)",
                            (leg_x+28, zy), font, 0.42, zclr, 1, cv2.LINE_AA)

        # ════════════════════════════════════════
        # 6. Centre-bottom hint when no zones yet
        # ════════════════════════════════════════
        if len(self.zones) == 0 and len(self.current) == 0:
            hint = "Click on the frame to start drawing your first no-parking zone"
            (hw, _), _ = cv2.getTextSize(hint, font, 0.52, 1)
            hx = max(10, w//2 - hw//2)
            hy = h - 18
            cv2.putText(display, hint, (hx+1, hy+1), font, 0.52, CLR_BLACK, 2, cv2.LINE_AA)
            cv2.putText(display, hint, (hx,   hy),   font, 0.52, CLR_YELLOW,1, cv2.LINE_AA)

        return display

    # ── main loop ────────────────────────────
    def run(self) -> List[np.ndarray]:
        """
        Returns a list of np.ndarray polygons (each shape [N,2], dtype int32).
        Returns empty list only if user exits without annotating anything.
        """
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1200, 750)
        cv2.setMouseCallback(self.window, self._mouse_cb)

        log.info("=== ANNOTATION WINDOW OPEN ===")
        log.info("Left-click to place vertices | ENTER to finish zone | F/ESC to exit")

        ctrl_held = False

        while True:
            display = self._render()
            cv2.imshow(self.window, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 255:   # no key
                continue

            # Track Ctrl state (approximation via key codes)
            # OpenCV doesn't expose modifier state directly on all platforms,
            # so we support BOTH bare keys and their Ctrl equivalents.

            if key in (13, 32):                # ENTER or SPACE → finish zone
                self._finish_current()

            elif key in (26, ord('z'), ord('Z')):   # Ctrl+Z or Z
                self._undo()

            elif key in (25, ord('y'), ord('Y')):   # Ctrl+Y or Y
                self._redo()

            elif key in (4, ord('d'), ord('D')):    # Ctrl+D or D
                self._delete_nearest()

            elif key in (ord('r'), ord('R')):       # R → reset in-progress
                if self.current:
                    log.info(f"Reset in-progress polygon ({len(self.current)} pts discarded).")
                    self.current = []

            elif key in (ord('f'), ord('F'), 27):   # F or ESC → done
                # Auto-close any open polygon before exiting
                if len(self.current) >= 3:
                    log.info("Auto-closing in-progress polygon on exit.")
                    self._finish_current()
                if len(self.zones) == 0:
                    log.warning("No zones defined — using default centre rectangle.")
                break

        cv2.destroyAllWindows()

        result = [np.array(z, dtype=np.int32) for z in self.zones]
        log.info(f"Annotation complete: {len(result)} zone(s) defined.")
        return result


# ──────────────────────────────────────────────
# BACKGROUND SUBTRACTOR + VEHICLE DETECTOR
# ──────────────────────────────────────────────
class VehicleDetector:
    """
    Classical DIP pipeline:
      1. MOG2 background subtraction
      2. Morphological cleanup
      3. Contour extraction & filtering
    Returns list[BBox].
    """

    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=BACKGROUND_HISTORY,
            varThreshold=BG_THRESHOLD,
            detectShadows=True,
        )
        kernel_size = 5
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def detect(self, frame: np.ndarray) -> List[BBox]:
        # 1. Convert to grayscale + denoise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # 2. Background subtraction
        fg_mask = self.bg_sub.apply(gray)

        # 3. Remove shadows (value 127 in MOG2) → binary
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 4. Morphological ops to fill holes & remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self.morph_kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=3)
        fg_mask = cv2.dilate(fg_mask, self.morph_kernel, iterations=2)

        # 5. Contour extraction
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[BBox] = []
        h, w = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_DETECTION_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Aspect ratio filter — vehicles are wider or squarish
            aspect = bw / (bh + 1e-5)
            if aspect < 0.3 or aspect > 5.0:
                continue
            detections.append(BBox(
                max(0, x), max(0, y),
                min(w, x+bw), min(h, y+bh)
            ))
        return detections

    def get_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the last fg_mask for debug display."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        fg = self.bg_sub.apply(gray)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.morph_kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.morph_kernel, iterations=3)
        return fg


# ──────────────────────────────────────────────
# PARKING SPOT MANAGER
# ──────────────────────────────────────────────
class ParkingSpotManager:
    """
    Divides a rectangular region (outside ALL no-parking zones) into
    a grid of parking spots and checks occupancy by overlap with tracks.
    """

    def __init__(self, frame_w: int, frame_h: int,
                 no_park_polys: List[np.ndarray],
                 rows=PARKING_SPOT_GRID_ROWS, cols=PARKING_SPOT_GRID_COLS):
        self.spots: List[ParkingSpot] = []
        # Place the grid in the BOTTOM portion of the frame
        margin = 10
        grid_h = frame_h // 5
        grid_y1 = frame_h - grid_h - margin
        grid_y2 = frame_h - margin
        cell_w = (frame_w - 2*margin) // cols
        cell_h = (grid_y2 - grid_y1) // rows

        sid = 0
        for r in range(rows):
            for c in range(cols):
                x1 = margin + c * cell_w
                y1 = grid_y1 + r * cell_h
                x2 = x1 + cell_w - 4
                y2 = y1 + cell_h - 4
                spot = ParkingSpot(sid, BBox(x1, y1, x2, y2))
                # Exclude spot if its centroid falls inside ANY no-parking zone
                cx, cy = (x1+x2)//2, (y1+y2)//2
                in_any_zone = any(
                    cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0
                    for poly in no_park_polys
                )
                if in_any_zone:
                    continue
                self.spots.append(spot)
                sid += 1

    def update(self, tracks: List[Track]):
        for spot in self.spots:
            spot.occupied = False
            for t in tracks:
                if t.missed == 0:  # only active tracks
                    iou = spot.bbox.iou(t.bbox)
                    if iou > 0.15:
                        spot.occupied = True
                        break

    def draw(self, frame: np.ndarray):
        for spot in self.spots:
            b = spot.bbox
            color = CLR_RED if spot.occupied else CLR_GREEN
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            label = f"P{spot.spot_id}"
            cv2.putText(frame, label, (b.x1+4, b.y1+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    @property
    def available(self): return sum(1 for s in self.spots if not s.occupied)
    @property
    def occupied_count(self): return sum(1 for s in self.spots if s.occupied)
    @property
    def total(self): return len(self.spots)


# ──────────────────────────────────────────────
# ALERT SYSTEM
# ──────────────────────────────────────────────
class AlertManager:
    def __init__(self, dwell_threshold_minutes: float = ILLEGAL_DWELL_MINUTES):
        self.threshold_sec = dwell_threshold_minutes * 60.0
        self.alert_count = 0

    def check(self, track: Track, fps_ratio: float = 1.0):
        """
        fps_ratio: multiply dwell by this when running on pre-recorded video
                   at faster-than-real-time playback.
        """
        if track.first_seen_in_zone is None:
            return
        if not track.alert_fired and track.dwell_seconds >= self.threshold_sec:
            track.alert_fired = True
            track.color = CLR_RED
            self.alert_count += 1
            msg = (
                f"🚨 ILLEGAL PARKING ALERT  |  "
                f"Vehicle ID={track.track_id}  |  "
                f"Dwell={track.dwell_seconds/60:.1f} min  |  "
                f"Threshold={self.threshold_sec/60:.1f} min  |  "
                f"BBox=({track.bbox.x1},{track.bbox.y1},{track.bbox.x2},{track.bbox.y2})"
            )
            log.warning(msg)


# ──────────────────────────────────────────────
# ZONE CHECKER
# ──────────────────────────────────────────────
def is_in_any_zone(polys: List[np.ndarray], bbox: BBox) -> bool:
    """
    Returns True if the centroid of the bounding box lies inside
    ANY of the supplied polygons.
    """
    cx, cy = float(bbox.cx), float(bbox.cy)
    return any(
        cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
        for poly in polys
    )


# ──────────────────────────────────────────────
# HUD RENDERER
# ──────────────────────────────────────────────
def draw_hud(frame: np.ndarray,
             tracks: List[Track],
             spot_mgr: ParkingSpotManager,
             alert_mgr: AlertManager,
             zone_polys: List[np.ndarray],
             frame_no: int,
             fps: float):
    h, w = frame.shape[:2]

    # ── Zone overlays (one per zone, each in its own colour) ──
    for i, zone_poly in enumerate(zone_polys):
        clr = ZONE_PALETTE[i % len(ZONE_PALETTE)]
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_poly], clr)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [zone_poly], True, clr, 2)
        # Zone label at centroid
        cx = int(zone_poly[:, 0].mean())
        cy = int(zone_poly[:, 1].mean())
        label = f"NO PARK Z{i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (cx-tw//2-3, cy-th-3), (cx+tw//2+3, cy+4),
                      CLR_BLACK, -1)
        cv2.putText(frame, label, (cx-tw//2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2, cv2.LINE_AA)

    # ── Parking spots ──
    spot_mgr.draw(frame)

    # ── Tracks ──
    for t in tracks:
        if t.missed > 5:
            continue
        b = t.bbox
        clr = t.color
        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), clr, 2)

        label = f"ID:{t.track_id}"
        if t.first_seen_in_zone is not None:
            dwell_min = t.dwell_seconds / 60.0
            label += f"  {dwell_min:.1f}m"
            bar_max_w = b.x2 - b.x1
            ratio = min(1.0, t.dwell_seconds / alert_mgr.threshold_sec)
            bar_w = int(bar_max_w * ratio)
            bar_clr = CLR_GREEN if ratio < 0.5 else (CLR_ORANGE if ratio < 0.9 else CLR_RED)
            cv2.rectangle(frame, (b.x1, b.y2+2), (b.x1+bar_w, b.y2+6), bar_clr, -1)

        # Flash red border on alert
        if t.alert_fired:
            cv2.rectangle(frame, (b.x1-3, b.y1-3), (b.x2+3, b.y2+3), CLR_RED, 3)
            cv2.putText(frame, "ALERT!", (b.x1, b.y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_RED, 2, cv2.LINE_AA)

        cv2.putText(frame, label, (b.x1+2, b.y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)

    # ── Info panel ──
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 270, 145
    sub = frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w]
    black = np.zeros_like(sub); cv2.addWeighted(black, 0.55, sub, 0.45, 0, sub)
    frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = sub

    lines = [
        (f"Frame: {frame_no}",                CLR_WHITE),
        (f"FPS: {fps:.1f}",                   CLR_WHITE),
        (f"Active Vehicles: {sum(1 for t in tracks if t.missed==0)}", CLR_CYAN),
        (f"In Zone: {sum(1 for t in tracks if t.first_seen_in_zone is not None and t.missed==0)}", CLR_ORANGE),
        (f"Alerts Fired: {alert_mgr.alert_count}",  CLR_RED),
        (f"Parking: {spot_mgr.available}/{spot_mgr.total} free", CLR_GREEN),
    ]
    for i, (txt, clr) in enumerate(lines):
        cv2.putText(frame, txt, (panel_x+8, panel_y+18+i*21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, clr, 1, cv2.LINE_AA)

    # ── Threshold reminder ──
    cv2.putText(frame, f"Alert threshold: {ILLEGAL_DWELL_MINUTES:.1f} min",
                (w-280, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
class IllegalParkingDetector:
    def __init__(self, video_path: str,
                 dwell_minutes: float = ILLEGAL_DWELL_MINUTES,
                 output_path: Optional[str] = None,
                 show_window: bool = True,
                 zone_points: Optional[List] = None):
        self.video_path = video_path
        self.dwell_minutes = dwell_minutes
        self.output_path = output_path
        self.show_window = show_window

        # zone_points may be:
        #   None               → open annotation window
        #   [[x,y],...]        → single zone (old format, auto-wrapped)
        #   [[[x,y],...], ...] → multiple zones
        self.preset_zones: Optional[List[np.ndarray]] = None
        if zone_points is not None:
            # Detect if it's a flat list of [x,y] pairs (single zone)
            if zone_points and not isinstance(zone_points[0][0], (list, tuple)):
                zone_points = [zone_points]
            self.preset_zones = [
                np.array(z, dtype=np.int32) for z in zone_points
            ]

        self.detector  = VehicleDetector()
        self.tracker   = SortLiteTracker()
        self.alert_mgr = AlertManager(dwell_minutes)

    # ── STEP 1: Annotation ──────────────────
    def _annotate_zones(self, first_frame: np.ndarray) -> List[np.ndarray]:
        if self.preset_zones is not None:
            log.info(f"Using {len(self.preset_zones)} preset zone(s).")
            return self.preset_zones
        log.info("Opening multi-zone annotation window …")
        ann = MultiZoneAnnotator(first_frame)
        zones = ann.run()
        if not zones:
            # Fallback: default centre rectangle as one zone
            h, w = first_frame.shape[:2]
            zones = [np.array([
                [w//4, h//4], [3*w//4, h//4],
                [3*w//4, 3*h//4], [w//4, 3*h//4],
            ], dtype=np.int32)]
            log.warning("No zones annotated — using default centre rectangle.")
        return zones

    # ── STEP 2 & 3: Process video ───────────
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            log.error(f"Cannot open video: {self.video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Video: {frame_w}×{frame_h} @ {src_fps:.1f} fps, {total_frames} frames")

        # Read first frame
        ret, first_frame = cap.read()
        if not ret:
            log.error("Could not read first frame."); return

        # STEP 1 — Annotate zones
        zone_polys = self._annotate_zones(first_frame)
        log.info(f"{len(zone_polys)} no-parking zone(s) active.")

        # Spot manager — exclude spots inside any zone
        spot_mgr = ParkingSpotManager(frame_w, frame_h, zone_polys)
        log.info(f"Parking spots defined: {spot_mgr.total}")

        # Output writer
        writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output_path, fourcc, src_fps,
                                     (frame_w, frame_h))
            log.info(f"Writing output to: {self.output_path}")

        # Reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_no = 0
        fps_timer = time.time()
        display_fps = src_fps

        log.info("▶  Processing video … press 'q' or ESC to stop.")

        if self.show_window:
            cv2.namedWindow("Illegal Parking Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Illegal Parking Detector", 1100, 700)

        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("End of video.")
                break
            frame_no += 1

            # ── STEP 2: Detect vehicles ──
            detections = self.detector.detect(frame)

            # ── STEP 2: Track vehicles ──
            tracks = self.tracker.update(detections)

            # ── STEP 3: Zone check + dwell timing ──
            for t in tracks:
                if t.missed == 0:
                    if is_in_any_zone(zone_polys, t.bbox):
                        t.enter_zone()
                        self.alert_mgr.check(t)
                    else:
                        t.leave_zone()

            # Update parking spot occupancy
            spot_mgr.update(tracks)

            # ── Compute display FPS ──
            now = time.time()
            if frame_no % 15 == 0:
                display_fps = 15.0 / max(1e-5, now - fps_timer)
                fps_timer = now

            # ── Draw HUD ──
            draw_hud(frame, tracks, spot_mgr, self.alert_mgr,
                     zone_polys, frame_no, display_fps)

            if writer:
                writer.write(frame)

            if self.show_window:
                cv2.imshow("Illegal Parking Detector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    log.info("Stopped by user.")
                    break

        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

        # Summary
        log.info("=" * 60)
        log.info(f"SUMMARY  |  Frames processed: {frame_no}")
        log.info(f"         |  Zones monitored : {len(zone_polys)}")
        log.info(f"         |  Total alerts    : {self.alert_mgr.alert_count}")
        log.info(f"         |  Output          : {self.output_path or 'Not saved'}")
        log.info("=" * 60)

        return {
            "frames_processed": frame_no,
            "zones": len(zone_polys),
            "total_alerts": self.alert_mgr.alert_count,
            "output_path": self.output_path,
        }


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Illegal Parking Detection System")
    parser.add_argument("video", help="Path to input .mov/.mp4 video file")
    parser.add_argument("--dwell", type=float, default=ILLEGAL_DWELL_MINUTES,
                        help=f"Dwell-time threshold in minutes (default {ILLEGAL_DWELL_MINUTES})")
    parser.add_argument("--output", "-o", default=None,
                        help="Optional path to save annotated output video")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable real-time window (headless mode)")
    parser.add_argument("--zone", default=None,
                        help=(
                            'Pre-defined zone(s) as JSON. '
                            'Single zone: \'[[100,100],[400,100],[400,400],[100,400]]\'. '
                            'Multiple zones: \'[[[100,100],[400,100],[400,400],[100,400]],'
                            '[[500,100],[700,100],[700,300],[500,300]]]\''
                        ))
    args = parser.parse_args()

    zone_points = None
    if args.zone:
        try:
            zone_points = json.loads(args.zone)
            log.info(f"Using preset zone(s): {zone_points}")
        except json.JSONDecodeError:
            log.warning("Could not parse --zone JSON; will open annotation window.")

    detector = IllegalParkingDetector(
        video_path=args.video,
        dwell_minutes=args.dwell,
        output_path=args.output,
        show_window=not args.no_display,
        zone_points=zone_points,
    )
    detector.run()


if __name__ == "__main__":
    main()