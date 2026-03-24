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
# POLYGON ANNOTATION TOOL
# ──────────────────────────────────────────────
class PolygonAnnotator:
    """
    Opens an OpenCV window showing the first frame.
    Left-click to add polygon vertices.
    Press ENTER/SPACE to confirm, ESC to cancel, 'r' to reset.
    """

    def __init__(self, frame: np.ndarray):
        self.frame = frame.copy()
        self.points: List[Tuple[int,int]] = []
        self.done = False
        self.window = "[ ANNOTATE NO-PARKING ZONE ] Left-click to add points | ENTER=confirm | r=reset | ESC=cancel"

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.done:
            self.points.append((x, y))

    def run(self) -> Optional[np.ndarray]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1100, 700)
        cv2.setMouseCallback(self.window, self._mouse_cb)

        while True:
            display = self.frame.copy()
            h, w = display.shape[:2]

            # Instructions overlay
            instructions = [
                "LEFT-CLICK to place polygon vertices",
                "ENTER / SPACE  → confirm zone",
                "'r'            → reset points",
                "ESC            → cancel",
            ]
            for i, txt in enumerate(instructions):
                cv2.putText(display, txt, (10, 25 + i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_BLACK, 3, cv2.LINE_AA)
                cv2.putText(display, txt, (10, 25 + i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_WHITE, 1, cv2.LINE_AA)

            # Draw polygon so far
            if len(self.points) >= 2:
                pts_arr = np.array(self.points, dtype=np.int32)
                cv2.polylines(display, [pts_arr], isClosed=False,
                              color=CLR_YELLOW, thickness=2)
            for pt in self.points:
                cv2.circle(display, pt, 5, CLR_RED, -1)

            # Close polygon preview when ≥3 points
            if len(self.points) >= 3:
                pts_arr = np.array(self.points, dtype=np.int32)
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts_arr], CLR_ZONE)
                cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
                cv2.polylines(display, [pts_arr], isClosed=True,
                              color=CLR_ZONE, thickness=2)

            cv2.putText(display, f"Points: {len(self.points)}", (w-170, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_YELLOW, 2)

            cv2.imshow(self.window, display)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 32):  # ENTER or SPACE
                if len(self.points) >= 3:
                    log.info(f"Zone confirmed with {len(self.points)} vertices.")
                    cv2.destroyAllWindows()
                    return np.array(self.points, dtype=np.int32)
                else:
                    log.warning("Need at least 3 points to define a zone.")
            elif key == ord('r'):
                self.points = []
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                log.warning("Annotation cancelled by user.")
                return None

        cv2.destroyAllWindows()
        return None


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
    Divides a rectangular region (outside the no-parking zone) into
    a grid of parking spots and checks occupancy by overlap with tracks.
    """

    def __init__(self, frame_w: int, frame_h: int,
                 no_park_poly: Optional[np.ndarray],
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
                # Only include spots not heavily overlapping the no-park zone
                if no_park_poly is not None:
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    if cv2.pointPolygonTest(no_park_poly, (cx, cy), False) >= 0:
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
def is_in_zone(poly: np.ndarray, bbox: BBox, threshold: float = 0.5) -> bool:
    """
    Returns True if the centre of the bounding box is inside the polygon.
    A fraction check (what fraction of bbox corners are inside) can be used
    for stricter/looser matching.
    """
    cx, cy = bbox.cx, bbox.cy
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


# ──────────────────────────────────────────────
# HUD RENDERER
# ──────────────────────────────────────────────
def draw_hud(frame: np.ndarray,
             tracks: List[Track],
             spot_mgr: ParkingSpotManager,
             alert_mgr: AlertManager,
             zone_poly: np.ndarray,
             frame_no: int,
             fps: float):
    h, w = frame.shape[:2]

    # ── Zone overlay ──
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_poly], CLR_ZONE)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [zone_poly], True, CLR_ZONE, 2)
    # Zone label
    zx, zy = zone_poly[0]
    cv2.putText(frame, "NO PARKING ZONE", (int(zx)+4, int(zy)-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_ZONE, 2, cv2.LINE_AA)

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
                 zone_points: Optional[List[Tuple[int,int]]] = None):
        self.video_path = video_path
        self.dwell_minutes = dwell_minutes
        self.output_path = output_path
        self.show_window = show_window
        self.preset_zone = (np.array(zone_points, dtype=np.int32)
                            if zone_points else None)

        self.detector = VehicleDetector()
        self.tracker  = SortLiteTracker()
        self.alert_mgr = AlertManager(dwell_minutes)

    # ── STEP 1: Annotation ──────────────────
    def _annotate_zone(self, first_frame: np.ndarray) -> np.ndarray:
        if self.preset_zone is not None:
            log.info("Using preset zone polygon.")
            return self.preset_zone
        log.info("Opening annotation window …")
        ann = PolygonAnnotator(first_frame)
        poly = ann.run()
        if poly is None:
            # Fallback: centre 40% of frame
            h, w = first_frame.shape[:2]
            poly = np.array([
                [w//4, h//4], [3*w//4, h//4],
                [3*w//4, 3*h//4], [w//4, 3*h//4],
            ], dtype=np.int32)
            log.warning("No zone annotated — using default centre rectangle.")
        return poly

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

        # STEP 1 — Annotate zone
        zone_poly = self._annotate_zone(first_frame)

        # Spot manager
        spot_mgr = ParkingSpotManager(frame_w, frame_h, zone_poly)
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
                    if is_in_zone(zone_poly, t.bbox):
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
                     zone_poly, frame_no, display_fps)

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
        log.info(f"         |  Total alerts fired: {self.alert_mgr.alert_count}")
        log.info(f"         |  Output: {self.output_path or 'Not saved'}")
        log.info("=" * 60)

        return {
            "frames_processed": frame_no,
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
                        help='Pre-defined zone as JSON, e.g. \'[[100,100],[400,100],[400,400],[100,400]]\'')
    args = parser.parse_args()

    zone_points = None
    if args.zone:
        try:
            zone_points = json.loads(args.zone)
            log.info(f"Using preset zone: {zone_points}")
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
