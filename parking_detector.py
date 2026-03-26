import cv2
import numpy as np
import time
import json
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment

# ──────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ──────────────────────────────────────────────
ILLEGAL_DWELL_MINUTES: float = 10.0  # alert threshold (minutes)
IOU_THRESHOLD:         float = 0.25  # min IoU to match detection → track
MAX_MISSED_FRAMES:     int   = 30    # frames before a track is pruned
MIN_DETECTION_AREA:    int   = 1500  # px² — ignore tiny blobs
BACKGROUND_HISTORY:    int   = 300   # frames for MOG2 background model
BG_THRESHOLD:          float = 40.0  # MOG2 sensitivity

# Colours (BGR)
CLR_GREEN  = (50,  205,  50)
CLR_RED    = (0,     0, 220)
CLR_YELLOW = (0,   215, 255)
CLR_ORANGE = (0,   140, 255)
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0,     0,   0)
CLR_CYAN   = (255, 220,   0)

# Zone overlay colours — one per zone, cycles if >8 zones
ZONE_PALETTE: List[Tuple[int,int,int]] = [
    (0,   80, 255),   # zone 0 — blue-red
    (0,  200, 100),   # zone 1 — green
    (255,140,   0),   # zone 2 — cyan
    (180,  0, 180),   # zone 3 — magenta
    (0,  220, 220),   # zone 4 — yellow
    (255, 80,  80),   # zone 5 — blue
    (30, 180, 255),   # zone 6 — amber
    (100,255, 150),   # zone 7 — lime
]

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
    first_seen_in_zone: Optional[float] = None
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


# ──────────────────────────────────────────────
# SORT-LITE TRACKER
# ──────────────────────────────────────────────
class SortLiteTracker:
    def __init__(self):
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[BBox]) -> List[Track]:
        if not self.tracks:
            for d in detections:
                t = Track(self._next_id, d, color=self._rnd_color(self._next_id))
                self.tracks[self._next_id] = t
                self._next_id += 1
            return list(self.tracks.values())

        track_ids    = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]

        matched_tracks: set = set()
        matched_dets:   set = set()

        if detections and track_bboxes:
            cost = np.zeros((len(track_bboxes), len(detections)), dtype=np.float32)
            for r, tb in enumerate(track_bboxes):
                for c, db in enumerate(detections):
                    cost[r, c] = 1.0 - tb.iou(db)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - IOU_THRESHOLD):
                    tid = track_ids[r]
                    self.tracks[tid].bbox   = detections[c]
                    self.tracks[tid].missed = 0
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        for i, tid in enumerate(track_ids):
            if tid not in matched_tracks:
                self.tracks[tid].missed += 1

        for c, d in enumerate(detections):
            if c not in matched_dets:
                t = Track(self._next_id, d, color=self._rnd_color(self._next_id))
                self.tracks[self._next_id] = t
                self._next_id += 1

        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if t.missed <= MAX_MISSED_FRAMES}
        return list(self.tracks.values())

    @staticmethod
    def _rnd_color(seed: int) -> Tuple[int,int,int]:
        rng = np.random.default_rng(seed * 137 + 42)
        return tuple(int(x) for x in rng.integers(80, 220, 3))


# ──────────────────────────────────────────────
# MULTI-ZONE ANNOTATOR
# ──────────────────────────────────────────────
class MultiZoneAnnotator:
    """
    Controls
    --------
    Left-click          Add vertex to current polygon
    ENTER / SPACE       Finish current polygon → save, start new one
    Z / Ctrl+Z          Undo last vertex or last completed zone
    Y / Ctrl+Y          Redo
    D / Ctrl+D          Delete zone nearest to cursor
    R                   Reset in-progress polygon
    F / ESC             Done → begin detection
    """

    def __init__(self, frame: np.ndarray):
        self.frame          = frame.copy()
        self.window         = "ANNOTATE NO-PARKING ZONES  |  F or ESC when done"
        self.zones:         List[List[Tuple[int,int]]] = []
        self._redo_stack:   List[List[Tuple[int,int]]] = []
        self.current:       List[Tuple[int,int]]       = []
        self._mx: int = 0
        self._my: int = 0
        self._hover_delete: int = -1

    def _zone_color(self, idx: int) -> Tuple[int,int,int]:
        return ZONE_PALETTE[idx % len(ZONE_PALETTE)]

    def _poly_centroid(self, pts):
        arr = np.array(pts, dtype=np.float32)
        return int(arr[:,0].mean()), int(arr[:,1].mean())

    def _nearest_zone(self) -> int:
        best_idx, best_d = -1, float('inf')
        for i, z in enumerate(self.zones):
            cx, cy = self._poly_centroid(z)
            d = (cx-self._mx)**2 + (cy-self._my)**2
            if d < best_d:
                best_d, best_idx = d, i
        return best_idx

    def _undo(self):
        if self.current:
            self.current.pop()
        elif self.zones:
            self._redo_stack.append(self.zones.pop())
            log.info(f"Undo → {len(self.zones)} zones.")

    def _redo(self):
        if self._redo_stack:
            self.zones.append(self._redo_stack.pop())
            log.info(f"Redo → {len(self.zones)} zones.")

    def _delete_nearest(self):
        idx = self._nearest_zone()
        if idx >= 0:
            self._redo_stack.append(self.zones.pop(idx))
            log.info(f"Deleted zone {idx} → {len(self.zones)} zones.")

    def _finish_current(self):
        if len(self.current) >= 3:
            self.zones.append(list(self.current))
            self._redo_stack.clear()
            log.info(f"Zone {len(self.zones)-1} saved ({len(self.current)} pts).")
            self.current = []
        else:
            log.warning("Need ≥ 3 points.")

    def _mouse_cb(self, event, x, y, flags, param):
        self._mx, self._my = x, y
        self._hover_delete = self._nearest_zone()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y))
            self._redo_stack.clear()

    # ── drawing helpers ──
    @staticmethod
    def _filled_rect(img, x1, y1, x2, y2, color, alpha=0.82):
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(img.shape[1],x2), min(img.shape[0],y2)
        if x2<=x1 or y2<=y1: return
        sub = img[y1:y2, x1:x2]
        cv2.addWeighted(np.full_like(sub, color), alpha, sub, 1-alpha, 0, sub)
        img[y1:y2, x1:x2] = sub

    @staticmethod
    def _border_rect(img, x1, y1, x2, y2, color, t=2):
        cv2.rectangle(img, (x1,y1), (x2,y2), color, t)

    @staticmethod
    def _key_badge(img, x, y, label, badge_clr, text_clr=CLR_BLACK):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.70, 2)
        px, py = 10, 6
        bx1, by1 = x, y-th-py
        bx2, by2 = x+tw+px*2, y+py
        cv2.rectangle(img, (bx1,by1), (bx2,by2), badge_clr, -1)
        cv2.rectangle(img, (bx1,by1), (bx2,by2), CLR_WHITE, 2)
        cv2.putText(img, label, (bx1+px, y), font, 0.70, text_clr, 2, cv2.LINE_AA)
        return bx2+12

    def _render(self) -> np.ndarray:
        display = self.frame.copy()
        h, w    = display.shape[:2]
        font    = cv2.FONT_HERSHEY_SIMPLEX

        # ── Completed zones ──
        for i, zone in enumerate(self.zones):
            clr = self._zone_color(i)
            pts = np.array(zone, dtype=np.int32)
            over = display.copy()
            cv2.fillPoly(over, [pts], clr)
            alpha = 0.40 if i != self._hover_delete else 0.58
            cv2.addWeighted(over, alpha, display, 1-alpha, 0, display)
            cv2.polylines(display, [pts], True, clr,
                          4 if i == self._hover_delete else 2)
            for pt in zone:
                cv2.circle(display, tuple(pt), 6, CLR_WHITE, -1)
                cv2.circle(display, tuple(pt), 6, clr, 2)
            cx, cy = self._poly_centroid(zone)
            lbl = f"Zone {i}"
            (tw, th), _ = cv2.getTextSize(lbl, font, 0.90, 2)
            cv2.rectangle(display,
                          (cx-tw//2-8, cy-th-6), (cx+tw//2+8, cy+6),
                          CLR_BLACK, -1)
            cv2.rectangle(display,
                          (cx-tw//2-8, cy-th-6), (cx+tw//2+8, cy+6),
                          clr, 2)
            cv2.putText(display, lbl, (cx-tw//2, cy), font, 0.90, clr, 2, cv2.LINE_AA)
            if i == self._hover_delete:
                hint = "[ D ]  delete this zone"
                (hw, _), _ = cv2.getTextSize(hint, font, 0.65, 2)
                cv2.putText(display, hint, (cx-hw//2, cy+28),
                            font, 0.65, (0,80,255), 2, cv2.LINE_AA)

        # ── In-progress polygon ──
        clr_cur = self._zone_color(len(self.zones))
        if len(self.current) >= 2:
            cv2.polylines(display,
                          [np.array(self.current, dtype=np.int32)],
                          False, clr_cur, 3)
        if len(self.current) >= 3:
            over2 = display.copy()
            cv2.fillPoly(over2, [np.array(self.current, dtype=np.int32)], clr_cur)
            cv2.addWeighted(over2, 0.18, display, 0.82, 0, display)
            cv2.polylines(display,
                          [np.array(self.current, dtype=np.int32)],
                          True, clr_cur, 2)
        for pt in self.current:
            cv2.circle(display, pt, 7, clr_cur, -1)
            cv2.circle(display, pt, 8, CLR_WHITE, 2)
        if self.current:
            cv2.line(display, self.current[-1], (self._mx, self._my),
                     clr_cur, 2, cv2.LINE_AA)

        # ── Controls panel (top-left) ──
        FS_TITLE = 0.85; FS_DESC = 0.72
        LH = 46; PAD_L = 16; DIV_H = 14; TITLE_H = 52
        GOLD   = (20, 160, 220)
        ORANGE = (0,  120, 255)
        GREEN  = (30, 140,  30)
        RED_B  = (30,  30, 190)

        rows_data = [
            ("Left-Click",  GOLD,   "Add a vertex to current polygon",  CLR_WHITE),
            ("Enter/Space", GOLD,   "Finish zone  →  start a new one",  CLR_WHITE),
            None,
            ("Z / Ctrl+Z",  ORANGE, "Undo last vertex or whole zone",   CLR_WHITE),
            ("Y / Ctrl+Y",  ORANGE, "Redo last undone zone",            CLR_WHITE),
            ("D / Ctrl+D",  RED_B,  "Delete zone nearest to cursor",    (100,160,255)),
            ("R",           ORANGE, "Reset in-progress polygon",        CLR_WHITE),
            None,
            ("F  /  Esc",   GREEN,  "Done  →  begin detection",         (100,255,100)),
        ]

        def _bw(lbl):
            (tw,_),_ = cv2.getTextSize(lbl, font, 0.70, 2)
            return tw + 20 + 12

        def _dw(txt):
            (tw,_),_ = cv2.getTextSize(txt, font, FS_DESC, 2)
            return tw

        max_badge = max(_bw(r[0]) for r in rows_data if r)
        max_desc  = max(_dw(r[2]) for r in rows_data if r)
        panel_w   = max(PAD_L + max_badge + max_desc + PAD_L, 560)
        n_real    = sum(1 for r in rows_data if r)
        n_div     = sum(1 for r in rows_data if r is None)
        panel_h   = TITLE_H + n_real*LH + n_div*DIV_H + 20

        px1, py1 = 14, 14
        px2, py2 = px1+panel_w, py1+panel_h

        self._filled_rect(display, px1, py1, px2, py2, (10,10,10), 0.88)
        self._border_rect(display, px1, py1, px2, py2, (220,220,220), 2)
        cv2.rectangle(display, (px1,py1), (px2,py1+TITLE_H), (35,35,35), -1)
        cv2.rectangle(display, (px1,py1), (px2,py1+TITLE_H), (220,220,220), 2)
        cv2.putText(display, "NO-PARKING ZONE ANNOTATOR",
                    (px1+PAD_L, py1+TITLE_H-14), font, FS_TITLE, CLR_YELLOW, 2, cv2.LINE_AA)

        ry = py1+TITLE_H+LH-6
        for row in rows_data:
            if row is None:
                cv2.line(display, (px1+PAD_L, ry-LH//2+DIV_H//2),
                         (px2-PAD_L, ry-LH//2+DIV_H//2), (100,100,100), 1)
                ry += DIV_H
                continue
            lbl, bclr, desc, dclr = row
            after = self._key_badge(display, px1+PAD_L, ry, lbl, bclr)
            cv2.putText(display, desc, (after, ry), font, FS_DESC, dclr, 2, cv2.LINE_AA)
            ry += LH

        # ── Status card (top-right) ──
        FS_S = 0.72; SLH = 38; STH = 44
        status = [
            (f"Zones saved    :  {len(self.zones)}",       (80,255,200)),
            (f"In-progress pts:  {len(self.current)}",
             CLR_YELLOW if self.current else (140,140,140)),
            (f"Redo available :  {len(self._redo_stack)}",
             CLR_GREEN if self._redo_stack else (140,140,140)),
        ]
        sw = max(_dw(t) for t,_ in status) + PAD_L*2
        sw = max(sw, 380)
        sh = STH + len(status)*SLH + 12
        sx1 = w-sw-14; sy1 = 14; sx2 = w-14; sy2 = sy1+sh
        self._filled_rect(display, sx1, sy1, sx2, sy2, (10,10,10), 0.88)
        self._border_rect(display, sx1, sy1, sx2, sy2, (220,220,220), 2)
        cv2.rectangle(display, (sx1,sy1), (sx2,sy1+STH), (35,35,35), -1)
        cv2.rectangle(display, (sx1,sy1), (sx2,sy1+STH), (220,220,220), 2)
        cv2.putText(display, "STATUS", (sx1+PAD_L, sy1+STH-12),
                    font, FS_TITLE, CLR_YELLOW, 2, cv2.LINE_AA)
        for k, (txt, clr) in enumerate(status):
            cv2.putText(display, txt, (sx1+PAD_L, sy1+STH+(k+1)*SLH),
                        font, FS_S, clr, 2, cv2.LINE_AA)

        # ── Zone legend (bottom-left) ──
        if self.zones:
            LEG_LH=36; LEG_TH=40; SW=20
            lw=300; lh2=LEG_TH+len(self.zones)*LEG_LH+10
            lx1=14; ly1=h-lh2-14; lx2=lx1+lw; ly2=ly1+lh2
            self._filled_rect(display, lx1, ly1, lx2, ly2, (10,10,10), 0.88)
            self._border_rect(display, lx1, ly1, lx2, ly2, (220,220,220), 2)
            cv2.putText(display, "Zone legend", (lx1+PAD_L, ly1+28),
                        font, 0.70, (200,200,200), 2, cv2.LINE_AA)
            for zi, zone in enumerate(self.zones):
                zclr = self._zone_color(zi)
                zy   = ly1+LEG_TH+(zi+1)*LEG_LH-4
                cv2.rectangle(display,
                              (lx1+PAD_L, zy-SW+4), (lx1+PAD_L+SW, zy+4), zclr, -1)
                cv2.rectangle(display,
                              (lx1+PAD_L, zy-SW+4), (lx1+PAD_L+SW, zy+4), CLR_WHITE, 1)
                cv2.putText(display, f"Zone {zi}   ({len(zone)} vertices)",
                            (lx1+PAD_L+SW+8, zy), font, 0.68, zclr, 2, cv2.LINE_AA)

        # ── First-click hint (centre-bottom) ──
        if not self.zones and not self.current:
            hint = "Click anywhere on the frame to place the first vertex"
            (hw, _), _ = cv2.getTextSize(hint, font, 0.80, 2)
            hx = max(10, w//2-hw//2); hy = h-24
            cv2.putText(display, hint, (hx+2,hy+2), font, 0.80, CLR_BLACK, 3, cv2.LINE_AA)
            cv2.putText(display, hint, (hx,  hy),   font, 0.80, CLR_YELLOW, 2, cv2.LINE_AA)

        return display

    def run(self) -> List[np.ndarray]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 780)
        cv2.setMouseCallback(self.window, self._mouse_cb)
        log.info("Annotation window open — draw your no-parking zones, then press F or ESC.")

        while True:
            cv2.imshow(self.window, self._render())
            key = cv2.waitKey(30) & 0xFF
            if key == 255: continue

            if   key in (13, 32):                            self._finish_current()
            elif key in (26, ord('z'), ord('Z')):            self._undo()
            elif key in (25, ord('y'), ord('Y')):            self._redo()
            elif key in (4,  ord('d'), ord('D')):            self._delete_nearest()
            elif key in (ord('r'), ord('R')):
                if self.current:
                    log.info(f"Reset in-progress ({len(self.current)} pts).")
                    self.current = []
            elif key in (ord('f'), ord('F'), 27):
                if len(self.current) >= 3:
                    log.info("Auto-closing in-progress polygon.")
                    self._finish_current()
                break

        cv2.destroyAllWindows()
        result = [np.array(z, dtype=np.int32) for z in self.zones]
        log.info(f"Annotation complete: {len(result)} zone(s).")
        return result


# ──────────────────────────────────────────────
# VEHICLE DETECTOR  (classical DIP)
# ──────────────────────────────────────────────
class VehicleDetector:
    """
    MOG2 background subtraction → morphological cleanup → contour filtering.
    Returns List[BBox] of detected vehicle blobs per frame.
    """
    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=BACKGROUND_HISTORY,
            varThreshold=BG_THRESHOLD,
            detectShadows=True,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> List[BBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        fg = self.bg_sub.apply(gray)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        fg = cv2.dilate(fg, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        detections: List[BBox] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_DETECTION_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if not (0.3 < bw / (bh + 1e-5) < 5.0):
                continue
            detections.append(BBox(max(0,x), max(0,y), min(w,x+bw), min(h,y+bh)))
        return detections


# ──────────────────────────────────────────────
# ALERT SYSTEM
# ──────────────────────────────────────────────
class AlertManager:
    def __init__(self, dwell_minutes: float = ILLEGAL_DWELL_MINUTES):
        self.threshold_sec = dwell_minutes * 60.0
        self.alert_count   = 0

    def check(self, track: Track):
        if track.first_seen_in_zone is None:
            return
        if not track.alert_fired and track.dwell_seconds >= self.threshold_sec:
            track.alert_fired = True
            track.color       = CLR_RED
            self.alert_count += 1
            log.warning(
                f"ILLEGAL PARKING ALERT | "
                f"Vehicle ID={track.track_id} | "
                f"Dwell={track.dwell_seconds/60:.1f} min | "
                f"BBox=({track.bbox.x1},{track.bbox.y1},{track.bbox.x2},{track.bbox.y2})"
            )


# ──────────────────────────────────────────────
# ZONE CHECKER
# ──────────────────────────────────────────────
def is_in_any_zone(polys: List[np.ndarray], bbox: BBox) -> bool:
    cx, cy = float(bbox.cx), float(bbox.cy)
    return any(
        cv2.pointPolygonTest(p, (cx, cy), False) >= 0
        for p in polys
    )


# ──────────────────────────────────────────────
# HUD RENDERER
# ──────────────────────────────────────────────
def draw_hud(frame: np.ndarray,
             tracks: List[Track],
             alert_mgr: AlertManager,
             zone_polys: List[np.ndarray],
             frame_no: int,
             fps: float):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Zone overlays ──
    for i, poly in enumerate(zone_polys):
        clr = ZONE_PALETTE[i % len(ZONE_PALETTE)]
        over = frame.copy()
        cv2.fillPoly(over, [poly], clr)
        cv2.addWeighted(over, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [poly], True, clr, 2)
        cx = int(poly[:,0].mean()); cy = int(poly[:,1].mean())
        lbl = f"NO PARK Z{i}"
        (tw, th), _ = cv2.getTextSize(lbl, font, 0.7, 2)
        cv2.rectangle(frame, (cx-tw//2-5, cy-th-5), (cx+tw//2+5, cy+5), CLR_BLACK, -1)
        cv2.rectangle(frame, (cx-tw//2-5, cy-th-5), (cx+tw//2+5, cy+5), clr, 2)
        cv2.putText(frame, lbl, (cx-tw//2, cy), font, 0.7, clr, 2, cv2.LINE_AA)

    # ── Vehicle bounding boxes + dwell bars ──
    for t in tracks:
        if t.missed > 5:
            continue
        b   = t.bbox
        clr = t.color

        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), clr, 2)

        lbl = f"ID:{t.track_id}"
        if t.first_seen_in_zone is not None:
            lbl += f"  {t.dwell_seconds/60:.1f}m"
            ratio   = min(1.0, t.dwell_seconds / alert_mgr.threshold_sec)
            bar_w   = int((b.x2 - b.x1) * ratio)
            bar_clr = (CLR_GREEN if ratio < 0.5 else
                       CLR_ORANGE if ratio < 0.9 else CLR_RED)
            cv2.rectangle(frame, (b.x1, b.y2+2), (b.x1+bar_w, b.y2+7), bar_clr, -1)

        if t.alert_fired:
            cv2.rectangle(frame, (b.x1-3, b.y1-3), (b.x2+3, b.y2+3), CLR_RED, 3)
            cv2.putText(frame, "ALERT!", (b.x1, b.y1-10),
                        font, 0.7, CLR_RED, 2, cv2.LINE_AA)

        cv2.putText(frame, lbl, (b.x1+2, b.y1-6),
                    font, 0.5, clr, 1, cv2.LINE_AA)

    # ── Info panel (top-left) ──
    active  = sum(1 for t in tracks if t.missed == 0)
    in_zone = sum(1 for t in tracks if t.first_seen_in_zone is not None and t.missed == 0)

    panel_lines = [
        (f"Frame          : {frame_no}",                   CLR_WHITE),
        (f"FPS            : {fps:.1f}",                    CLR_WHITE),
        (f"Zones          : {len(zone_polys)}",            CLR_CYAN),
        (f"Active vehicles: {active}",                     CLR_CYAN),
        (f"In no-park zone: {in_zone}",                    CLR_ORANGE),
        (f"Alerts fired   : {alert_mgr.alert_count}",      CLR_RED),
        (f"Threshold      : {ILLEGAL_DWELL_MINUTES:.1f} min", CLR_WHITE),
    ]

    FS = 1.0; LH = 38; PAD = 10
    panel_w = 520
    panel_h = len(panel_lines) * LH + PAD * 2

    sub = frame[PAD:PAD+panel_h, PAD:PAD+panel_w]
    cv2.addWeighted(np.zeros_like(sub), 0.55, sub, 0.45, 0, sub)
    frame[PAD:PAD+panel_h, PAD:PAD+panel_w] = sub
    cv2.rectangle(frame, (PAD, PAD), (PAD+panel_w, PAD+panel_h), (180,180,180), 1)

    for i, (txt, clr) in enumerate(panel_lines):
        cv2.putText(frame, txt,
                    (PAD+10, PAD+LH//2 + i*LH + 8),
                    font, FS, clr, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
class IllegalParkingDetector:
    def __init__(self,
                 video_path:   str,
                 dwell_minutes: float = ILLEGAL_DWELL_MINUTES,
                 output_path:  Optional[str] = None,
                 show_window:  bool = True,
                 zone_points:  Optional[List] = None):
        self.video_path    = video_path
        self.output_path   = output_path
        self.show_window   = show_window
        self.detector      = VehicleDetector()
        self.tracker       = SortLiteTracker()
        self.alert_mgr     = AlertManager(dwell_minutes)

        self.preset_zones: Optional[List[np.ndarray]] = None
        if zone_points is not None:
            if zone_points and not isinstance(zone_points[0][0], (list, tuple)):
                zone_points = [zone_points]
            self.preset_zones = [np.array(z, dtype=np.int32) for z in zone_points]

    def _annotate_zones(self, first_frame: np.ndarray) -> List[np.ndarray]:
        if self.preset_zones is not None:
            log.info(f"Using {len(self.preset_zones)} preset zone(s).")
            return self.preset_zones
        log.info("Opening annotation window…")
        zones = MultiZoneAnnotator(first_frame).run()
        if not zones:
            h, w = first_frame.shape[:2]
            zones = [np.array([[w//4,h//4],[3*w//4,h//4],
                                [3*w//4,3*h//4],[w//4,3*h//4]], dtype=np.int32)]
            log.warning("No zones annotated — using default centre rectangle.")
        return zones

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            log.error(f"Cannot open video: {self.video_path}"); return

        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info(f"Video: {frame_w}×{frame_h} @ {src_fps:.1f} fps, {total_fr} frames")

        ret, first_frame = cap.read()
        if not ret:
            log.error("Cannot read first frame."); return

        zone_polys = self._annotate_zones(first_frame)
        log.info(f"{len(zone_polys)} no-parking zone(s) active.")

        writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output_path, fourcc, src_fps, (frame_w, frame_h))
            log.info(f"Writing output to: {self.output_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if self.show_window:
            cv2.namedWindow("Illegal Parking Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Illegal Parking Detector", 1280, 780)

        frame_no   = 0
        fps_timer  = time.time()
        disp_fps   = src_fps
        log.info("Processing — press Q or ESC to stop.")

        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("End of video."); break
            frame_no += 1

            detections = self.detector.detect(frame)
            tracks     = self.tracker.update(detections)

            for t in tracks:
                if t.missed == 0:
                    if is_in_any_zone(zone_polys, t.bbox):
                        t.enter_zone()
                        self.alert_mgr.check(t)
                    else:
                        t.leave_zone()

            if frame_no % 15 == 0:
                now      = time.time()
                disp_fps = 15.0 / max(1e-5, now - fps_timer)
                fps_timer = now

            draw_hud(frame, tracks, self.alert_mgr, zone_polys, frame_no, disp_fps)

            if writer:
                writer.write(frame)

            if self.show_window:
                cv2.imshow("Illegal Parking Detector", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    log.info("Stopped by user."); break

        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

        log.info("=" * 60)
        log.info(f"OUTPUT | Frames: {frame_no} | Zones: {len(zone_polys)} | Alerts: {self.alert_mgr.alert_count}")
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
    parser = argparse.ArgumentParser(description="Illegal Parking Detection System")
    parser.add_argument("video",   help="Path to .mov/.mp4 video file")
    parser.add_argument("--dwell", type=float, default=ILLEGAL_DWELL_MINUTES,
                        help=f"Alert threshold in minutes (default {ILLEGAL_DWELL_MINUTES})")
    parser.add_argument("--output", "-o", default=None,
                        help="Save annotated output to this .mp4 path")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (no window)")
    parser.add_argument("--zone", default=None,
                        help=(
                            "Pre-defined zone(s) as JSON. "
                            "Single: '[[x,y],...]'  "
                            "Multiple: '[[[x,y],...],[[x,y],...]]'"
                        ))
    args = parser.parse_args()

    zone_points = None
    if args.zone:
        try:
            zone_points = json.loads(args.zone)
        except json.JSONDecodeError:
            log.warning("Could not parse --zone JSON; opening annotation window.")

    IllegalParkingDetector(
        video_path    = args.video,
        dwell_minutes = args.dwell,
        output_path   = args.output,
        show_window   = not args.no_display,
        zone_points   = zone_points,
    ).run()


if __name__ == "__main__":
    main()