"""
core/illegal_parking.py 
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.config import (
    CLR_BLACK, CLR_CYAN, CLR_GREEN, CLR_ORANGE, CLR_RED,
    CLR_WHITE, DISPLAY_H, DISPLAY_W,
    ILLEGAL_DWELL_MINUTES, ZONE_PALETTE, BBox, Track,
)
from core.video_utils import make_writer, open_video, video_props
from detectors.classical import preprocess_frame
from trackers.sort_lite import SortLiteTracker
from ui._hud_utils import alpha_rect, draw_panel, draw_text_box

log = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, dwell_minutes: float = ILLEGAL_DWELL_MINUTES) -> None:
        self.threshold_sec = dwell_minutes * 60.0
        self.alert_count   = 0

    def check(self, track: Track) -> None:
        if track.first_seen_in_zone is None or track.alert_fired:
            return
        if track.dwell_seconds >= self.threshold_sec:
            track.alert_fired = True
            track.color       = CLR_RED
            self.alert_count += 1
            log.warning(
                "ILLEGAL PARKING | ID=%d | Dwell=%.1f min | BBox=(%d,%d,%d,%d)",
                track.track_id, track.dwell_seconds / 60,
                track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2,
            )


def _zone_bounds(polys: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
    """Pre-compute axis-aligned bounding boxes for each zone polygon."""
    bounds = []
    for p in polys:
        x1, y1 = p[:, 0].min(), p[:, 1].min()
        x2, y2 = p[:, 0].max(), p[:, 1].max()
        bounds.append((int(x1), int(y1), int(x2), int(y2)))
    return bounds


def in_any_zone(
    polys: List[np.ndarray],
    bounds: List[Tuple[int, int, int, int]],
    bbox: BBox,
) -> bool:
    """
    Fast zone membership test.
    1. Bounding-box pre-filter (integer arithmetic, ~10× faster than PPC test).
    2. pointPolygonTest only if the centroid is inside the AABB.
    """
    cx, cy = float(bbox.cx), float(bbox.cy)
    pt = (cx, cy)
    for (zx1, zy1, zx2, zy2), poly in zip(bounds, polys):
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            if cv2.pointPolygonTest(poly, pt, False) >= 0:
                return True
    return False


def _motion_leaving_zone(track: Track) -> bool:
    """
    Heuristic: if a track is moving fast and has very recently entered a zone
    it may be a transient crossing — don't start dwell timer yet.
    Speed threshold: 8 px/frame  (~20 km/h at typical parking-lot video scales).
    Grace period: first 0.5 s after entering.
    """
    if track.first_seen_in_zone is None:
        return False
    speed = (track.vx ** 2 + track.vy ** 2) ** 0.5
    dwell = time.time() - track.first_seen_in_zone
    return speed > 8.0 and dwell < 0.5


_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hud(
    frame: np.ndarray,
    tracks: List[Track],
    alert_mgr: AlertManager,
    zone_polys: List[np.ndarray],
    frame_no: int,
    fps: float,
    free_slots: Optional[int] = None,
    total_slots: Optional[int] = None,
) -> None:
    """
    Render HUD onto *frame* in-place.
    Uses alpha_rect for the info panel (ROI copy only, not full frame).
    """
    
    for i, poly in enumerate(zone_polys):
        clr  = ZONE_PALETTE[i % len(ZONE_PALETTE)]
        over = frame.copy()
        cv2.fillPoly(over, [poly], clr)
        cv2.addWeighted(over, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [poly], True, clr, 2)
        cx  = int(poly[:, 0].mean())
        cy  = int(poly[:, 1].mean())
        lbl = f"NO PARK Z{i}"
        (tw, th), _ = cv2.getTextSize(lbl, _FONT, 0.65, 2)
        cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 4),
                              (cx + tw // 2 + 4, cy + 4), CLR_BLACK, -1)
        cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 4),
                              (cx + tw // 2 + 4, cy + 4), clr, 2)
        cv2.putText(frame, lbl, (cx - tw // 2, cy), _FONT, 0.65, clr, 2, cv2.LINE_AA)

    for t in tracks:
        if t.missed > 5:
            continue
        b, clr = t.bbox, t.color
        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), clr, 2)
        lbl = f"ID:{t.track_id}"
        if t.first_seen_in_zone is not None:
            lbl += f"  {t.dwell_seconds / 60:.1f}m"
            ratio   = min(1.0, t.dwell_seconds / max(alert_mgr.threshold_sec, 1e-5))
            bar_w   = int((b.x2 - b.x1) * ratio)
            bar_clr = CLR_GREEN if ratio < 0.5 else CLR_ORANGE if ratio < 0.9 else CLR_RED
            cv2.rectangle(frame, (b.x1, b.y2 + 2), (b.x1 + bar_w, b.y2 + 7), bar_clr, -1)
        if t.alert_fired:
            cv2.rectangle(frame, (b.x1 - 3, b.y1 - 3), (b.x2 + 3, b.y2 + 3), CLR_RED, 3)
            cv2.putText(frame, "ALERT!", (b.x1, b.y1 - 10),
                        _FONT, 0.7, CLR_RED, 2, cv2.LINE_AA)
        cv2.putText(frame, lbl, (b.x1 + 2, b.y1 - 6), _FONT, 0.50, clr, 1, cv2.LINE_AA)

    active  = sum(1 for t in tracks if t.missed == 0)
    in_zone = sum(1 for t in tracks if t.first_seen_in_zone is not None and t.missed == 0)
    lines: List[Tuple[str, Tuple]] = [
        (f"Frame          : {frame_no}",                          CLR_WHITE),
        (f"FPS            : {fps:.1f}",                           CLR_WHITE),
        (f"Zones          : {len(zone_polys)}",                   CLR_CYAN),
        (f"Active vehicles: {active}",                            CLR_CYAN),
        (f"In no-park zone: {in_zone}",                           CLR_ORANGE),
        (f"Alerts fired   : {alert_mgr.alert_count}",             CLR_RED),
        (f"Threshold      : {alert_mgr.threshold_sec / 60:.1f} min", CLR_WHITE),
    ]
    if free_slots is not None:
        lines.append((f"Free slots     : {free_slots}/{total_slots}", CLR_GREEN))

    draw_panel(frame, lines, x=10, y=10,
               font_scale=0.72, thickness=2, line_height=34,
               bg_color=(10, 10, 10), bg_alpha=0.75,
               border_color=(160, 160, 160))

class IllegalParkingDetector:
    def __init__(
        self,
        video_path:     str,
        dwell_minutes:  float = ILLEGAL_DWELL_MINUTES,
        output_path:    Optional[str] = None,
        show_window:    bool = True,
        zone_polys:     Optional[List[np.ndarray]] = None,
        detector=None,
        slot_positions: Optional[list] = None,
        picker_w:       int = 0,
        picker_h:       int = 0,
    ) -> None:
        self.video_path     = video_path
        self.output_path    = output_path
        self.show_window    = show_window
        self.preset_zones   = zone_polys
        self.slot_positions = slot_positions or []
        self.picker_w       = picker_w
        self.picker_h       = picker_h
        self.tracker        = SortLiteTracker()
        self.alert_mgr      = AlertManager(dwell_minutes)
        self.detector       = detector

    def _get_zones(self, first_frame: np.ndarray) -> List[np.ndarray]:
        if self.preset_zones:
            return self.preset_zones
        from ui.zone_annotator import ZoneAnnotator
        zones = ZoneAnnotator(first_frame).run()
        if not zones:
            h, w = first_frame.shape[:2]
            zones = [np.array(
                [[w // 4, h // 4], [3 * w // 4, h // 4],
                 [3 * w // 4, 3 * h // 4], [w // 4, 3 * h // 4]],
                dtype=np.int32,
            )]
            log.warning("No zones drawn — using default centre rectangle.")
        return zones


    def run(self) -> dict:
        cap = open_video(self.video_path)
        fw, fh, src_fps, total = video_props(cap)
        log.info("Video: %d×%d @ %.1f fps, %d frames", fw, fh, src_fps, total)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read first frame.")
        zone_polys  = self._get_zones(first_frame)
        zone_bounds = _zone_bounds(zone_polys)
        log.info("%d no-parking zone(s) active.", len(zone_polys))

        from core.config import PICKER_W, PICKER_H, SLOT_W, SLOT_H
        sx = fw / (self.picker_w or PICKER_W)
        sy = fh / (self.picker_h or PICKER_H)

        writer = make_writer(self.output_path, src_fps, (fw, fh)) if self.output_path else None

        if self.show_window:
            cv2.namedWindow("Parking System", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Parking System", DISPLAY_W, DISPLAY_H)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Rolling FPS window (16 samples)
        _fps_times: List[float] = [time.perf_counter()]
        _fps_window = 16
        disp_fps    = src_fps
        frame_no    = 0

        log.info("Processing — press Q or ESC to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1

            gray, processed = preprocess_frame(frame)
            detections      = self.detector.detect(frame, gray)
            tracks          = self.tracker.update(detections)

            for t in tracks:
                if t.missed == 0:
                    if in_any_zone(zone_polys, zone_bounds, t.bbox):
                        t.enter_zone()
                        # Motion gate: suppress transient fast crossings
                        if not _motion_leaving_zone(t):
                            self.alert_mgr.check(t)
                    else:
                        t.leave_zone()

            free_slots = None
            if self.slot_positions:
                from core.slot_occupancy import draw_slots
                free_slots = draw_slots(frame, processed, self.slot_positions, sx, sy)

            now = time.perf_counter()
            _fps_times.append(now)
            if len(_fps_times) > _fps_window:
                _fps_times.pop(0)
            if len(_fps_times) >= 2:
                disp_fps = (len(_fps_times) - 1) / (_fps_times[-1] - _fps_times[0])

            draw_hud(frame, tracks, self.alert_mgr, zone_polys, frame_no, disp_fps,
                     free_slots, len(self.slot_positions))

            if writer:
                writer.write(frame)
            if self.show_window:
                cv2.imshow("Parking System", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        summary = dict(frames=frame_no, zones=len(zone_polys),
                       alerts=self.alert_mgr.alert_count, output=self.output_path)
        log.info("Done | Frames:%d Zones:%d Alerts:%d",
                 frame_no, len(zone_polys), self.alert_mgr.alert_count)
        return summary