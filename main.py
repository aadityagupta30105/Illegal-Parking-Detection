#!/usr/bin/env python3
"""
main.py — Unified Parking System CLI
─────────────────────────────────────
Modes
-----
  picker    Annotate rectangular parking slot positions on the first video frame.
            The picker image is extracted automatically if it doesn't exist.

  occupancy Run slot-occupancy detection only (classical preprocessing).

  detect    Full pipeline: vehicle detection + tracking + zone-based illegal
            parking + optional slot-occupancy overlay.

Usage examples
--------------
  python main.py picker    clip.mp4
  python main.py occupancy clip.mp4
  python main.py detect    clip.mp4 --yolo --dwell 2.0 --output out.mp4
  python main.py detect    clip.mp4 --zone '[[x,y],[x,y],[x,y]]'
  python main.py detect    clip.mp4 --no-display
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("parking.log", mode="w"),
    ],
)
log = logging.getLogger("ParkingSystem")


# ── Mode handlers ─────────────────────────────────────────────────────────────
def mode_picker(args: argparse.Namespace) -> None:
    from ui.slot_picker import run_picker
    run_picker(
        video_path=args.video,
        pos_file=args.pos_file,
        img_path=args.picker_img,
    )


def mode_occupancy(args: argparse.Namespace) -> None:
    from core.slot_occupancy import run_slot_occupancy
    run_slot_occupancy(args.video, pos_file=args.pos_file)


def mode_detect(args: argparse.Namespace) -> None:
    from detectors.yolo import build_detector
    detector = build_detector(args.yolo, args.yolo_model, args.yolo_conf)

    # ── Zone polygons (optional) ───────────────────────────────────────────
    zone_polys: Optional[List[np.ndarray]] = None
    if args.zone:
        try:
            pts = json.loads(args.zone)
            if pts and not isinstance(pts[0][0], list):
                pts = [pts]
            zone_polys = [np.array(z, dtype=np.int32) for z in pts]
        except (json.JSONDecodeError, TypeError, IndexError, ValueError) as e:
            log.warning("Could not parse --zone JSON (%s); opening annotation window.", e)

    # ── Slot positions (optional) ──────────────────────────────────────────
    slot_positions: list = []
    if not args.no_slots:
        try:
            from core.slot_occupancy import load_positions
            slot_positions = load_positions(args.pos_file)
            log.info("Loaded %d slot position(s).", len(slot_positions))
        except FileNotFoundError as exc:
            log.info("No slot positions (%s); slot overlay disabled.", exc)

    from core.config import PICKER_H, PICKER_W
    from core.illegal_parking import IllegalParkingDetector
    IllegalParkingDetector(
        video_path    = args.video,
        dwell_minutes = args.dwell,
        output_path   = args.output,
        show_window   = not args.no_display,
        zone_polys    = zone_polys,
        detector      = detector,
        slot_positions= slot_positions,
        picker_w      = PICKER_W,
        picker_h      = PICKER_H,
    ).run()


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Parking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # picker
    pk = sub.add_parser("picker", help="Annotate slot positions")
    pk.add_argument("video")
    pk.add_argument("--pos-file",   default="CarParkPos")
    pk.add_argument("--picker-img", default="clipimage.png")

    # occupancy
    occ = sub.add_parser("occupancy", help="Slot occupancy detection only")
    occ.add_argument("video")
    occ.add_argument("--pos-file", default="CarParkPos")

    # detect
    det = sub.add_parser("detect", help="Illegal-parking detection (+ optional slot overlay)")
    det.add_argument("video")
    det.add_argument("--dwell",      type=float, default=0.5,
                     help="Alert threshold in minutes (default 0.5)")
    det.add_argument("--output", "-o", default=None,
                     help="Write annotated video to this .mp4 path")
    det.add_argument("--no-display", action="store_true",
                     help="Headless — no GUI window")
    det.add_argument("--zone",       default=None,
                     help="Pre-defined zone(s) as JSON: '[[x,y],...]'")
    det.add_argument("--yolo",       action="store_true",
                     help="Use YOLOv8 instead of classical MOG2 detector")
    det.add_argument("--yolo-model", default="yolov8n.pt")
    det.add_argument("--yolo-conf",  type=float, default=0.35)
    det.add_argument("--pos-file",   default="CarParkPos")
    det.add_argument("--no-slots",   action="store_true",
                     help="Disable slot-occupancy overlay even if CarParkPos exists")

    return p


def main() -> None:
    args = build_parser().parse_args()
    dispatch = {"picker": mode_picker, "occupancy": mode_occupancy, "detect": mode_detect}
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()