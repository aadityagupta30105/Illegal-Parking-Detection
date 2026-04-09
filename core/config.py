"""
core/config.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

#Detection / Tracking
ILLEGAL_DWELL_MINUTES: float = 0.5
IOU_THRESHOLD:         float = 0.25
MAX_MISSED_FRAMES:     int   = 30
MIN_DETECTION_AREA:    int   = 1500

#MOG2 Background Subtraction
BACKGROUND_HISTORY: int   = 300
BG_THRESHOLD:       float = 40.0

#Slot Picker / Slot Dimensions
SLOT_W: int = 15
SLOT_H: int = 30

#Picker canvas resolution
PICKER_W: int = 1280
PICKER_H: int = 720

#Classical Slot-Occupancy Preprocessing
BLUR_KSIZE:     Tuple[int, int] = (3, 3)
BLUR_SIGMA:     int             = 1
THRESH_BLOCK:   int             = 25
THRESH_C:       int             = 16
MEDIAN_KSIZE:   int             = 5
DILATE_ITER:    int             = 1
FREE_THRESHOLD: int             = 900

#Display
DISPLAY_W: int = 1280
DISPLAY_H: int = 720

#YOLO
VEHICLE_CLASSES = {2, 3, 5, 7}   # COCO: car, motorcycle, bus, truck

#Colours (BGR)
CLR_GREEN  = (50,  205,  50)
CLR_RED    = (0,     0, 220)
CLR_YELLOW = (0,   215, 255)
CLR_ORANGE = (0,   140, 255)
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0,     0,   0)
CLR_CYAN   = (255, 220,   0)

ZONE_PALETTE: List[Tuple[int, int, int]] = [
    (0,   80, 255), (0,  200, 100), (255, 140,   0), (180,   0, 180),
    (0,  220, 220), (255, 80,  80), (30,  180, 255), (100, 255, 150),
]

#Shared Data Structures
@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int

    @property
    def cx(self) -> int: return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int: return (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1); iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2); iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


@dataclass
class Track:
    track_id:           int
    bbox:               BBox
    missed:             int                           = 0
    first_seen_in_zone: Optional[float]               = None
    alert_fired:        bool                          = False
    color:              Tuple[int, int, int]          = field(default_factory=lambda: CLR_GREEN)
    # Velocity estimate (dx, dy per frame) for motion-gated zone entry
    vx:                 float                         = 0.0
    vy:                 float                         = 0.0

    def enter_zone(self) -> None:
        if self.first_seen_in_zone is None:
            self.first_seen_in_zone = time.time()

    def leave_zone(self) -> None:
        self.first_seen_in_zone = None
        self.alert_fired = False

    @property
    def dwell_seconds(self) -> float:
        return (time.time() - self.first_seen_in_zone) if self.first_seen_in_zone else 0.0