"""
yolo_detector.py — Drop-in replacement for VehicleDetector
using YOLOv8 (requires: pip install ultralytics)

Usage in parking_detector.py:
    from yolo_detector import YOLOVehicleDetector as VehicleDetector
"""

from typing import List

try:
    from ultralytics import YOLO
    import numpy as np
    from parking_detector import BBox, MIN_DETECTION_AREA

    # COCO class IDs that represent vehicles
    VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    class YOLOVehicleDetector:
        def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
            self.model = YOLO(model_path)
            self.conf = conf
            print(f"[YOLO] Model loaded: {model_path}")

        def detect(self, frame: np.ndarray) -> List[BBox]:
            results = self.model(frame, verbose=False, conf=self.conf)[0]
            detections: List[BBox] = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = BBox(x1, y1, x2, y2)
                if bbox.area < MIN_DETECTION_AREA:
                    continue
                detections.append(bbox)
            return detections

        def get_fg_mask(self, frame: np.ndarray) -> np.ndarray:
            """Stub — YOLO has no foreground mask."""
            import numpy as np
            return np.zeros(frame.shape[:2], dtype=np.uint8)

except ImportError:
    print("[yolo_detector] ultralytics not installed — YOLOVehicleDetector unavailable.")
    print("  Install with: pip install ultralytics")
