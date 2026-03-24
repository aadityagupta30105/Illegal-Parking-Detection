# Real-Time Illegal Parking Detection System
**Course: BCSE403L — Digital Image Processing**

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                   │
│   .mov / .mp4 video file  ──►  OpenCV VideoCapture              │
└──────────────────────────────┬──────────────────────────────────┘
                               │ first frame
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 1 — ANNOTATION (one-time)                     │
│   PolygonAnnotator (OpenCV GUI)                                  │
│   • Display first frame                                          │
│   • User left-clicks to place polygon vertices                   │
│   • ENTER confirms the no-parking zone                           │
│   • Zone stored as np.ndarray of (x,y) points                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ zone_poly
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 2 — VEHICLE DETECTION (per frame)             │
│   VehicleDetector (classical DIP)                                │
│   1. Grayscale + Gaussian blur                                   │
│   2. MOG2 Background Subtraction  → foreground mask             │
│   3. Shadow removal (thresholding MOG2 shadow value)             │
│   4. Morphological open/close/dilate  → clean blobs             │
│   5. Contour extraction + area / aspect-ratio filter            │
│   → List[BBox]                                                   │
│                                                                  │
│   ── Optional upgrade ──                                         │
│   YOLOVehicleDetector (yolo_detector.py)                         │
│   Replaces step above with YOLOv8n inference                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ detections: List[BBox]
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 2 — VEHICLE TRACKING                          │
│   SortLiteTracker (IoU + Hungarian assignment)                   │
│   • Associates each detection with the closest existing track    │
│   • Assigns persistent integer IDs                               │
│   • Removes tracks missing for > MAX_MISSED_FRAMES frames        │
│   → List[Track]  (each track has stable ID, BBox, history)      │
└──────────────────────────────┬──────────────────────────────────┘
                               │ tracks: List[Track]
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 3 — ZONE CHECK & DWELL TIMING                 │
│   For each active track:                                         │
│   • is_in_zone(zone_poly, bbox)  — point-in-polygon test        │
│   • If inside:  record entry timestamp  → accumulate dwell time │
│   • If outside: reset timer                                      │
│   AlertManager.check(track)                                      │
│   • If dwell_seconds ≥ threshold → fire ALERT (log + visual)    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               PARKING SPOT OCCUPANCY                             │
│   ParkingSpotManager                                             │
│   • Grid of spots defined outside the no-parking zone           │
│   • IoU check between each spot and each active track           │
│   • Shows AVAILABLE / OCCUPIED counts on HUD                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               HUD RENDERER + OUTPUT                              │
│   draw_hud(): zone overlay, bounding boxes, dwell bars,         │
│   alerts, info panel, parking spot grid                          │
│   Optional: cv2.VideoWriter → annotated .mp4 output             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Algorithm Workflow

```
for each frame F:
  ┌─ VehicleDetector.detect(F) ─────────────────────────────────┐
  │  gray ← grayscale(F)                                         │
  │  blur ← GaussianBlur(gray, 7×7)                              │
  │  fg   ← MOG2.apply(blur)          # background model update  │
  │  mask ← threshold(fg, 200)        # remove shadows           │
  │  mask ← morphOpen(mask) ∘ morphClose(mask) ∘ dilate(mask)   │
  │  cnts ← findContours(mask)                                   │
  │  dets ← [BBox(cnt) for cnt in cnts if area>MIN and AR ok]   │
  └──────────────────────────────────────────────────────────────┘
         ↓ dets
  ┌─ SortLiteTracker.update(dets) ──────────────────────────────┐
  │  cost[i,j] = 1 − IoU(track_i, det_j)                       │
  │  matched ← Hungarian(cost)  if cost < (1 − IOU_THRESH)      │
  │  update matched tracks; increment missed for unmatched       │
  │  create new tracks for unmatched detections                  │
  │  prune tracks with missed > MAX_MISSED_FRAMES               │
  └──────────────────────────────────────────────────────────────┘
         ↓ tracks
  for t in tracks:
    if centroid(t.bbox) inside zone_poly:
      t.enter_zone()           # record time.time() if first entry
      if t.dwell_sec ≥ threshold:
        FIRE ALERT (log + visual)
    else:
      t.leave_zone()           # reset timer

  ParkingSpotManager.update(tracks)
  draw_hud(frame, ...)
  display / write frame
```

---

## 3. Libraries & Models Used

| Component | Library / Model | Purpose |
|-----------|----------------|---------|
| Video I/O | `opencv-python` | Frame capture, display, write |
| Background subtraction | `cv2.createBackgroundSubtractorMOG2` | Foreground detection |
| Morphological ops | OpenCV | Mask cleanup |
| Contour analysis | OpenCV | Blob → bounding box |
| Hungarian assignment | `scipy.optimize.linear_sum_assignment` | Optimal track↔detection matching |
| Tracking | Custom `SortLiteTracker` (SORT-lite) | Persistent vehicle IDs |
| Zone test | `cv2.pointPolygonTest` | In/out zone check |
| Polygon annotation | Custom `PolygonAnnotator` (OpenCV GUI) | Interactive ROI drawing |
| *(optional)* YOLO | `ultralytics YOLOv8n` | Higher-accuracy detection |

---

## 4. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Basic run (opens annotation window)
```bash
python parking_detector.py path/to/your_video.mov
```

### Set dwell threshold to 2 minutes (for testing)
```bash
python parking_detector.py video.mov --dwell 2
```

### Save annotated output video
```bash
python parking_detector.py video.mov --output output_annotated.mp4
```

### Headless mode (no display window, e.g. server)
```bash
python parking_detector.py video.mov --no-display --output result.mp4
```

### Pre-define zone via JSON (skip annotation window)
```bash
python parking_detector.py video.mov \
  --zone "[[100,150],[500,150],[500,450],[100,450]]"
```

### Full example
```bash
python parking_detector.py hospital_entrance.mov \
  --dwell 10 \
  --output detected.mp4
```

### Change global variables in code
Edit the top of `parking_detector.py`:
```python
ILLEGAL_DWELL_MINUTES: float = 10.0   # ← alert threshold
MIN_DETECTION_AREA: int = 1500        # ← min blob size in px²
IOU_THRESHOLD: float = 0.25           # ← tracker sensitivity
MAX_MISSED_FRAMES: int = 30           # ← track persistence
```

### Interactive controls (while video plays)
| Key | Action |
|-----|--------|
| `q` or `ESC` | Stop processing |

---

## 5. Annotation Instructions

When the annotation window opens:
1. **Left-click** to place polygon vertices around the no-parking zone
2. The polygon preview fills in red as you add points
3. Press **ENTER** or **SPACE** to confirm
4. Press **r** to reset and start over
5. Press **ESC** to cancel (a default centre rectangle is used)

---

## 6. Output

- **Console / `parking_alerts.log`** — timestamped alert messages:
  ```
  2024-01-15 14:32:11  [WARNING]  🚨 ILLEGAL PARKING ALERT | Vehicle ID=3 | Dwell=10.2 min | ...
  ```
- **Annotated video** (if `--output` specified) — MP4 with all overlays
- **Real-time window** — live detection, tracking, and alerts

---

## 7. Production Improvements

### Detection
- Replace MOG2 with **YOLOv8** (`yolo_detector.py`) for licence-plate-level precision
- Add **licence plate OCR** (EasyOCR / PaddleOCR) to identify specific vehicles
- Train a custom YOLO model on local CCTV datasets (e.g. PKLot, Vellore footage)

### Tracking
- Upgrade to **DeepSORT** or **ByteTrack** for robust re-identification across occlusion
- Add **Kalman filter** for smoother trajectory prediction

### Robustness
- Adaptive illumination normalisation (CLAHE) for night/rain conditions
- Shadow removal via colour space (HSV) analysis
- Multi-camera synchronisation for coverage gaps

### Alerts
- Email / SMS notification via SMTP / Twilio when alert fires
- REST API endpoint (FastAPI) to push alerts to a traffic dashboard
- Database (PostgreSQL) logging of all violations with frame snapshots

### Scalability
- GPU inference pipeline (TensorRT-optimised YOLO)
- Multi-threaded frame capture + processing queue
- Docker containerisation for deployment on edge devices (Jetson Nano)
- RTSP stream input for live CCTV feeds

### Evaluation
- Benchmark with PKLot dataset ground truth
- Report Precision / Recall / F1 per parking spot
- Compare classical DIP vs YOLO accuracy under rain / night conditions
