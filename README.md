# Real-Time Illegal Parking Detection System
---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                  │
│   .mov / .mp4 video file  ──►  OpenCV VideoCapture              │
└──────────────────────────────┬──────────────────────────────────┘
                               │ first frame
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 1 — MULTI-ZONE ANNOTATION (one-time)             │
│   MultiZoneAnnotator (OpenCV GUI)                               │
│   • Displays first frame with interactive controls panel        │
│   • User draws multiple polygon no-parking zones                │
│   • Full undo / redo / delete support                           │
│   • Each zone stored as np.ndarray of (x,y) points              │
└──────────────────────────────┬──────────────────────────────────┘
                               │ List[np.ndarray]
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 2 — VEHICLE DETECTION (per frame)            │
│   VehicleDetector (classical DIP pipeline)                      │
│   1. Grayscale + Gaussian blur                                  │
│   2. MOG2 Background Subtraction  → foreground mask             │
│   3. Shadow removal (thresholding MOG2 shadow value 127)        │
│   4. Morphological open / close / dilate  → clean blobs         │
│   5. Contour extraction + area / aspect-ratio filter            │
│   → List[BBox]                                                  │
│                                                                 │
│   ── Optional upgrade ──                                        │
│   YOLOVehicleDetector (yolo_detector.py)                        │
│   Replaces step above with YOLOv8n inference                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ detections: List[BBox]
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 2 — VEHICLE TRACKING                         │
│   SortLiteTracker (IoU + Hungarian assignment)                  │
│   • Associates each detection with closest existing track       │
│   • Assigns persistent integer IDs                              │
│   • Removes tracks missing for > MAX_MISSED_FRAMES frames       │
│   → List[Track]                                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │ tracks: List[Track]
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 3 — ZONE CHECK & DWELL TIMING                │
│   For each active track:                                        │
│   • is_in_any_zone(zone_polys, bbox)  — point-in-polygon test   │
│   • If inside any zone: record entry time, accumulate dwell     │
│   • If outside all zones: reset timer                           │
│   AlertManager.check(track)                                     │
│   • dwell_seconds ≥ threshold → fire ALERT (log + visual)       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               PARKING SPOT OCCUPANCY                            │
│   ParkingSpotManager                                            │
│   • Grid of spots defined outside ALL no-parking zones          │
│   • IoU check between each spot and each active track           │
│   • Shows AVAILABLE / OCCUPIED counts on HUD                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               HUD RENDERER + OUTPUT                             │
│   draw_hud(): per-zone coloured overlays, bounding boxes,       │
│   dwell progress bars, alert flashes, info panel, spot grid     │
│   Optional: cv2.VideoWriter → annotated .mp4 output             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Algorithm Workflow

```
for each frame F:
  ┌─ VehicleDetector.detect(F) ─────────────────────────────────┐
  │  gray ← grayscale(F)                                        │
  │  blur ← GaussianBlur(gray, 7×7)                             │
  │  fg   ← MOG2.apply(blur)          # background model update │
  │  mask ← threshold(fg, 200)        # remove shadows (127→0)  │
  │  mask ← morphOpen * morphClose * dilate                     │
  │  cnts ← findContours(mask)                                  │
  │  dets ← [BBox(cnt) for cnt in cnts if area>MIN and AR ok]   │
  └─────────────────────────────────────────────────────────────┘
         ↓ dets
  ┌─ SortLiteTracker.update(dets) ──────────────────────────────┐
  │  cost[i,j] = 1 − IoU(track_i, det_j)                        │
  │  matched ← Hungarian(cost) where cost < (1 − IOU_THRESH)    │
  │  update matched; increment missed for unmatched             │
  │  create new tracks for unmatched detections                 │
  │  prune tracks with missed > MAX_MISSED_FRAMES               │
  └─────────────────────────────────────────────────────────────┘
         ↓ tracks
  for t in tracks:
    if centroid(t.bbox) inside ANY zone_poly:
      t.enter_zone()           # record time.time() on first entry
      if t.dwell_sec ≥ threshold:
        FIRE ALERT (log + red flash + "ALERT!" label)
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
| Background subtraction | `cv2.createBackgroundSubtractorMOG2` | Foreground mask |
| Morphological ops | OpenCV | Mask cleanup |
| Contour analysis | OpenCV | Blob → bounding box |
| Hungarian assignment | `scipy.optimize.linear_sum_assignment` | Optimal track↔detection matching |
| Tracking | Custom `SortLiteTracker` | Persistent vehicle IDs |
| Zone test | `cv2.pointPolygonTest` | In/out zone check (per polygon) |
| Annotation GUI | Custom `MultiZoneAnnotator` (OpenCV) | Multi-zone interactive drawing |
| *(optional)* YOLO | `ultralytics YOLOv8n` | Higher-accuracy detection |

---

## 4. Annotation Screen — Full Guide

When you run the script, the **first video frame** opens in a fullscreen annotation window. This is where you define all your no-parking zones before detection begins.

### What you see

```
┌──────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────┐          ┌───────────┐ │
│  │ NO-PARKING ZONE ANNOTATOR    │          │  STATUS   │ │
│  ├──────────────────────────────┤          │ Zones : 1 │ │
│  │ [Left-Click]  Add vertex     │          │ Pts   : 3 │ │
│  │ [Enter/Space] Finish zone    │          │ Redo  : 0 │ │
│  ├ · · · · · · · · · · · · · · ·┤          └───────────┘ │
│  │ [Z/Ctrl+Z]    Undo           │                        │
│  │ [Y/Ctrl+Y]    Redo           │                        │
│  │ [D/Ctrl+D]    Delete zone    │   [video frame with    │
│  │ [R]           Reset current  │    polygon overlays]   │
│  ├ · · · · · · · · · · · · · · ·┤                        │
│  │ [F / Esc]     Done           │                        │
│  └──────────────────────────────┘                        │
│                                                          │
│  ┌─────────────────┐   ← zone legend (bottom-left)       │
│  │ ■ Zone 0  (4pts)│                                     │
│  │ ■ Zone 1  (6pts)│                                     │
│  └─────────────────┘                                     │
└──────────────────────────────────────────────────────────┘
```

### Controls reference

| Key | Action |
|-----|--------|
| **Left-Click** | Place a vertex in the current polygon being drawn |
| **Enter** or **Space** | Close the current polygon → save it as a zone, start a new one |
| **Z** or **Ctrl+Z** | **Undo** — removes the last placed vertex if still drawing, or removes the last completed zone |
| **Y** or **Ctrl+Y** | **Redo** — restores the most recently undone zone |
| **D** or **Ctrl+D** | **Delete** — removes the zone whose centroid is closest to the mouse cursor |
| **R** | **Reset** — discards the polygon currently being drawn (completed zones are unaffected) |
| **F** or **Esc** | **Done** — auto-closes any open polygon and exits to video detection |

### Step-by-step workflow

1. **Click** to place the first vertex of a no-parking zone on the frame.
2. Continue clicking to add more vertices. A live rubber-band line shows where the next edge will go.
3. Once you have ≥ 3 points and the shape looks right, press **Enter** or **Space** to finalise the zone. It fills with a semi-transparent colour and gets a "Zone 0" label.
4. Immediately start clicking to draw the next zone — each one gets a different colour automatically.
5. Repeat for as many zones as needed.
6. If you make a mistake:
   - Press **Z** to undo the last vertex (while drawing), or to remove the last finished zone.
   - Press **R** to throw away the current in-progress polygon entirely.
   - Hover your mouse over an existing zone and press **D** to delete it.
7. When all zones are defined, press **F** or **Esc** to begin video processing.

### Visual feedback

- **Colour-coded zones** — each zone has its own colour from an 8-colour palette so overlapping zones are easy to distinguish.
- **Rubber-band line** — a dashed line follows your cursor from the last placed vertex, showing exactly where the next edge will go.
- **Vertex dots** — white-rimmed circles at each vertex of completed and in-progress polygons.
- **Hover-to-delete** — hovering the mouse over a completed zone highlights it brighter and shows a `[D] delete` hint at its label.
- **Status card** (top-right) — live counts of saved zones, current in-progress vertices, and available redo steps.
- **Zone legend** (bottom-left) — lists every saved zone with its colour swatch and vertex count.

---

## 5. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Basic run (annotation window opens)
```bash
python parking_detector.py path/to/video.mov
```

### Set dwell threshold (minutes)
```bash
python parking_detector.py video.mov --dwell 2
```

### Save annotated output video
```bash
python parking_detector.py video.mov --output result.mp4
```

### Headless / server mode (no display window)
```bash
python parking_detector.py video.mov --no-display --output result.mp4
```

### Pre-define a single zone (skip annotation)
```bash
python parking_detector.py video.mov \
  --zone "[[100,150],[500,150],[500,450],[100,450]]"
```

### Pre-define multiple zones (skip annotation)
```bash
python parking_detector.py video.mov \
  --zone "[[[100,100],[400,100],[400,400],[100,400]],[[500,100],[750,100],[750,350],[500,350]]]"
```

### Full example
```bash
python parking_detector.py hospital_entrance.mov \
  --dwell 10 \
  --output detected.mp4
```

### Tuning global variables
Edit the top of `parking_detector.py`:
```python
ILLEGAL_DWELL_MINUTES: float = 10.0   # alert threshold in minutes
MIN_DETECTION_AREA:    int   = 1500   # min blob area in pixels²
IOU_THRESHOLD:         float = 0.25   # tracker sensitivity
MAX_MISSED_FRAMES:     int   = 30     # frames before a track is pruned
PARKING_SPOT_GRID_ROWS: int  = 2      # rows of legal parking spots
PARKING_SPOT_GRID_COLS: int  = 5      # columns of legal parking spots
```

### Controls during video playback
| Key | Action |
|-----|--------|
| `q` or `Esc` | Stop processing |

---

## 6. Output

- **Console + `parking_alerts.log`** — timestamped alert entries:
  ```
  2024-01-15 14:32:11  [WARNING]  🚨 ILLEGAL PARKING ALERT  |  Vehicle ID=3  |  Dwell=10.2 min  |  Threshold=10.0 min  |  BBox=(220,140,380,290)
  ```
- **Annotated video** (if `--output` given) — MP4 with all overlays baked in.
- **Real-time window** — live feed with zone overlays, bounding boxes, dwell bars, and parking grid.

---

## 7. Optional YOLO Upgrade

For higher detection accuracy (especially in complex or crowded scenes), replace the classical DIP detector with YOLOv8:

```bash
pip install ultralytics
```

Then in `parking_detector.py`, change the import at the top of `IllegalParkingDetector.__init__`:

```python
# Replace this line:
self.detector = VehicleDetector()

# With this:
from yolo_detector import YOLOVehicleDetector
self.detector = YOLOVehicleDetector(model_path="yolov8n.pt", conf=0.35)
```

YOLOv8n downloads automatically on first use (~6 MB). The rest of the pipeline is unchanged.

---

## 8. Production Improvements

**Detection accuracy**
- Train a custom YOLOv8 model on local CCTV data (Vellore traffic conditions).
- Add licence-plate OCR with EasyOCR / PaddleOCR to log offending plate numbers.
- Apply CLAHE illumination normalisation for night and rainy conditions.
- Add HSV-based shadow suppression for parking lots with strong sunlight.

**Tracking robustness**
- Upgrade to DeepSORT or ByteTrack for re-identification across occlusions.
- Add a Kalman filter for smoother trajectory prediction and better tracking at low frame rates.

**Alerts & logging**
- Push email / SMS alerts via SMTP / Twilio when a violation is confirmed.
- Save a frame snapshot at the moment of each alert.
- Write violations to a PostgreSQL database (vehicle ID, zone, timestamp, snapshot path).
- Expose a FastAPI REST endpoint for integration with a traffic management dashboard.

**Scalability**
- Accept RTSP streams as input for live CCTV feeds.
- Multi-thread frame capture + processing using a queue for higher throughput.
- GPU inference via TensorRT-optimised YOLOv8 for real-time processing at 1080p.
- Docker containerisation for deployment on edge devices (Jetson Nano / Raspberry Pi).
- Multi-camera support with shared zone configuration.

**Evaluation**
- Benchmark against the PKLot dataset for detection accuracy (Precision / Recall / F1).
- Compare classical DIP vs YOLO under rain, night, and glare conditions.
- Measure false-positive rate per zone type (hospital entrance vs market area).
