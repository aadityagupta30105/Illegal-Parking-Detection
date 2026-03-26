# Real-Time Illegal Parking Detection System


## 1. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│   .mov / .mp4 video file  ──►  OpenCV VideoCapture               │
└───────────────────────────────┬──────────────────────────────────┘
                                │ first frame
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│          STEP 1 — MULTI-ZONE ANNOTATION  (one-time)              │
│                                                                  │
│   MultiZoneAnnotator  (interactive OpenCV GUI)                   │
│   • First frame displayed full-screen with controls panel        │
│   • User draws any number of no-parking zone polygons            │
│   • Full undo / redo / delete / reset support                    │
│   • Each zone stored as np.ndarray of (x, y) vertices            │
└───────────────────────────────┬──────────────────────────────────┘
                                │ List[np.ndarray]  (zone polygons)
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│          STEP 2 — VEHICLE DETECTION  (every frame)               │
│                                                                  │
│   VehicleDetector  (classical DIP pipeline)                      │
│     1.  Grayscale  +  Gaussian blur  (7×7)                       │
│     2.  MOG2 background subtraction  →  foreground mask          │
│     3.  Shadow removal  (threshold at 200 on MOG2 output)        │
│     4.  Morphological open → close → dilate  (clean blobs)       │
│     5.  findContours  +  area / aspect-ratio filter              │
│   →  List[BBox]                                                  │
│                                                                  │
│  Optional upgrade: swap in YOLOVehicleDetector (yolo_detector.py)│
└───────────────────────────────┬──────────────────────────────────┘
                                │ detections: List[BBox]
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│          STEP 2 — VEHICLE TRACKING  (every frame)                │
│                                                                  │
│   SortLiteTracker  (IoU-based, Hungarian assignment)             │
│   • Builds cost matrix:  cost[i,j] = 1 − IoU(track_i, det_j)     │
│   • Solves with scipy.optimize.linear_sum_assignment             │
│   • Assigns stable integer IDs across frames                     │
│   • Prunes tracks missing for > MAX_MISSED_FRAMES frames         │
│   →  List[Track]  (each with persistent ID and bbox)             │
└───────────────────────────────┬──────────────────────────────────┘
                                │ tracks: List[Track]
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│          STEP 3 — ZONE CHECK  +  DWELL TIMING                    │
│                                                                  │
│   For every active track:                                        │
│     is_in_any_zone(zone_polys, bbox)  →  cv2.pointPolygonTest    │
│     Inside  →  record wall-clock entry time, accumulate dwell    │
│     Outside →  reset timer                                       │
│                                                                  │
│   AlertManager.check(track)                                      │
│     dwell_seconds ≥ threshold  →  fire ALERT                     │
│       • WARNING written to console + parking_alerts.log          │
│       • Red border + "ALERT!" label on-screen                    │
│       • Dwell progress bar fills green → orange → red            │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│          HUD RENDERER  +  OUTPUT                                 │
│                                                                  │
│   draw_hud():                                                    │
│     • Semi-transparent zone overlays (one colour per zone)       │
│     • Per-vehicle bounding box  +  ID label  +  dwell bar        │
│     • "ALERT!" flash on violating vehicles                       │
│     • Info panel: frame, FPS, zones, active vehicles,            │
│                   in-zone count, alerts fired, threshold         │
│                                                                  │
│   Optional: cv2.VideoWriter  →  annotated .mp4 saved to disk     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Algorithm Workflow

```
for each frame F:

  ┌─ VehicleDetector.detect(F) ──────────────────────────────────┐
  │  gray  ←  grayscale(F)                                       │
  │  blur  ←  GaussianBlur(gray, 7×7)                            │
  │  fg    ←  MOG2.apply(blur)         # updates background model│
  │  mask  ←  threshold(fg, 200)       # removes shadow value 127│
  │  mask  ←  morphOpen  * morphClose * dilate                   │
  │  cnts  ←  findContours(mask)                                 │
  │  dets  ←  [BBox(c) for c in cnts if area > MIN and AR ok]    │
  └──────────────────────────────────────────────────────────────┘
         ↓ dets: List[BBox]

  ┌─ SortLiteTracker.update(dets) ───────────────────────────────┐
  │  cost[i,j]  =  1 − IoU(track_i, det_j)                       │
  │  (r, c)     ←  Hungarian(cost)  if cost < (1 − IOU_THRESH)   │
  │  matched tracks  →  update bbox, reset missed counter        │
  │  unmatched tracks →  increment missed counter                │
  │  unmatched dets   →  create new Track with fresh ID          │
  │  prune  tracks where missed > MAX_MISSED_FRAMES              │
  └──────────────────────────────────────────────────────────────┘
         ↓ tracks: List[Track]

  for t in tracks where t.missed == 0:
      if centroid(t.bbox) inside ANY zone_poly:
          t.enter_zone()          # records time.time() on first entry
          AlertManager.check(t)   # fires if dwell ≥ threshold
      else:
          t.leave_zone()          # resets timer + alert flag

  draw_hud(frame, tracks, alert_mgr, zone_polys, frame_no, fps)
  display frame  /  write to output video
```

---

## 3. Libraries & Models Used

| Component | Library / Model | Purpose |
|-----------|----------------|---------|
| Video I/O | `opencv-python` | Frame capture, display, write |
| Background subtraction | `cv2.createBackgroundSubtractorMOG2` | Foreground mask generation |
| Morphological ops | OpenCV | Mask denoising and gap-filling |
| Contour analysis | OpenCV | Blob extraction → bounding box |
| Hungarian assignment | `scipy.optimize.linear_sum_assignment` | Optimal track ↔ detection matching |
| Tracking | Custom `SortLiteTracker` | Persistent per-vehicle IDs |
| Zone membership | `cv2.pointPolygonTest` | In / out check per zone polygon |
| Annotation GUI | Custom `MultiZoneAnnotator` | Interactive multi-zone drawing |
| *(optional)* YOLO | `ultralytics YOLOv8n` | Higher-accuracy vehicle detection |

---

## 4. Annotation Screen — Full Guide

When the script starts, the first video frame opens in a full-screen annotation window. You draw all no-parking zones here before detection begins.

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  ┌────────────────────────────────────────┐      ┌───────────────┐  │
│  │  NO-PARKING ZONE ANNOTATOR             │      │    STATUS     │  │
│  ├────────────────────────────────────────┤      │ Zones    : 2  │  │
│  │  [Left-Click]   Add vertex             │      │ In-prog  : 0  │  │
│  │  [Enter/Space]  Finish zone            │      │ Redo     : 0  │  │
│  ├ ·  ·  · ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·┤      └───────────────┘  │
│  │  [Z / Ctrl+Z]   Undo                   │                         │
│  │  [Y / Ctrl+Y]   Redo                   │   [ video frame with    │
│  │  [D / Ctrl+D]   Delete nearest zone    │     polygon overlays ]  │
│  │  [R]            Reset in-progress      │                         │
│  ├ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · ·  ·┤                         │
│  │  [F / Esc]      Done → begin detection │                         │
│  └────────────────────────────────────────┘                         │
│                                                                     │
│  ┌──────────────────────┐  ← Zone legend (bottom-left)              │
│  │  ■  Zone 0  (4 pts)  │                                           │
│  │  ■  Zone 1  (6 pts)  │                                           │
│  └──────────────────────┘                                           │
│                                                                     │
│  Click anywhere on the frame to place the first vertex   ← hint     │
└─────────────────────────────────────────────────────────────────────┘
```

### Controls reference

| Key | Action |
|-----|--------|
| **Left-Click** | Place a vertex in the polygon currently being drawn |
| **Enter** or **Space** | Close and save the current polygon as a zone, then start a new one |
| **Z** or **Ctrl+Z** | Undo — removes the last vertex if still drawing, or removes the last completed zone |
| **Y** or **Ctrl+Y** | Redo — restores the last undone zone |
| **D** or **Ctrl+D** | Delete — removes the zone whose centroid is closest to the mouse cursor |
| **R** | Reset — discards the in-progress polygon only (completed zones are untouched) |
| **F** or **Esc** | Done — auto-closes any open polygon and exits to video processing |

### Step-by-step workflow

1. Left-click to place the first vertex anywhere on the frame.
2. Keep clicking to add more vertices. A live rubber-band line follows your cursor from the last point.
3. When your polygon covers the no-parking zone (≥ 3 points), press **Enter** or **Space**. The zone is saved, filled with a colour, and labelled "Zone 0".
4. Start clicking immediately to draw a second zone. Each zone gets a distinct colour from an 8-colour palette.
5. Repeat for as many zones as needed — hospital entrances, bus stops, yellow-box junctions, fire lanes, etc.
6. Mistakes:
   - **Z** removes the last vertex while drawing, or the whole last finished zone.
   - **R** discards the polygon you are currently drawing.
   - Hover over any zone and press **D** to delete it.
   - **Y** restores a zone removed by Z or D.
7. Press **F** or **Esc** when all zones are defined. Detection begins immediately.

### Visual feedback during annotation

- Each zone has its own **colour** — overlapping zones are always distinguishable.
- A **rubber-band line** follows the cursor from the last vertex so you can preview the next edge.
- **White-rimmed dots** mark every vertex of both completed and in-progress polygons.
- The **hover highlight** brightens a zone when your cursor is over it and shows a `[ D ] delete this zone` hint.
- The **Status card** (top-right) shows live counts: zones saved, in-progress vertex count, redo depth.
- The **Zone legend** (bottom-left) lists every saved zone with its colour swatch and vertex count.
- A **first-click hint** appears at the bottom-centre until you start drawing.

---

## 5. How to Run

### Install dependencies

```bash
pip install opencv-python numpy scipy
```

### Basic run — annotation window opens automatically

```bash
python parking_detector.py path/to/video.mov
```

### Set a custom dwell threshold

```bash
# Alert after 2 minutes (useful for testing)
python parking_detector.py video.mov --dwell 2

# Production: alert after 10 minutes
python parking_detector.py video.mov --dwell 10
```

### Save annotated output video

```bash
python parking_detector.py video.mov --output result.mp4
```

### Headless mode — no display window (e.g. on a server)

```bash
python parking_detector.py video.mov --no-display --output result.mp4
```

### Skip the annotation window — provide zones as JSON

Single zone:
```bash
python parking_detector.py video.mov \
  --zone "[[120,100],[560,100],[560,420],[120,420]]"
```

Multiple zones:
```bash
python parking_detector.py video.mov \
  --zone "[[[100,100],[400,100],[400,400],[100,400]],[[500,100],[750,100],[750,350],[500,350]]]"
```

### Full production example

```bash
python parking_detector.py hospital_entrance.mov \
  --dwell 10 \
  --output violations_$(date +%Y%m%d).mp4
```

### Tunable constants — edit the top of `parking_detector.py`

```python
ILLEGAL_DWELL_MINUTES: float = 10.0  # minutes before an alert fires
IOU_THRESHOLD:         float = 0.25  # raise to be stricter about track matching
MAX_MISSED_FRAMES:     int   = 30    # lower to prune lost tracks faster
MIN_DETECTION_AREA:    int   = 1500  # raise to ignore smaller blobs (pedestrians)
BACKGROUND_HISTORY:    int   = 300   # MOG2 frames to build background model
BG_THRESHOLD:          float = 40.0  # MOG2 sensitivity (lower = more sensitive)
```

### Controls during video playback

| Key | Action |
|-----|--------|
| **Q** or **Esc** | Stop processing and exit |

---

## 6. Output

### Console + log file

Every alert is written to both stdout and `parking_alerts.log` in the working directory:

```
2024-01-15 14:32:11  [WARNING]  ILLEGAL PARKING ALERT | Vehicle ID=3 | Dwell=10.2 min | BBox=(220,140,380,290)
```

The log file is overwritten on each new run. To append instead, change `mode="w"` to `mode="a"` in the `logging.FileHandler` call at the top of `parking_detector.py`.

### Annotated video

If `--output` is provided, the processed video is saved as an `.mp4` with all overlays baked in — zone fills, bounding boxes, dwell bars, alert flashes, and the info panel.

### Real-time window

The live window shows:

- **Zone overlays** — each no-parking zone filled with its colour (18 % opacity) and outlined with a 2 px border. The zone label sits at the polygon centroid.
- **Vehicle bounding boxes** — coloured per unique track ID.
- **Dwell progress bar** — a thin bar below each vehicle currently inside a zone. Fills left to right as time accumulates: green (< 50 % of threshold) → orange (50–90 %) → red (> 90 %).
- **ALERT flash** — a thick red border and "ALERT!" label appear on any vehicle that has exceeded the dwell threshold.
- **Info panel** (top-left, semi-transparent):

```
Frame           : 1042
FPS             : 24.8
Zones           : 2
Active vehicles : 7
In no-park zone : 1
Alerts fired    : 0
Threshold       : 10.0 min
```

---

## 7. Optional YOLO Upgrade

The default detector uses MOG2 background subtraction. This works well for stationary cameras but has two known limitations: it needs roughly 300 frames (~12 seconds at 25 fps) to warm up its background model, and it can struggle with stationary vehicles (which blend into the background after a while).

For higher accuracy, swap in YOLOv8:

```bash
pip install ultralytics
```

Then inside `IllegalParkingDetector.__init__` in `parking_detector.py`, replace:

```python
self.detector = VehicleDetector()
```

with:

```python
from yolo_detector import YOLOVehicleDetector
self.detector = YOLOVehicleDetector(model_path="yolov8n.pt", conf=0.35)
```

`yolov8n.pt` (~6 MB) downloads automatically on first use. Everything downstream — tracker, zone check, alert system, HUD — is unchanged.

Detected vehicle classes (COCO IDs): car (2), motorcycle (3), bus (5), truck (7).

---

## 8. Why Parking Spot Counting Was Removed

An earlier version attempted to count available vs occupied parking bays using Hough line detection and texture analysis (pixel variance + Laplacian edge energy). After real-world testing on drone and overhead CCTV footage, the approach proved unreliable for the following reasons:

- **Car roof colour varies widely** — white, silver, dark, red. No single brightness or texture threshold separates all car roofs from empty asphalt across different vehicles.
- **Lighting changes** — sun angle, cloud shadow, and time of day shift both asphalt and car-roof pixel values by ±30 grayscale units across a single recording, breaking any fixed threshold.
- **Drone altitude** — at typical survey heights each bay is only 60–100 px wide. A 3 px painted white line dominates the texture signal for both occupied and empty bays at this scale.
- **Hough line instability** — parking bay lines are short, parallel, and densely packed, producing hundreds of spurious short segments that cluster incorrectly and yield wrong bay boundaries.

The system now focuses exclusively on what classical DIP can do reliably: detecting vehicle movement and tracking dwell time within user-defined restricted zones.

### Reliable alternatives for bay counting

If per-bay occupancy counting is a requirement for your project, the following approaches work correctly:

| Approach | Difficulty | Accuracy |
|----------|-----------|---------|
| Manual bay annotation + background subtraction against a known-empty reference frame | Medium | High |
| YOLOv8 vehicle detection + per-bay IoU overlap check | Medium | High |
| Fine-tuned CNN classifier (PKLot dataset) on pre-segmented bay ROIs | High | Very high |
| Commercial vision API (Google Vision, Azure Computer Vision) | Low | High |

---

## 9. Production Improvements

### Detection accuracy
- Train a custom YOLOv8 model on local CCTV footage covering Vellore traffic conditions and Indian vehicle types.
- Add licence-plate OCR (EasyOCR / PaddleOCR) to log the plate number of each violating vehicle alongside the alert.
- Apply CLAHE illumination normalisation for night-time and rainy conditions.
- Add HSV-based shadow masking to reduce false positives caused by cast shadows on the road surface.

### Tracking robustness
- Upgrade `SortLiteTracker` to **DeepSORT** or **ByteTrack** for re-identification across occlusions and temporary camera blind spots.
- Add a **Kalman filter** for smoother trajectory prediction between detection frames, reducing ID switches.

### Alerts and logging
- Send email / SMS notifications via SMTP / Twilio the moment a violation threshold is crossed.
- Capture and archive a frame snapshot for each violation as photographic evidence for enforcement.
- Write all violations to a **PostgreSQL** database (vehicle ID, zone index, entry time, dwell duration, snapshot path).
- Expose a **FastAPI** REST endpoint so a live traffic management dashboard can poll status and receive alerts.

### Scalability
- Accept **RTSP stream URLs** as input for live CCTV feeds — replace the file path argument with the stream URL.
- Use a **multi-threaded** capture + processing queue (`queue.Queue`) to decouple frame IO from inference latency.
- Run GPU inference via **TensorRT-optimised YOLOv8** for real-time throughput at full 1080p resolution.
- Containerise with **Docker** for reproducible deployment on edge devices (Jetson Nano, Raspberry Pi 5).
- Support **multiple simultaneous cameras** sharing a single zone configuration file loaded at startup.

### Evaluation
- Benchmark detection accuracy against the **PKLot** dataset (Precision / Recall / F1 per camera viewpoint).
- Compare MOG2 vs YOLOv8 alert accuracy under rain, night, headlight glare, and heavy vehicle occlusion.
- Measure false-positive and false-negative alert rates per zone type (hospital entrance, market area, bus stop, fire lane).