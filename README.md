# Unified Parking System

Real-time parking enforcement using OpenCV, MOG2/YOLOv8, and a Kalman-SORT tracker.

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Mark slot positions on first video frame
python main.py picker clip.mp4

# 2. (Optional) test slot occupancy only
python main.py occupancy clip.mp4

# 3. Full detection pipeline (opens zone-annotation window first)
python main.py detect clip.mp4

# 4. GUI (no CLI needed)
python gui.py
```

---

## Step-by-step Operation

| Step | Command | What happens |
|------|---------|--------------|
| 1 | `python main.py picker clip.mp4` | Opens a 1280×720 window showing the first frame. Left-click to place slot boxes, right-click to remove, Q to save. Positions saved to `CarParkPos`. |
| 2 | `python main.py detect clip.mp4` | Opens zone-annotation window. Draw no-parking polygons with left-click, press F/ESC when done. Detection + tracking starts immediately after. |
| 3 | `python main.py occupancy clip.mp4` | Runs slot-occupancy overlay only (no vehicle tracking). |
| 4 | `python gui.py` | Opens the Tkinter dashboard — same options, no CLI required. |

---

## Zone Annotator Controls

| Key | Action |
|-----|--------|
| Left-click | Add vertex |
| Enter / Space | Finish current polygon |
| Z | Undo last vertex / last zone |
| Y | Redo |
| D | Delete zone nearest cursor |
| R | Reset in-progress polygon |
| F / ESC | Done — start detection |

---

## CLI Reference — `main.py`

### `picker` mode
| Argument | Default | Description |
|----------|---------|-------------|
| `video` | *(required)* | Path to input video |
| `--pos-file` | `CarParkPos` | Output pickle for slot positions |
| `--picker-img` | `clipimage.png` | Background image (auto-extracted from video if missing) |

### `occupancy` mode
| Argument | Default | Description |
|----------|---------|-------------|
| `video` | *(required)* | Path to input video |
| `--pos-file` | `CarParkPos` | Slot positions pickle |

### `detect` mode
| Argument | Default | Description |
|----------|---------|-------------|
| `video` | *(required)* | Path to input video |
| `--dwell` | `0.5` | Minutes a vehicle must stay in a no-park zone before an alert fires |
| `--output`, `-o` | *(none)* | Write annotated video to this `.mp4` path |
| `--no-display` | off | Headless mode — no GUI window (for servers) |
| `--zone` | *(none)* | Pre-define zone(s) as JSON, skipping the annotation window. Single zone: `'[[x,y],[x,y],[x,y]]'` — multiple: `'[[[x,y],...],[[x,y],...]]'` |
| `--yolo` | off | Use YOLOv8 detector instead of classical MOG2 |
| `--yolo-model` | `yolov8n.pt` | Path to YOLOv8 weights |
| `--yolo-conf` | `0.35` | YOLO confidence threshold (0–1) |
| `--pos-file` | `CarParkPos` | Slot positions pickle for occupancy overlay |
| `--no-slots` | off | Disable slot-occupancy overlay even when `CarParkPos` exists |

---

## Project Structure

```
.
├── main.py                  CLI entry point
├── gui.py                   Tkinter GUI dashboard
├── requirements.txt
│
├── core/
│   ├── config.py            All global parameters + BBox/Track dataclasses
│   ├── illegal_parking.py   Main detection pipeline + HUD renderer
│   ├── slot_occupancy.py    Slot-occupancy detection + draw_slots()
│   └── video_utils.py       VideoCapture / VideoWriter helpers
│
├── detectors/
│   ├── classical.py         MOG2 background subtraction detector
│   └── yolo.py              YOLOv8 detector (optional)
│
├── trackers/
│   └── sort_lite.py         Kalman-SORT + ByteTrack-style two-stage matcher
│
└── ui/
    ├── _hud_utils.py        Shared semi-transparent HUD drawing helpers
    ├── zone_annotator.py    Interactive polygon zone annotator
    └── slot_picker.py       Interactive rectangular slot picker
```

---

## Key Configuration (`core/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ILLEGAL_DWELL_MINUTES` | `0.5` | Default alert threshold |
| `IOU_THRESHOLD` | `0.25` | Min IoU for track-detection match |
| `MAX_MISSED_FRAMES` | `30` | Frames before a stale track is deleted |
| `MIN_DETECTION_AREA` | `1500` | Minimum blob area (px²) to consider |
| `PICKER_W / PICKER_H` | `1280 × 720` | Picker canvas resolution |
| `SLOT_W / SLOT_H` | `60 × 120` | Slot rectangle size in picker space |
| `FREE_THRESHOLD` | `900` | Non-zero pixel count below which a slot is free |

---

## Tracker Notes

`trackers/sort_lite.py` implements **Kalman-SORT with ByteTrack-style two-stage matching**:

1. **Predict** — every active track's Kalman filter advances one time step, giving a smooth predicted bbox even under occlusion.
2. **High-confidence match** (IoU ≥ 0.30) — Hungarian assignment between predicted tracks and all detections.
3. **Low-confidence match** (IoU ≥ 0.15) — second pass on remaining unmatched tracks vs. remaining detections (recovers partially-occluded vehicles).
4. **Confirmation gate** — a new track must be matched in ≥ 2 consecutive frames before it appears on the HUD, suppressing flicker from false detections.
