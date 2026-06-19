# Roshdi Object Detection — Local Camera Proximity Test

This package adds a local webcam test for the Roshdi Object Detection module.
It uses YOLO detections and converts them into voice-friendly proximity warnings.

The original GitHub Object Detection module uses YOLO11n pretrained on Microsoft COCO 80 classes and already contains spatial output fields such as `horizontal_position` and `distance_hint`. This local test package extends that idea with:

- 5 FPS processing control
- confidence filtering
- group-based proximity thresholds
- near / very near warning logic
- left / center / right position detection
- temporal smoothing across the last 3 processed frames
- 2-second cooldown for repeated warnings
- optional text-to-speech

## Files

```text
proximity_warning.py       # Core threshold/smoothing/cooldown logic
live_camera_proximity.py   # Webcam + YOLO + visual overlay + optional speech
requirements.txt           # Python dependencies
run_camera.bat             # Windows quick run
run_camera.sh              # Linux/macOS quick run
```

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Run with local webcam

```bash
python live_camera_proximity.py
```

Or specify camera/model/settings:

```bash
python live_camera_proximity.py --camera 0 --weights yolo11n.pt --conf 0.40 --process-fps 5
```

Enable speech:

```bash
python live_camera_proximity.py --speak --lang en
```

Arabic message field:

```bash
python live_camera_proximity.py --speak --lang ar
```

Print the structured output JSON:

```bash
python live_camera_proximity.py --print-json
```

Press `Q` in the camera window to stop.

## Important Notes

This system does not estimate real distance in meters. It estimates visual proximity using normalized bounding-box size:

```text
area_ratio = bbox_area / frame_area
height_ratio = bbox_height / frame_height
```

The threshold values are starting points. You should tune them using real camera tests.
For example, place a chair at 0.5m, 1m, and 2m, then check the printed `area_ratio` and adjust the thresholds in `proximity_warning.py`.

## Threshold Design

The module does not use one fixed threshold for all 80 COCO classes.
It uses groups:

- large danger objects: person, car, bus, truck, motorcycle, bicycle
- medium obstacles: chair, bench, couch, dining table, suitcase, backpack
- small objects: bottle, cup, cell phone, book, remote, mouse, keyboard
- default group for all other classes

This is better than one global threshold because a car, chair, person, and bottle have very different real-world sizes.

## Copy Into Your Existing Repo

Copy these files into your existing `Object Detection/` folder:

```text
proximity_warning.py
live_camera_proximity.py
requirements.txt
run_camera.bat
run_camera.sh
README_LOCAL_TEST.md
```

Then run:

```bash
python live_camera_proximity.py --weights yolo11n.pt
```

If you have a custom model:

```bash
python live_camera_proximity.py --weights path/to/best.pt
```
