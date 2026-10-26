# Agrynex Raspberry Pi

Runs on a Raspberry Pi at the farm. It watches the camera for animals (MobileNet-SSD) and humans (YOLOv3). When it detects one, it switches on the siren and the sprinkler.

| GPIO | Device    |
|------|-----------|
| 12   | Siren     |
| 16   | Sprinkler |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./download_models.sh
```

## Run

```bash
python newer_yolo_detection.py
```

Open `http://<pi-address>:5000` to see the live feed and turn animal or human detection on and off.

`yolo_detection_coyote_video.py` is a test version that reads `coyote_meets_camera (540p).mp4` instead of the camera and doesn't use GPIO. The video isn't in the repo, so put a copy in this folder before running it.
