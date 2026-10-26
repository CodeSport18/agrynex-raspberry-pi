# Agrynex Raspberry Pi

Runs on a Raspberry Pi at the farm. It watches the camera for animals (MobileNet-SSD) and humans (YOLOv3). When it detects one, it switches on the siren and the sprinkler.

| GPIO | Device    |
|------|-----------|
| 12   | Siren     |
| 16   | Sprinkler |

The sprinkler can also be switched by a Shelly smart relay (Gen2 or newer, such as the Shelly 1 Gen4) anywhere on the same network, so it doesn't need to be wired to the Pi. Set `AGRYNEX_SHELLY_SPRINKLER_IP` in `.env` to the Shelly's IP address, and give the Shelly a fixed IP on your router. The Pi then switches the Shelly together with GPIO 16 through Shelly's local RPC API. Each "on" command also sets the Shelly's own off-timer to a few seconds after the run should end, so if the Pi crashes or loses Wi-Fi, the sprinkler still turns off. If you protected the Shelly with a password in the Shelly app, also set `AGRYNEX_SHELLY_PASSWORD`.

It connects to the [Agrynex cloud server](https://github.com/CodeSport18/agrynex-cloud-server), so the mobile app can:

- see alerts with a snapshot and a short clip;
- watch live video;
- turn the system on or off;
- switch the siren and sprinkler on or off.

If the internet drops, detection, the siren and the sprinkler keep working on their own. The Pi reconnects when the internet comes back.

## Setup

```bash
sudo apt install ffmpeg          # encodes alert clips
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./download_models.sh
cp .env.example .env             # then fill in the server address and device token
```

`AGRYNEX_DEVICE_TOKEN` in `.env` must match `DEVICE_TOKEN` on the cloud server. Without `.env`, the program runs locally only, as it did before the cloud connection was added.

## Run

```bash
python newer_yolo_detection.py
```

Open `http://<pi-address>:5000` on the farm network to see the local feed and turn animal or human detection on and off. These switches stay in sync with the app. A change on either side shows up on the other straight away, without refreshing: the Pi pushes changes to this page with Server-Sent Events. A change made here while offline is sent to the cloud when the Pi reconnects.

## How it behaves

- **Detection:** when an animal or human is seen, the siren and sprinkler run for the number of seconds set in the app (default 10).
- **Alerts:** each detection sends an alert to the cloud. Alerts are at least 30 seconds apart by default, which you can change in the app's settings. An alert includes a snapshot (at most 1280 px wide) and an H.264 clip covering a few seconds before and after the detection. To keep uploads small, clips are scaled to at most 640 px wide and compressed harder than ffmpeg's default; a 15-second clip is usually under 1 MB.
- **System off:** switching the system off in the app pauses both detection and the camera. The siren and sprinkler can still be switched by hand.
- **Manual control:** a siren or sprinkler switched on from the app turns itself off after the safety limit set in the app (default 5 minutes).
- **Saved settings:** the last settings from the server are saved in `cloud_settings.json`, so a restart without internet keeps them.

## Files

| File | What it does |
|---|---|
| `newer_yolo_detection.py` | Camera loop, detection and the local web viewer |
| `controller.py` | Connects detection to the relays, alert clips and the cloud |
| `relays.py` | Siren and sprinkler timing (automatic runs and manual control with a safety limit) |
| `shelly.py` | Switches a Shelly relay over the local network, with its own off-timer as a safety net |
| `clip_recorder.py` | Keeps recent frames and encodes alert clips with ffmpeg |
| `cloud_client.py` | Socket.IO connection to the server, plus a background queue that uploads alerts |
| `yolo_detection_coyote_video.py` | Older test version that reads `coyote_meets_camera (540p).mp4` and doesn't use GPIO or the cloud |

## Testing on a Mac or PC

The main program runs without a Pi if you use gpiozero's mock pins and a video file:

```bash
GPIOZERO_PIN_FACTORY=mock AGRYNEX_VIDEO_SOURCE="coyote_meets_camera (540p).mp4" python newer_yolo_detection.py
```

The coyote video isn't in the repo, so put a copy in this folder first.
