"""Ties detection to the siren, sprinkler, alert clips and the cloud server."""
import json
import os
import threading
import time
from datetime import datetime, timezone

import cv2

from clip_recorder import ClipRecorder, encode_mp4
from cloud_client import CloudClient
from relays import Relays

# The last settings from the server, so they survive a restart without internet.
SAVED_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_settings.json")

DEFAULT_SETTINGS = {
    "system_on": True,
    "live_view_enabled": True,
    "clips_enabled": True,
    "snapshots_enabled": True,
    "auto_duration_s": 10,
    "manual_max_s": 300,
    "clip_pre_s": 5,
    "clip_post_s": 10,
    "alert_cooldown_s": 30,
}

LIVE_MAX_WIDTH = 640
LIVE_JPEG_QUALITY = 70


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class Controller:
    def __init__(self, server_url=None, device_token=None):
        self.lock = threading.Lock()
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings.update(self._load_saved_settings())
        self.camera = None
        self.last_alert_time = 0.0
        self.last_detection_at = None
        self.cloud = None
        self.relays = Relays(on_change=self.send_state)
        self.recorder = ClipRecorder()
        if server_url and device_token:
            self.cloud = CloudClient(
                server_url, device_token,
                on_config=self.on_config,
                on_command=self.on_command,
                on_live=lambda on: None,
                get_state=self.state,
                get_live_frame=self.get_live_frame,
                encode_clip=encode_mp4,
            )
        else:
            print("Cloud: AGRYNEX_SERVER_URL or AGRYNEX_DEVICE_TOKEN not set, running locally only")
        self._apply_settings()

    def start(self, camera):
        self.camera = camera
        if self.cloud:
            self.cloud.start()

    @property
    def system_on(self):
        return self.settings["system_on"]

    # --- Settings and commands from the server ---

    def on_config(self, config):
        if not isinstance(config, dict):
            return
        with self.lock:
            was_on = self.settings["system_on"]
            for key, default in DEFAULT_SETTINGS.items():
                if key in config and type(config[key]) is type(default):
                    self.settings[key] = config[key]
            turned_off = was_on and not self.settings["system_on"]
            self._save_settings()
        self._apply_settings()
        if turned_off:
            print("System switched off from the cloud")
            self.relays.stop_auto()
            self.recorder.finish_all()
        self.send_state()

    def on_command(self, target, on):
        if target in ("siren", "sprinkler"):
            print(f"Cloud: {target} {'on' if on else 'off'}")
            self.relays.set_manual(target, on)

    def _apply_settings(self):
        s = self.settings
        self.relays.configure(s["auto_duration_s"], s["manual_max_s"])
        self.recorder.configure(s["clip_pre_s"], s["clip_post_s"])

    def _load_saved_settings(self):
        try:
            with open(SAVED_SETTINGS_PATH) as f:
                saved = json.load(f)
            return {k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        try:
            with open(SAVED_SETTINGS_PATH, "w") as f:
                json.dump(self.settings, f)
        except OSError as e:
            print(f"Couldn't save cloud settings: {e}")

    # --- Called from the camera loop ---

    def add_frame(self, frame):
        """Every annotated frame, so alert clips can include the seconds before a detection."""
        if self.cloud and self.settings["clips_enabled"]:
            self.recorder.add_frame(frame)

    def on_detection(self, frame, animal_detections, human_detections):
        """An animal or human was seen and the detection cooldown has passed."""
        self.relays.trigger_auto()
        self.last_detection_at = _now_iso()
        now = time.time()
        if self.cloud and now - self.last_alert_time >= self.settings["alert_cooldown_s"]:
            self.last_alert_time = now
            self._report_alert(frame, animal_detections, human_detections)
        self.send_state()

    def _report_alert(self, frame, animal_detections, human_detections):
        detections = [{"kind": "animal", "label": d["label"], "confidence": d["confidence"]}
                      for d in animal_detections]
        detections += [{"kind": "human", "label": "person", "confidence": d["confidence"]}
                       for d in human_detections]
        snapshot = None
        if self.settings["snapshots_enabled"]:
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            snapshot = jpeg.tobytes() if ok else None
        want_clip = self.settings["clips_enabled"]
        upload = self.cloud.report_alert(detections, self.last_detection_at, snapshot, want_clip)
        if want_clip:
            self.recorder.start(lambda frames: self.cloud.attach_clip(upload, frames))

    # --- State for the server ---

    def state(self):
        return {
            "system_on": self.system_on,
            "camera_ok": bool(self.camera and self.camera.camera_ok),
            "streaming": bool(self.cloud and self.cloud.live_on),
            "last_detection_at": self.last_detection_at,
            **self.relays.state(),
        }

    def send_state(self):
        if self.cloud:
            self.cloud.send_state()

    def get_live_frame(self):
        if not self.camera or not self.system_on or not self.settings["live_view_enabled"]:
            return None, None
        return self.camera.get_live_jpeg(LIVE_MAX_WIDTH, LIVE_JPEG_QUALITY)
