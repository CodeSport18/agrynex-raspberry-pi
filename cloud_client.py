"""Connection to the Agrynex cloud server.

A Socket.IO link carries settings and commands down and state and live video
frames up. Alerts and their media go over HTTP from a background thread, so a
slow upload never holds up detection.
"""
import queue
import threading
import time

import requests
import socketio

NAMESPACE = "/device"
HEARTBEAT_S = 30
LIVE_INTERVAL_S = 0.2  # About 5 live frames per second
# Keep retrying an alert upload this long before giving up on it.
UPLOAD_GIVE_UP_S = 30 * 60
MAX_QUEUED_UPLOADS = 50


class AlertUpload:
    def __init__(self, detections, detected_at, snapshot, want_clip):
        self.detections = detections
        self.detected_at = detected_at
        self.snapshot = snapshot
        self.want_clip = want_clip
        self.created = time.time()
        self.alert_id = None
        self.uploads = {}
        self.clip = None


class CloudClient:
    def __init__(self, server_url, device_token, on_config, on_command, on_live,
                 get_state, get_live_frame, encode_clip):
        self.server_url = server_url.rstrip("/")
        self.device_token = device_token
        self.on_config = on_config
        self.on_command = on_command
        self.on_live = on_live
        self.get_state = get_state
        self.get_live_frame = get_live_frame
        self.encode_clip = encode_clip
        self.live_on = False
        self.jobs = queue.Queue(maxsize=MAX_QUEUED_UPLOADS)
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {device_token}"

        self.sio = socketio.Client(reconnection=True, reconnection_delay_max=30)
        self.sio.on("connect", self._connected, namespace=NAMESPACE)
        self.sio.on("disconnect", self._disconnected, namespace=NAMESPACE)
        self.sio.on("config", self.on_config, namespace=NAMESPACE)
        self.sio.on("command", self._command, namespace=NAMESPACE)
        self.sio.on("live", self._live, namespace=NAMESPACE)

    def start(self):
        for target in (self._connect_loop, self._heartbeat_loop, self._live_loop, self._upload_loop):
            threading.Thread(target=target, daemon=True).start()

    @property
    def connected(self):
        # sio.connected only turns true after the connect handler has run, so
        # check the namespace instead.
        return NAMESPACE in self.sio.namespaces

    # --- Socket.IO ---

    def _connect_loop(self):
        # socketio.Client reconnects by itself after a drop, but not if the
        # very first connection fails, so keep trying until that works.
        while not self.sio.connected:
            try:
                self.sio.connect(self.server_url, namespaces=[NAMESPACE], transports=["websocket"],
                                 auth={"token": self.device_token}, wait_timeout=15)
            except socketio.exceptions.ConnectionError as e:
                print(f"Cloud: can't connect ({e}), retrying in 15s")
                time.sleep(15)

    def _connected(self):
        print("Cloud: connected")
        self.send_state()

    def _disconnected(self, *_):
        print("Cloud: disconnected")
        self._live({"on": False})

    def _command(self, data):
        if isinstance(data, dict):
            self.on_command(data.get("target"), bool(data.get("on")))

    def _live(self, data):
        self.live_on = bool(isinstance(data, dict) and data.get("on"))
        self.on_live(self.live_on)
        self.send_state()

    def send_state(self):
        if self.connected:
            try:
                self.sio.emit("state", self.get_state(), namespace=NAMESPACE)
            except socketio.exceptions.BadNamespaceError:
                pass

    def _heartbeat_loop(self):
        while True:
            time.sleep(HEARTBEAT_S)
            self.send_state()

    def _live_loop(self):
        last_seq = None
        while True:
            time.sleep(LIVE_INTERVAL_S)
            if not (self.live_on and self.connected):
                continue
            seq, jpeg = self.get_live_frame()
            if jpeg is None or seq == last_seq:
                continue
            last_seq = seq
            try:
                self.sio.emit("frame", jpeg, namespace=NAMESPACE)
            except socketio.exceptions.BadNamespaceError:
                pass

    # --- Alerts ---

    def report_alert(self, detections, detected_at, snapshot, want_clip):
        """Queue a new alert. Returns the upload, to pass to attach_clip later."""
        upload = AlertUpload(detections, detected_at, snapshot, want_clip)
        self._queue(("create", upload, None))
        return upload

    def attach_clip(self, upload, frames):
        self._queue(("clip", upload, frames))

    def _queue(self, job):
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            print("Cloud: upload queue is full, dropping an alert upload")

    def _upload_loop(self):
        while True:
            kind, upload, frames = self.jobs.get()
            try:
                if kind == "create":
                    self._with_retry(upload, lambda: self._create_alert(upload))
                elif upload.alert_id:
                    self._with_retry(upload, lambda: self._upload_clip(upload, frames))
            except GiveUp as e:
                print(f"Cloud: gave up on an alert upload: {e}")
            except Exception as e:
                print(f"Cloud: alert upload failed: {e}")

    def _with_retry(self, upload, step):
        delay = 5
        while True:
            try:
                return step()
            except (requests.ConnectionError, requests.Timeout, ServerError) as e:
                if time.time() - upload.created > UPLOAD_GIVE_UP_S:
                    raise GiveUp(e)
                print(f"Cloud: upload problem ({e}), retrying in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)

    def _post(self, path, body):
        res = self.http.post(f"{self.server_url}{path}", json=body, timeout=20)
        if res.status_code >= 500:
            raise ServerError(f"server returned {res.status_code}")
        res.raise_for_status()
        return res.json()

    def _put_media(self, upload, kind, data):
        target = upload.uploads[kind]
        res = requests.put(target["url"], data=data, headers={"Content-Type": target["content_type"]},
                           timeout=120)
        if res.status_code >= 500:
            raise ServerError(f"storage returned {res.status_code}")
        return res.ok

    def _create_alert(self, upload):
        if upload.alert_id is None:
            data = self._post("/api/device/alerts", {
                "detections": upload.detections,
                "detected_at": upload.detected_at,
                "snapshot": upload.snapshot is not None,
                "clip": upload.want_clip,
            })
            upload.alert_id = data["id"]
            upload.uploads = data["uploads"]
        if upload.snapshot is not None:
            ok = self._put_media(upload, "snapshot", upload.snapshot)
            self._post(f"/api/device/alerts/{upload.alert_id}/media", {"snapshot": "ready" if ok else "failed"})
            upload.snapshot = None

    def _upload_clip(self, upload, frames):
        # Encode once, even if the upload has to be retried.
        if upload.clip is None and frames:
            upload.clip = self.encode_clip(frames) or b""
        ok = bool(upload.clip) and self._put_media(upload, "clip", upload.clip)
        self._post(f"/api/device/alerts/{upload.alert_id}/media", {"clip": "ready" if ok else "failed"})


class ServerError(Exception):
    pass


class GiveUp(Exception):
    pass
