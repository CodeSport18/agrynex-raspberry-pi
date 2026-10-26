"""Keeps the last few seconds of video so an alert clip can start before the detection."""
import collections
import os
import shutil
import subprocess
import tempfile
import threading
import time

import cv2

CLIP_FPS = 10
JPEG_QUALITY = 80


class Recording:
    def __init__(self, frames, end_time, on_done):
        self.frames = frames
        self.end_time = end_time
        self.on_done = on_done


class ClipRecorder:
    def __init__(self):
        self.pre_s = 5
        self.post_s = 10
        self.buffer = collections.deque()
        self.recordings = []
        self.last_frame_time = 0.0
        self.lock = threading.Lock()

    def configure(self, pre_s, post_s):
        with self.lock:
            self.pre_s = pre_s
            self.post_s = post_s

    def add_frame(self, frame):
        """Called for every annotated camera frame. Keeps about CLIP_FPS of them."""
        now = time.time()
        with self.lock:
            if now - self.last_frame_time < 1.0 / CLIP_FPS:
                return
            if self.pre_s <= 0 and not self.recordings:
                self.buffer.clear()
                return
            self.last_frame_time = now
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        item = (now, jpeg.tobytes())
        finished = []
        with self.lock:
            self.buffer.append(item)
            while self.buffer and self.buffer[0][0] < now - self.pre_s:
                self.buffer.popleft()
            for rec in self.recordings:
                rec.frames.append(item)
                if now >= rec.end_time:
                    finished.append(rec)
            self.recordings = [r for r in self.recordings if r not in finished]
        for rec in finished:
            rec.on_done(rec.frames)

    def start(self, on_done):
        """Start a clip from the buffered frames. on_done(frames) runs when it ends."""
        with self.lock:
            rec = Recording(list(self.buffer), time.time() + self.post_s, on_done)
            self.recordings.append(rec)

    def finish_all(self):
        """End clips early, for example when the camera is switched off."""
        with self.lock:
            finished, self.recordings = self.recordings, []
            self.buffer.clear()
        for rec in finished:
            rec.on_done(rec.frames)


def encode_mp4(frames):
    """Turn (timestamp, jpeg) frames into an H.264 MP4 that phones and browsers can play.

    Returns the MP4 bytes, or None if ffmpeg isn't installed or fails.
    """
    if len(frames) < 2 or not shutil.which("ffmpeg"):
        return None
    duration = frames[-1][0] - frames[0][0]
    fps = min(max(len(frames) / duration, 1.0), 30.0) if duration > 0 else CLIP_FPS
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", f"{fps:.2f}", "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast",
             # Even width and height and standard-range yuv420p, which
             # iPhones need. JPEG frames are full range by default.
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_range=tv,format=yuv420p",
             "-movflags", "+faststart", path],
            input=b"".join(jpeg for _, jpeg in frames),
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-500:]}")
            return None
        with open(path, "rb") as f:
            return f.read()
    except subprocess.TimeoutExpired:
        print("ffmpeg timed out")
        return None
    finally:
        os.remove(path)
