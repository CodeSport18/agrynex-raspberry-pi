"""Siren and sprinkler outputs.

Each output is on while it's switched on by hand (up to manual_max_s) or for
auto_duration_s after a detection, whichever ends later. An output can also
drive network devices, such as a Shelly relay for the sprinkler, alongside its
GPIO pin.
"""
import threading
import time

from gpiozero import LED

SIREN_PIN = 12
SPRINKLER_PIN = 16
OUTPUT_NAMES = ("siren", "sprinkler")


class Relays:
    def __init__(self, on_change=None, devices=None):
        self.pins = {"siren": LED(SIREN_PIN), "sprinkler": LED(SPRINKLER_PIN)}
        # Extra devices per output, each with set(on, off_at) and close().
        self.devices = devices or {}
        self.manual_until = {name: 0.0 for name in OUTPUT_NAMES}
        self.auto_until = {name: 0.0 for name in OUTPUT_NAMES}
        self.is_on = {name: False for name in OUTPUT_NAMES}
        self.auto_duration_s = 10
        self.manual_max_s = 300
        self.on_change = on_change
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def configure(self, auto_duration_s, manual_max_s):
        with self.lock:
            self.auto_duration_s = auto_duration_s
            self.manual_max_s = manual_max_s
            # Shorten a manual run that is now past the new cap.
            cap = time.time() + manual_max_s
            for name in OUTPUT_NAMES:
                self.manual_until[name] = min(self.manual_until[name], cap)
        self._apply()

    def trigger_auto(self):
        """A detection: run both outputs for auto_duration_s."""
        with self.lock:
            until = time.time() + self.auto_duration_s
            for name in OUTPUT_NAMES:
                self.auto_until[name] = max(self.auto_until[name], until)
        self._apply()

    def set_manual(self, name, on):
        """Switch one output on (up to manual_max_s) or off. Off also ends an automatic run."""
        if name not in OUTPUT_NAMES:
            return
        with self.lock:
            if on:
                self.manual_until[name] = time.time() + self.manual_max_s
            else:
                self.manual_until[name] = 0.0
                self.auto_until[name] = 0.0
        self._apply()

    def stop_auto(self):
        with self.lock:
            for name in OUTPUT_NAMES:
                self.auto_until[name] = 0.0
        self._apply()

    def state(self):
        now = time.time()
        with self.lock:
            out = {}
            for name in OUTPUT_NAMES:
                manual = self.manual_until[name] > now
                auto = self.auto_until[name] > now
                ends = [t for t in (self.manual_until[name], self.auto_until[name]) if t > now]
                out[name] = {
                    "on": manual or auto,
                    "manual": manual,
                    "auto": auto,
                    "off_at": max(ends) if ends else None,
                }
                if self.devices.get(name):
                    # False if any network device for this output can't be reached.
                    out[name]["devices_ok"] = all(d.reachable is not False for d in self.devices[name])
            return out

    def off(self):
        with self.lock:
            for name in OUTPUT_NAMES:
                self.manual_until[name] = self.auto_until[name] = 0.0
        self._apply()

    def close(self):
        """Everything off, including network devices, before the program exits."""
        self.off()
        for devices in self.devices.values():
            for device in devices:
                device.close()

    def _apply(self):
        now = time.time()
        changed = False
        targets = {}
        with self.lock:
            for name in OUTPUT_NAMES:
                want = self.manual_until[name] > now or self.auto_until[name] > now
                if want != self.is_on[name]:
                    self.is_on[name] = want
                    if want:
                        self.pins[name].on()
                    else:
                        self.pins[name].off()
                    changed = True
                ends = [t for t in (self.manual_until[name], self.auto_until[name]) if t > now]
                targets[name] = (want, max(ends) if ends else None)
        # Network devices are told outside the lock; they send in the background.
        for name, (want, off_at) in targets.items():
            for device in self.devices.get(name, []):
                device.set(want, off_at)
        if changed and self.on_change:
            self.on_change()

    def _loop(self):
        # Switches outputs off when their time runs out.
        while True:
            time.sleep(0.2)
            self._apply()
