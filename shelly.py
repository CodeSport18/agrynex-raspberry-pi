"""Shelly smart relay (Gen2 or newer, such as the Shelly 1 Gen4) on the local network.

The Pi switches it through Shelly's local HTTP API whenever the matching GPIO
output changes. Each "on" command also tells the Shelly to switch itself off a
few seconds after the run should end, so a crashed or disconnected Pi can't
leave the sprinkler running.
"""
import threading
import time

import requests
from requests.auth import HTTPDigestAuth

# The Shelly's own off-timer fires this long after the Pi's planned off time,
# so it only matters if the Pi's off command never arrives.
SAFETY_MARGIN_S = 5
# While on, the "on" command is repeated this often in case the Shelly restarted.
REFRESH_S = 30
RETRY_S = 3
TIMEOUT_S = 3


class ShellySwitch:
    def __init__(self, name, ip, password=None, switch_id=0):
        self.name = name
        self.url = f"http://{ip}/rpc/Switch.Set"
        self.switch_id = switch_id
        # Shelly's web login uses the fixed user name "admin".
        self.auth = HTTPDigestAuth("admin", password) if password else None
        self.lock = threading.Lock()
        self.want_on = False
        self.off_at = None
        self.sent = None  # (on, off_at) the Shelly last accepted
        self.sent_at = 0.0
        self.reachable = None
        self.wake = threading.Event()
        # Starts by sending "off", the same state the GPIO pins start in.
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, on, off_at=None):
        """Called every time the relays are checked. Only real changes wake the sender."""
        with self.lock:
            same_end = off_at is None or (self.off_at is not None and abs(off_at - self.off_at) < 1)
            if on == self.want_on and (not on or same_end):
                return
            self.want_on, self.off_at = on, off_at
        self.wake.set()

    def close(self):
        """Switch off right away, for when the program exits."""
        try:
            self._send(False, None)
        except requests.RequestException as e:
            print(f"Shelly {self.name}: couldn't switch off on exit ({e})")

    def _send(self, on, off_at):
        params = {"id": self.switch_id, "on": "true" if on else "false"}
        if on and off_at:
            params["toggle_after"] = max(1, round(off_at - time.time() + SAFETY_MARGIN_S))
        res = requests.get(self.url, params=params, auth=self.auth, timeout=TIMEOUT_S)
        res.raise_for_status()

    def _loop(self):
        while True:
            with self.lock:
                on, off_at = self.want_on, self.off_at
                due = self.sent != (on, off_at) or (on and time.time() - self.sent_at >= REFRESH_S)
            if due:
                try:
                    self._send(on, off_at)
                except requests.RequestException as e:
                    if self.reachable is not False:
                        print(f"Shelly {self.name}: can't reach {self.url} ({e}), retrying")
                    self.reachable = False
                    self.wake.wait(RETRY_S)
                    self.wake.clear()
                    continue
                with self.lock:
                    self.sent, self.sent_at = (on, off_at), time.time()
                if not self.reachable:
                    print(f"Shelly {self.name}: connected")
                self.reachable = True
            self.wake.wait(REFRESH_S if on else None)
            self.wake.clear()
