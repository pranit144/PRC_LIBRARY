"""
Fail-safe transport layer for the SDK.

Design goal: monitoring must NEVER crash or meaningfully slow down the
user's training process. Every public method swallows and logs its own
errors instead of propagating them.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

import urllib.request
import urllib.error

from .events import Event

logger = logging.getLogger("prc")


class LocalBuffer:
    """Append-only local JSONL buffer. Always succeeds locally, even if
    the server is unreachable, so no training data is lost."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: Event) -> None:
        try:
            with self._lock:
                with open(self.path, "a") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
        except Exception:
            logger.exception("prc: failed to write to local buffer (non-fatal)")

    def read_all(self) -> List[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out


class HttpSender:
    """Best-effort HTTP sender to the prc server. Runs on a background
    thread with a bounded queue so `log()` calls never block training."""

    def __init__(self, server_url: Optional[str], max_queue: int = 10000, timeout: float = 2.0):
        self.server_url = server_url.rstrip("/") if server_url else None
        self.timeout = timeout
        self._queue: "queue.Queue[Event]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self.server_url:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def send(self, event: Event) -> None:
        if not self.server_url:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("prc: send queue full, dropping event (non-fatal)")
        except Exception:
            logger.exception("prc: failed to enqueue event (non-fatal)")

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._post(event)

    def _post(self, event: Event) -> None:
        try:
            body = json.dumps(event.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                f"{self.server_url}/api/events",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception:
            # Server down / unreachable: this is expected and fine. The
            # event is already safe in the local buffer.
            logger.debug("prc: could not reach server (non-fatal)")

    def close(self, flush_timeout: float = 5.0) -> None:
        if not self._thread:
            return
        self._stop.set()
        deadline = time.time() + flush_timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        self._thread.join(timeout=1.0)
