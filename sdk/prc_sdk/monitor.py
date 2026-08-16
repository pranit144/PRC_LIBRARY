"""
The main user-facing entrypoint for the prc SDK.

    from prc_sdk import Monitor

    monitor = Monitor(project="image-classifier", run_name="experiment-01")
    for epoch in range(10):
        for batch in train_loader:
            loss = train_step(batch)
            monitor.log(step=step, epoch=epoch, train_loss=float(loss), learning_rate=lr)
    monitor.finish()

Monitoring is fail-safe: any internal error is caught and logged, never
raised into the user's training loop.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .events import (
    Event,
    new_run_id,
    metric_event,
    run_started_event,
    run_finished_event,
    checkpoint_event,
    anomaly_event,
)
from .transport import LocalBuffer, HttpSender
from .live_url import announce_live_url

logger = logging.getLogger("prc")


def _safe(fn):
    """Decorator: never let an SDK call raise into user code."""

    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            logger.exception(f"prc: error in {fn.__name__} (non-fatal, training continues)")
            return None

    wrapper.__name__ = fn.__name__
    return wrapper


class Monitor:
    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        server_url: Optional[str] = None,
        local_dir: str = ".prc",
        run_id: Optional[str] = None,
        show_live_url: bool = True,
    ):
        self.project = project
        self.run_name = run_name or f"run-{int(time.time())}"
        self.config = config or {}
        self.run_id = run_id or new_run_id()
        self._finished = False
        self._start_time = time.time()
        self._last_step = -1
        self._last_epoch = -1

        self.server_url = server_url or os.environ.get("PRC_SERVER_URL", "http://localhost:8000")

        try:
            buffer_path = Path(local_dir) / self.project / f"{self.run_id}.jsonl"
            self._buffer = LocalBuffer(buffer_path)
        except Exception:
            logger.exception("prc: could not initialize local buffer (non-fatal)")
            self._buffer = None

        try:
            self._sender = HttpSender(self.server_url)
        except Exception:
            logger.exception("prc: could not initialize sender (non-fatal)")
            self._sender = None

        self._emit(run_started_event(self.run_id, self.project, self.run_name, self.config))
        logger.info(f"prc: started run '{self.run_id}' ({self.project}/{self.run_name})")

        if show_live_url:
            announce_live_url(self.server_url, self.run_id)

    # -- internal -----------------------------------------------------
    def _emit(self, event: Event) -> None:
        if self._buffer is not None:
            self._buffer.append(event)
        if self._sender is not None:
            self._sender.send(event)

    # -- public API -----------------------------------------------------
    @_safe
    def log(
        self,
        step: int,
        epoch: int = 0,
        **metrics: float,
    ) -> None:
        """Log a set of scalar metrics for a given step/epoch."""
        self._last_step = step
        self._last_epoch = epoch
        self._emit(metric_event(self.run_id, step, epoch, metrics))

    @_safe
    def log_gradient_stats(self, step: int, epoch: int, stats: Dict[str, Any]) -> None:
        self._emit(Event(run_id=self.run_id, event="gradient_stats", step=step, epoch=epoch, data=stats))

    @_safe
    def log_activation_stats(self, step: int, epoch: int, stats: Dict[str, Any]) -> None:
        self._emit(Event(run_id=self.run_id, event="activation_stats", step=step, epoch=epoch, data=stats))

    @_safe
    def log_system_metrics(self, step: int, stats: Dict[str, Any]) -> None:
        self._emit(Event(run_id=self.run_id, event="system_metrics", step=step, data=stats))

    @_safe
    def log_checkpoint(self, step: int, epoch: int, path: str, metrics: Optional[Dict[str, float]] = None) -> None:
        self._emit(checkpoint_event(self.run_id, step, epoch, path, metrics or {}))

    @_safe
    def log_warning(self, message: str, step: Optional[int] = None, epoch: Optional[int] = None) -> None:
        self._emit(Event(run_id=self.run_id, event="warning", step=step, epoch=epoch, data={"message": message}))

    @_safe
    def log_anomaly(self, anomaly_type: str, severity: str, confidence: float, message: str,
                     step: Optional[int] = None, epoch: Optional[int] = None) -> None:
        self._emit(anomaly_event(self.run_id, step or self._last_step, epoch or self._last_epoch,
                                  anomaly_type, severity, confidence, message))

    @_safe
    def hyperparameter_changed(self, name: str, old_value: Any, new_value: Any,
                                step: Optional[int] = None) -> None:
        self._emit(Event(
            run_id=self.run_id, event="hyperparameter_changed", step=step or self._last_step,
            data={"name": name, "old_value": old_value, "new_value": new_value},
        ))

    @_safe
    def epoch_started(self, epoch: int) -> None:
        self._emit(Event(run_id=self.run_id, event="epoch_started", epoch=epoch))

    @_safe
    def epoch_finished(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> None:
        self._emit(Event(run_id=self.run_id, event="epoch_finished", epoch=epoch, data={"metrics": metrics or {}}))

    @_safe
    def finish(self, status: str = "completed") -> None:
        if self._finished:
            return
        self._finished = True
        self._emit(run_finished_event(self.run_id, status))
        if self._sender is not None:
            self._sender.close()
        logger.info(f"prc: finished run '{self.run_id}' ({status})")

    # Support use as a context manager.
    def __enter__(self) -> "Monitor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(status="failed" if exc_type else "completed")
