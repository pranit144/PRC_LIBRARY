"""
prc event protocol.

This is the single source of truth for what an "event" looks like as it
flows: training script -> SDK -> local buffer -> server -> storage -> dashboard.

The schema is intentionally framework-independent and versioned so it can
evolve without breaking older clients.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1

# The full set of event types the protocol currently understands.
EVENT_TYPES = {
    "run_started",
    "run_finished",
    "epoch_started",
    "epoch_finished",
    "metric",
    "gradient_stats",
    "activation_stats",
    "checkpoint",
    "warning",
    "anomaly",
    "hyperparameter_changed",
    "forecast_updated",
    "system_metrics",
}


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Event:
    """A single, self-contained training event."""

    run_id: str
    event: str
    schema_version: int = SCHEMA_VERSION
    timestamp: str = field(default_factory=now_iso)
    step: Optional[int] = None
    epoch: Optional[int] = None
    # Free-form payload whose shape depends on `event` (metrics dict,
    # gradient stats dict, checkpoint info, message string, etc).
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.event not in EVENT_TYPES:
            raise ValueError(
                f"Unknown event type '{self.event}'. Must be one of {sorted(EVENT_TYPES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        d = dict(d)
        # Tolerate events from older/newer schema versions by dropping
        # unknown top-level keys rather than crashing.
        known = {f for f in cls.__dataclass_fields__.keys()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


def metric_event(run_id: str, step: int, epoch: int, metrics: Dict[str, float]) -> Event:
    return Event(run_id=run_id, event="metric", step=step, epoch=epoch, data={"metrics": metrics})


def run_started_event(run_id: str, project: str, run_name: str, config: Dict[str, Any]) -> Event:
    return Event(
        run_id=run_id,
        event="run_started",
        data={"project": project, "run_name": run_name, "config": config},
    )


def run_finished_event(run_id: str, status: str = "completed") -> Event:
    return Event(run_id=run_id, event="run_finished", data={"status": status})


def checkpoint_event(run_id: str, step: int, epoch: int, path: str, metrics: Dict[str, float]) -> Event:
    return Event(
        run_id=run_id,
        event="checkpoint",
        step=step,
        epoch=epoch,
        data={"path": path, "metrics": metrics},
    )


def anomaly_event(run_id: str, step: int, epoch: int, anomaly_type: str,
                   severity: str, confidence: float, message: str) -> Event:
    return Event(
        run_id=run_id,
        event="anomaly",
        step=step,
        epoch=epoch,
        data={
            "type": anomaly_type,
            "severity": severity,
            "confidence": confidence,
            "message": message,
        },
    )


def forecast_updated_event(run_id: str, step: int, forecast: Dict[str, Any]) -> Event:
    return Event(run_id=run_id, event="forecast_updated", step=step, data={"forecast": forecast})
