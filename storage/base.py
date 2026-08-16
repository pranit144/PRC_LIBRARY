"""
Storage abstraction. The server code depends only on this interface, not
on any specific database, so SQLite can be swapped for PostgreSQL later
without touching API/websocket code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Storage(ABC):
    @abstractmethod
    def save_event(self, event: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def list_projects(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_project(self, project: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_runs(self, project: Optional[str] = None) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_events(self, run_id: str, event_type: Optional[str] = None,
                     since_step: Optional[int] = None) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_metrics(self, run_id: str) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def save_forecast(self, run_id: str, step: int, forecast: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def get_latest_forecast(self, run_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def save_anomaly(self, run_id: str, anomaly: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def list_anomalies(self, run_id: str) -> List[Dict[str, Any]]:
        ...
