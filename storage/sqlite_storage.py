from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    name TEXT PRIMARY KEY,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project TEXT,
    run_name TEXT,
    config TEXT,
    status TEXT DEFAULT 'running',
    started_at TEXT,
    finished_at TEXT,
    last_step INTEGER DEFAULT 0,
    last_epoch INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    event TEXT,
    step INTEGER,
    epoch INTEGER,
    timestamp TEXT,
    data TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_run_event ON events(run_id, event);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    step INTEGER,
    forecast TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    anomaly TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    step INTEGER,
    epoch INTEGER,
    path TEXT,
    metrics TEXT
);
"""


class SqliteStorage(Storage):
    def __init__(self, db_path: str = "prc.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True) if Path(db_path).parent != Path("") else None
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- internal helpers ------------------------------------------------
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    # -- events ------------------------------------------------------
    def save_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            cur = self._conn.cursor()
            run_id = event["run_id"]
            etype = event["event"]
            data = event.get("data", {})

            if etype == "run_started":
                project = data.get("project")
                run_name = data.get("run_name")
                config = json.dumps(data.get("config", {}))
                cur.execute(
                    "INSERT OR IGNORE INTO projects(name, created_at) VALUES (?, ?)",
                    (project, event["timestamp"]),
                )
                cur.execute(
                    """INSERT OR REPLACE INTO runs(run_id, project, run_name, config, status, started_at)
                       VALUES (?, ?, ?, ?, 'running', ?)""",
                    (run_id, project, run_name, config, event["timestamp"]),
                )
            elif etype == "run_finished":
                status = data.get("status", "completed")
                cur.execute(
                    "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                    (status, event["timestamp"], run_id),
                )
            elif etype == "checkpoint":
                cur.execute(
                    "INSERT INTO checkpoints(run_id, step, epoch, path, metrics) VALUES (?, ?, ?, ?, ?)",
                    (run_id, event.get("step"), event.get("epoch"), data.get("path"),
                     json.dumps(data.get("metrics", {}))),
                )

            if event.get("step") is not None or event.get("epoch") is not None:
                cur.execute(
                    "UPDATE runs SET last_step = COALESCE(?, last_step), last_epoch = COALESCE(?, last_epoch) WHERE run_id = ?",
                    (event.get("step"), event.get("epoch"), run_id),
                )

            cur.execute(
                "INSERT INTO events(run_id, event, step, epoch, timestamp, data) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, etype, event.get("step"), event.get("epoch"), event.get("timestamp"), json.dumps(data)),
            )
            self._conn.commit()

    # -- projects ------------------------------------------------------
    def list_projects(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_project(self, project: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM projects WHERE name = ?", (project,)).fetchone()
        return self._row_to_dict(row) if row else None

    # -- runs ------------------------------------------------------
    def list_runs(self, project: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if project:
                rows = self._conn.execute(
                    "SELECT * FROM runs WHERE project = ? ORDER BY started_at DESC", (project,)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["config"] = json.loads(d.get("config") or "{}")
            out.append(d)
        return out

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = self._row_to_dict(row)
        d["config"] = json.loads(d.get("config") or "{}")
        return d

    # -- events / metrics ------------------------------------------------------
    def list_events(self, run_id: str, event_type: Optional[str] = None,
                     since_step: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM events WHERE run_id = ?"
        params: List[Any] = [run_id]
        if event_type:
            query += " AND event = ?"
            params.append(event_type)
        if since_step is not None:
            query += " AND (step IS NULL OR step > ?)"
            params.append(since_step)
        query += " ORDER BY id ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["data"] = json.loads(d.get("data") or "{}")
            out.append(d)
        return out

    def list_metrics(self, run_id: str) -> List[Dict[str, Any]]:
        events = self.list_events(run_id, event_type="metric")
        out = []
        for e in events:
            row = {"step": e["step"], "epoch": e["epoch"], "timestamp": e["timestamp"]}
            row.update(e["data"].get("metrics", {}))
            out.append(row)
        return out

    # -- forecasts ------------------------------------------------------
    def save_forecast(self, run_id: str, step: int, forecast: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO forecasts(run_id, step, forecast, created_at) VALUES (?, ?, ?, datetime('now'))",
                (run_id, step, json.dumps(forecast)),
            )
            self._conn.commit()

    def get_latest_forecast(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM forecasts WHERE run_id = ? ORDER BY id DESC LIMIT 1", (run_id,)
            ).fetchone()
        if not row:
            return None
        d = self._row_to_dict(row)
        d["forecast"] = json.loads(d["forecast"])
        return d

    # -- anomalies ------------------------------------------------------
    def save_anomaly(self, run_id: str, anomaly: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO anomalies(run_id, anomaly, created_at) VALUES (?, ?, datetime('now'))",
                (run_id, json.dumps(anomaly)),
            )
            self._conn.commit()

    def list_anomalies(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM anomalies WHERE run_id = ? ORDER BY id ASC", (run_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["anomaly"] = json.loads(d["anomaly"])
            out.append(d)
        return out
