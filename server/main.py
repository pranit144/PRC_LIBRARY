from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage import SqliteStorage
from analytics import AnalyticsEngine
from forecasting import SimpleTrendForecastEngine
from assistant import DeterministicAssistant

from .ws_manager import ConnectionManager

DB_PATH = os.environ.get("PRC_DB_PATH", "prc.db")
DASHBOARD_DIST = Path(os.environ.get("PRC_DASHBOARD_DIST", Path(__file__).resolve().parents[1] / "dashboard" / "dist"))

app = FastAPI(title="prc API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = SqliteStorage(DB_PATH)
analytics_engine = AnalyticsEngine()
forecast_engine = SimpleTrendForecastEngine()
assistant = DeterministicAssistant()
ws_manager = ConnectionManager()

# All API routes live under /api/* so they never collide with the
# dashboard's own client-side routes (e.g. the frontend's /runs/:id page
# vs the API's /runs/{run_id} JSON endpoint), since both are served from
# the same port/process.
api = APIRouter(prefix="/api")


class EventIn(BaseModel):
    run_id: str
    event: str
    schema_version: int = 1
    timestamp: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    data: Dict[str, Any] = {}


@api.get("/health")
def health():
    return {"status": "ok"}


@api.get("/projects")
def list_projects():
    return storage.list_projects()


@api.get("/projects/{project}")
def get_project(project: str):
    p = storage.get_project(project)
    if not p:
        raise HTTPException(status_code=404, detail="project not found")
    runs = storage.list_runs(project=project)
    return {**p, "runs": runs}


@api.get("/runs")
def list_runs(project: Optional[str] = None):
    return storage.list_runs(project=project)


@api.get("/runs/{run_id}")
def get_run(run_id: str):
    run = storage.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    anomalies = [a["anomaly"] for a in storage.list_anomalies(run_id)]
    latest_forecast = storage.get_latest_forecast(run_id)
    return {
        **run,
        "anomalies": anomalies,
        "forecast": latest_forecast["forecast"] if latest_forecast else None,
    }


@api.get("/runs/{run_id}/events")
def get_events(run_id: str, event_type: Optional[str] = None, since_step: Optional[int] = None):
    return storage.list_events(run_id, event_type=event_type, since_step=since_step)


@api.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str):
    return storage.list_metrics(run_id)


@api.get("/runs/{run_id}/forecast")
def get_forecast(run_id: str, metric: str = "val_loss"):
    metrics = storage.list_metrics(run_id)
    forecast = forecast_engine.forecast(metrics, metric)
    if forecast.get("status") == "ok":
        step = metrics[-1]["step"] if metrics else 0
        storage.save_forecast(run_id, step, forecast)
    return forecast


@api.get("/runs/{run_id}/assistant")
def ask_assistant(run_id: str, question: str):
    run = storage.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    metrics = storage.list_metrics(run_id)
    anomalies = [a["anomaly"] for a in storage.list_anomalies(run_id)]
    latest_forecast = storage.get_latest_forecast(run_id)
    forecast = latest_forecast["forecast"] if latest_forecast else None
    answer = assistant.answer(question, run, metrics, anomalies, forecast)
    return {"question": question, "answer": answer}


@api.post("/events")
async def post_event(event: EventIn):
    event_dict = event.model_dump()
    storage.save_event(event_dict)

    # Run analytics on every metric event so anomalies stay fresh.
    if event.event == "metric":
        metrics = storage.list_metrics(event.run_id)
        gradient_events = storage.list_events(event.run_id, event_type="gradient_stats")
        gradient_history = [g["data"] for g in gradient_events]
        found = analytics_engine.analyze(metrics, gradient_history)

        existing_types = {a["anomaly"]["type"] for a in storage.list_anomalies(event.run_id)[-5:]}
        for anomaly in found:
            # Avoid re-saving the same anomaly type back-to-back on every
            # single metric tick; only record it again if it wasn't among
            # the most recent detections for this run.
            if anomaly["type"] in existing_types:
                continue
            storage.save_anomaly(event.run_id, anomaly)
            await ws_manager.broadcast(event.run_id, {"event": "anomaly", "data": anomaly})

    await ws_manager.broadcast(event.run_id, event_dict)
    return {"status": "accepted"}


@api.websocket("/ws/runs/{run_id}")
async def websocket_run(websocket: WebSocket, run_id: str):
    await ws_manager.connect(run_id, websocket)
    try:
        while True:
            # We don't expect incoming client messages, but keep the
            # connection alive and detect disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(run_id, websocket)
    except Exception:
        await ws_manager.disconnect(run_id, websocket)


app.include_router(api)


# -- serve the built dashboard from the same port/process, if present ----
# In local dev, run `npm run dev` in dashboard/ separately (Vite on 5173)
# instead - this static mount only activates once `npm run build` has
# produced dashboard/dist, which is what makes single-port deployment
# (and therefore Colab/Kaggle/remote proxying) possible.
if DASHBOARD_DIST.exists():
    assets_dir = DASHBOARD_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="dashboard-assets")

    index_file = DASHBOARD_DIST / "index.html"

    @app.get("/{full_path:path}")
    async def serve_dashboard(full_path: str):
        # Let any real static file through (favicon, etc); otherwise fall
        # back to index.html so the dashboard's client-side router
        # (react-router) can handle the path itself.
        candidate = DASHBOARD_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_file)
