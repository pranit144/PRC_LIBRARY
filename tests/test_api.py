import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_api.db")
    monkeypatch.setenv("PRC_DB_PATH", db_path)

    import importlib
    import server.main as main_mod
    importlib.reload(main_mod)  # fresh app + storage bound to the temp db
    return TestClient(main_mod.app)


def start_run(client, run_id="run_1", project="proj"):
    event = {
        "run_id": run_id, "event": "run_started", "timestamp": "2026-01-01T00:00:00Z",
        "data": {"project": project, "run_name": "r", "config": {"lr": 0.001}},
    }
    return client.post("/api/events", json=event)


def send_metric(client, run_id, step, train_loss, val_loss):
    ev = {
        "run_id": run_id, "event": "metric", "timestamp": "2026-01-01T00:00:00Z",
        "step": step, "epoch": 0, "data": {"metrics": {"train_loss": train_loss, "val_loss": val_loss}},
    }
    return client.post("/api/events", json=ev)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_full_run_flow(client):
    assert start_run(client).status_code == 200

    for step in range(12):
        val = 0.9 - step * 0.02 if step < 6 else 0.78 + (step - 6) * 0.03
        send_metric(client, "run_1", step, 1.0 - step * 0.05, val)

    r = client.get("/api/runs/run_1")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "running"
    assert body["last_step"] == 11
    assert len(body["anomalies"]) >= 1

    r = client.get("/api/runs/run_1/metrics")
    assert len(r.json()) == 12

    r = client.get("/api/runs/run_1/forecast?metric=val_loss")
    assert r.json()["status"] == "ok"

    r = client.get("/api/projects")
    assert any(p["name"] == "proj" for p in r.json())


def test_unknown_run_404s(client):
    r = client.get("/api/runs/does_not_exist")
    assert r.status_code == 404


def test_assistant_endpoint(client):
    start_run(client, run_id="run_2")
    for step in range(12):
        val = 0.9 - step * 0.02 if step < 6 else 0.78 + (step - 6) * 0.03
        send_metric(client, "run_2", step, 1.0 - step * 0.05, val)
    r = client.get("/api/runs/run_2/assistant", params={"question": "Should I stop this run?"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_websocket_streams_new_events(client):
    start_run(client, run_id="run_ws")
    with client.websocket_connect("/api/ws/runs/run_ws") as ws:
        send_metric(client, "run_ws", 0, 1.0, 0.9)
        msg = ws.receive_json()
        assert msg["event"] == "metric"
        assert msg["run_id"] == "run_ws"
