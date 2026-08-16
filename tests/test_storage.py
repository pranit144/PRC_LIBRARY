import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage import SqliteStorage
from prc_sdk.events import run_started_event, metric_event, run_finished_event, checkpoint_event


def make_storage(tmp_path):
    return SqliteStorage(str(tmp_path / "test.db"))


def test_run_lifecycle(tmp_path):
    s = make_storage(tmp_path)
    run_id = "run_abc"
    s.save_event(run_started_event(run_id, "proj1", "run1", {"lr": 0.01}).to_dict())

    run = s.get_run(run_id)
    assert run is not None
    assert run["status"] == "running"
    assert run["project"] == "proj1"
    assert run["config"]["lr"] == 0.01

    s.save_event(run_finished_event(run_id, status="completed").to_dict())
    run = s.get_run(run_id)
    assert run["status"] == "completed"
    assert run["finished_at"] is not None


def test_projects_created_from_run_started(tmp_path):
    s = make_storage(tmp_path)
    s.save_event(run_started_event("run_1", "my-project", "r1", {}).to_dict())
    projects = s.list_projects()
    assert any(p["name"] == "my-project" for p in projects)


def test_metrics_and_events(tmp_path):
    s = make_storage(tmp_path)
    run_id = "run_metrics"
    s.save_event(run_started_event(run_id, "p", "r", {}).to_dict())
    for step in range(5):
        s.save_event(metric_event(run_id, step, 0, {"train_loss": 1.0 - step * 0.1}).to_dict())

    metrics = s.list_metrics(run_id)
    assert len(metrics) == 5
    assert metrics[0]["train_loss"] == 1.0
    assert metrics[-1]["step"] == 4

    events = s.list_events(run_id, event_type="metric")
    assert len(events) == 5

    run = s.get_run(run_id)
    assert run["last_step"] == 4


def test_checkpoints_stored(tmp_path):
    s = make_storage(tmp_path)
    run_id = "run_ckpt"
    s.save_event(run_started_event(run_id, "p", "r", {}).to_dict())
    s.save_event(checkpoint_event(run_id, step=10, epoch=1, path="/tmp/model.pt",
                                   metrics={"val_loss": 0.5}).to_dict())
    events = s.list_events(run_id, event_type="checkpoint")
    assert len(events) == 1
    assert events[0]["data"]["path"] == "/tmp/model.pt"


def test_forecast_and_anomaly_persistence(tmp_path):
    s = make_storage(tmp_path)
    run_id = "run_fc"
    s.save_event(run_started_event(run_id, "p", "r", {}).to_dict())

    s.save_forecast(run_id, step=10, forecast={"metric": "val_loss", "predicted_final": 0.4})
    latest = s.get_latest_forecast(run_id)
    assert latest["forecast"]["predicted_final"] == 0.4

    s.save_anomaly(run_id, {"type": "overfitting", "severity": "medium", "confidence": 0.7, "message": "m"})
    anomalies = s.list_anomalies(run_id)
    assert len(anomalies) == 1
    assert anomalies[0]["anomaly"]["type"] == "overfitting"


def test_list_runs_by_project(tmp_path):
    s = make_storage(tmp_path)
    s.save_event(run_started_event("run_x", "proj-a", "r", {}).to_dict())
    s.save_event(run_started_event("run_y", "proj-b", "r", {}).to_dict())

    proj_a_runs = s.list_runs(project="proj-a")
    assert len(proj_a_runs) == 1
    assert proj_a_runs[0]["run_id"] == "run_x"

    all_runs = s.list_runs()
    assert len(all_runs) == 2
