import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))

import pytest
from prc_sdk.events import Event, metric_event, run_started_event, EVENT_TYPES, SCHEMA_VERSION


def test_event_types_are_known():
    assert "metric" in EVENT_TYPES
    assert "run_started" in EVENT_TYPES
    assert "forecast_updated" in EVENT_TYPES


def test_unknown_event_type_raises():
    with pytest.raises(ValueError):
        Event(run_id="r1", event="not_a_real_event")


def test_metric_event_roundtrip():
    e = metric_event("run_1", step=10, epoch=1, metrics={"loss": 0.5})
    d = e.to_dict()
    e2 = Event.from_dict(d)
    assert e2.run_id == "run_1"
    assert e2.event == "metric"
    assert e2.step == 10
    assert e2.data["metrics"]["loss"] == 0.5
    assert e2.schema_version == SCHEMA_VERSION


def test_from_dict_tolerates_unknown_fields():
    d = {
        "run_id": "run_1",
        "event": "metric",
        "step": 1,
        "epoch": 0,
        "timestamp": "2026-01-01T00:00:00Z",
        "data": {"metrics": {}},
        "some_future_field": "ignored",
    }
    e = Event.from_dict(d)
    assert e.run_id == "run_1"


def test_run_started_event_shape():
    e = run_started_event("run_2", project="p", run_name="r", config={"lr": 0.01})
    assert e.event == "run_started"
    assert e.data["project"] == "p"
    assert e.data["config"]["lr"] == 0.01
