import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))

import tempfile
from pathlib import Path

from prc_sdk import Monitor
from prc_sdk.transport import LocalBuffer


def test_monitor_writes_local_buffer(tmp_path):
    m = Monitor(project="proj", run_name="run1", local_dir=str(tmp_path), server_url=None)
    m.log(step=0, epoch=0, train_loss=1.0, val_loss=0.9)
    m.log(step=1, epoch=0, train_loss=0.9, val_loss=0.85)
    m.finish()

    buf_path = tmp_path / "proj" / f"{m.run_id}.jsonl"
    assert buf_path.exists()
    events = LocalBuffer(buf_path).read_all()
    types = [e["event"] for e in events]
    assert types[0] == "run_started"
    assert types[-1] == "run_finished"
    assert types.count("metric") == 2


def test_monitor_never_raises_on_bad_server_url(tmp_path):
    # Nonsense URL: sends should silently fail without raising.
    m = Monitor(project="proj", run_name="run2", local_dir=str(tmp_path),
                server_url="http://localhost:1")  # nothing listening
    m.log(step=0, epoch=0, train_loss=1.0)
    m.finish()  # should not raise


def test_monitor_context_manager(tmp_path):
    with Monitor(project="proj", run_name="run3", local_dir=str(tmp_path), server_url=None) as m:
        m.log(step=0, epoch=0, train_loss=1.0)
    assert m._finished is True


def test_monitor_survives_bad_metric_values(tmp_path):
    m = Monitor(project="proj", run_name="run4", local_dir=str(tmp_path), server_url=None)
    # Non-numeric metric values shouldn't crash logging (still JSON-serializable).
    m.log(step=0, epoch=0, note="not a number")
    m.finish()
