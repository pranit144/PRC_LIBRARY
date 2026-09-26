"""
Tests for the three Tier-1 integrations:
  - prc_sdk.hardware  (HardwareSampler + collect_hardware_snapshot)
  - prc_sdk.transformers  (PrcHfCallback)
  - prc_sdk.lightning     (PrcLightningCallback)

These tests do NOT require torch, transformers, pytorch-lightning, pynvml,
or psutil to be installed. Every external dependency is mocked so the suite
runs cleanly in the base dev environment (pip install -e ".[dev]").
"""
from __future__ import annotations

import sys
import os
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Make prc_sdk importable regardless of where pytest is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))

from prc_sdk.monitor import Monitor


# ===========================================================================
# Helpers: a fake Monitor that records what methods were called on it.
# ===========================================================================

class FakeMonitor:
    def __init__(self):
        self.calls = []
        self.config = {}
        self.run_id = "run_test"

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))

    def log(self, step, epoch=0, **metrics):
        self._record("log", step=step, epoch=epoch, **metrics)

    def log_system_metrics(self, step, stats):
        self._record("log_system_metrics", step=step, stats=stats)

    def log_checkpoint(self, step, epoch, path, metrics=None):
        self._record("log_checkpoint", step=step, epoch=epoch, path=path)

    def log_gradient_stats(self, step, epoch, stats):
        self._record("log_gradient_stats", step=step, epoch=epoch, stats=stats)

    def epoch_started(self, epoch):
        self._record("epoch_started", epoch=epoch)

    def epoch_finished(self, epoch, metrics=None):
        self._record("epoch_finished", epoch=epoch)

    def finish(self, status="completed"):
        self._record("finish", status=status)

    def called_names(self):
        return [c[0] for c in self.calls]


# ===========================================================================
# 1. Hardware module
# ===========================================================================

class TestCollectHardwareSnapshot(unittest.TestCase):
    """collect_hardware_snapshot should merge CPU/RAM and GPU results."""

    def test_returns_dict(self):
        from prc_sdk.hardware import collect_hardware_snapshot
        result = collect_hardware_snapshot()
        self.assertIsInstance(result, dict)

    def test_cpu_ram_included_when_psutil_available(self):
        """Patch psutil so we don't need it installed."""
        psutil_mock = types.ModuleType("psutil")
        psutil_mock.cpu_percent = lambda interval=None: 42.0
        vm = MagicMock()
        vm.used = 2 * 1024 ** 3
        vm.total = 8 * 1024 ** 3
        vm.percent = 25.0
        psutil_mock.virtual_memory = lambda: vm

        with patch.dict("sys.modules", {"psutil": psutil_mock, "pynvml": None}):
            from prc_sdk import hardware as hw_module
            import importlib
            importlib.reload(hw_module)
            result = hw_module._collect_cpu_ram()

        self.assertIn("cpu_utilization_pct", result)
        self.assertAlmostEqual(result["cpu_utilization_pct"], 42.0)
        self.assertIn("ram_used_mb", result)

    def test_gracefully_returns_empty_when_psutil_missing(self):
        """When psutil is not installed, cpu/ram collector returns {}."""
        with patch.dict("sys.modules", {"psutil": None}):
            from prc_sdk.hardware import _collect_cpu_ram
            result = _collect_cpu_ram()
        self.assertEqual(result, {})

    def test_gracefully_returns_empty_when_pynvml_missing(self):
        """When pynvml is not installed, gpu collector returns {}."""
        with patch.dict("sys.modules", {"pynvml": None}):
            from prc_sdk.hardware import _collect_gpu_pynvml
            result = _collect_gpu_pynvml()
        self.assertEqual(result, {})

    def test_gracefully_returns_empty_when_torch_missing(self):
        """When torch is not installed, torch.cuda fallback returns {}."""
        with patch.dict("sys.modules", {"torch": None}):
            from prc_sdk.hardware import _collect_gpu_torch
            result = _collect_gpu_torch()
        self.assertEqual(result, {})


class TestHardwareSampler(unittest.TestCase):
    """HardwareSampler background thread tests."""

    def test_start_stop_lifecycle(self):
        """Sampler thread starts, emits at least one system_metrics call, then stops."""
        import prc_sdk.hardware as hw_module

        monitor = FakeMonitor()
        called_event = threading.Event()

        # Intercept log_system_metrics to set the event immediately on first call.
        original_log = monitor.log_system_metrics
        def _intercepting_log(step, stats):
            original_log(step=step, stats=stats)
            called_event.set()
        monitor.log_system_metrics = _intercepting_log

        # Patch the module-level snapshot function BEFORE starting the thread.
        original_fn = hw_module.collect_hardware_snapshot
        hw_module.collect_hardware_snapshot = lambda: {"cpu_utilization_pct": 55.0}
        try:
            sampler = hw_module.HardwareSampler(monitor, interval_seconds=0.05)
            sampler.start()
            # Wait up to 1 s for a real call — much more reliable than a fixed sleep.
            was_called = called_event.wait(timeout=1.0)
            sampler.stop()
        finally:
            hw_module.collect_hardware_snapshot = original_fn

        self.assertTrue(was_called, "Expected log_system_metrics to be called by sampler")
        sys_calls = [c for c in monitor.calls if c[0] == "log_system_metrics"]
        self.assertGreaterEqual(len(sys_calls), 1)

    def test_start_idempotent(self):
        """Calling start() twice should not create two threads."""
        monitor = FakeMonitor()
        from prc_sdk.hardware import HardwareSampler
        sampler = HardwareSampler(monitor, interval_seconds=60.0)
        sampler.start()
        thread_before = sampler._thread
        sampler.start()  # second call — should be a no-op
        self.assertIs(sampler._thread, thread_before)
        sampler.stop()

    def test_stop_before_start_is_safe(self):
        """stop() before start() must not raise."""
        monitor = FakeMonitor()
        from prc_sdk.hardware import HardwareSampler
        sampler = HardwareSampler(monitor, interval_seconds=5.0)
        sampler.stop()  # should not raise

    def test_empty_snapshot_not_forwarded(self):
        """If snapshot returns {}, log_system_metrics must NOT be called."""
        monitor = FakeMonitor()
        with patch("prc_sdk.hardware.collect_hardware_snapshot", return_value={}):
            from prc_sdk.hardware import HardwareSampler
            sampler = HardwareSampler(monitor, interval_seconds=0.05)
            sampler.start()
            time.sleep(0.2)
            sampler.stop()
        sys_calls = [c for c in monitor.calls if c[0] == "log_system_metrics"]
        self.assertEqual(sys_calls, [])

    def test_monitor_enable_hardware_monitoring_flag(self, tmp_path=None):
        """Monitor(enable_hardware_monitoring=True) should auto-start a sampler."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with patch("prc_sdk.hardware.collect_hardware_snapshot", return_value={"cpu_utilization_pct": 5.0}):
                m = Monitor(
                    project="test",
                    run_name="hw-test",
                    local_dir=tmp,
                    server_url=None,
                    show_live_url=False,
                    enable_hardware_monitoring=True,
                    hardware_interval_seconds=0.1,
                )
                time.sleep(0.25)
                m.finish()

            self.assertIsNone(m._hardware_sampler._thread, "Thread should be None after stop()")


# ===========================================================================
# 2. Hugging Face callback
# ===========================================================================

def _make_hf_state(global_step=10, epoch=1.0, is_world_process_zero=True):
    state = MagicMock()
    state.global_step = global_step
    state.epoch = epoch
    state.is_world_process_zero = is_world_process_zero
    return state


def _make_hf_args(output_dir="/tmp/ckpt", **kwargs):
    args = MagicMock()
    args.output_dir = output_dir
    args.learning_rate = 2e-5
    args.per_device_train_batch_size = 8
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


class _FakeTrainerCallback:
    """Minimal stand-in for transformers.TrainerCallback."""
    pass


class TestPrcHfCallback(unittest.TestCase):

    def _build_callback(self, monitor=None):
        monitor = monitor or FakeMonitor()
        # Patch transformers so we never need it installed.
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.TrainerCallback = _FakeTrainerCallback

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            from prc_sdk import transformers as t_module
            import importlib
            # Reset cached class so the patch is picked up.
            t_module.PrcHfCallback._cls = None
            t_module._LazyHfCallback._cls = None
            cb_instance = t_module._make_prc_hf_callback()(monitor)
        return cb_instance, monitor

    def test_on_log_forwards_scalar_metrics(self):
        cb, monitor = self._build_callback()
        state = _make_hf_state(global_step=5, epoch=0.5)
        args = _make_hf_args()
        cb.on_log(args, state, None, logs={"loss": 0.8, "eval_loss": 0.9, "total_flos": "string"})

        log_calls = [c for c in monitor.calls if c[0] == "log"]
        self.assertEqual(len(log_calls), 1)
        kwargs = log_calls[0][1]
        self.assertEqual(kwargs["step"], 5)
        self.assertAlmostEqual(kwargs["train_loss"], 0.8)  # canonical rename
        self.assertAlmostEqual(kwargs["val_loss"], 0.9)    # canonical rename
        self.assertNotIn("total_flos", kwargs)             # string filtered out

    def test_on_log_skipped_for_non_rank0(self):
        cb, monitor = self._build_callback()
        state = _make_hf_state(is_world_process_zero=False)
        cb.on_log(_make_hf_args(), state, None, logs={"loss": 0.5})
        self.assertEqual(monitor.calls, [])

    def test_on_epoch_begin_called(self):
        cb, monitor = self._build_callback()
        state = _make_hf_state(global_step=0, epoch=2.0)
        cb.on_epoch_begin(_make_hf_args(), state, None)
        self.assertIn("epoch_started", monitor.called_names())

    def test_on_save_logs_checkpoint(self):
        cb, monitor = self._build_callback()
        state = _make_hf_state(global_step=100)
        args = _make_hf_args(output_dir="/models/llama")
        cb.on_save(args, state, None)
        ckpt_calls = [c for c in monitor.calls if c[0] == "log_checkpoint"]
        self.assertEqual(len(ckpt_calls), 1)
        self.assertIn("checkpoint-100", ckpt_calls[0][1]["path"])

    def test_on_train_begin_extracts_config(self):
        cb, monitor = self._build_callback()
        state = _make_hf_state()
        args = _make_hf_args()
        args.num_train_epochs = 3
        args.weight_decay = 0.01
        cb.on_train_begin(args, state, None)
        self.assertIn("learning_rate", monitor.config)

    def test_on_log_empty_logs_no_crash(self):
        cb, monitor = self._build_callback()
        state = _make_hf_state()
        cb.on_log(_make_hf_args(), state, None, logs={})  # should not raise


# ===========================================================================
# 3. PyTorch Lightning callback
# ===========================================================================

class _FakeLightningCallback:
    """Minimal stand-in for lightning.pytorch.Callback."""
    pass


def _make_trainer(global_step=10, current_epoch=1, is_global_zero=True,
                  callback_metrics=None):
    trainer = MagicMock()
    trainer.global_step = global_step
    trainer.current_epoch = current_epoch
    trainer.is_global_zero = is_global_zero
    trainer.callback_metrics = callback_metrics or {}
    # Mimic optional attributes.
    trainer.max_epochs = 10
    trainer.precision = "32"
    trainer.strategy = "auto"
    trainer.num_nodes = 1
    trainer.accumulate_grad_batches = 1
    trainer.gradient_clip_val = None
    trainer.gradient_clip_algorithm = None
    trainer.log_every_n_steps = 1
    trainer.lightning_module = MagicMock()
    trainer.lightning_module.hparams = {"lr": 1e-3, "batch_size": 32}
    return trainer


class TestPrcLightningCallback(unittest.TestCase):

    def _build_callback(self, monitor=None):
        monitor = monitor or FakeMonitor()
        fake_lightning = types.ModuleType("lightning")
        fake_lightning_pytorch = types.ModuleType("lightning.pytorch")
        fake_lightning_pytorch.Callback = _FakeLightningCallback
        fake_lightning.pytorch = fake_lightning_pytorch

        with patch.dict("sys.modules", {
            "lightning": fake_lightning,
            "lightning.pytorch": fake_lightning_pytorch,
            "pytorch_lightning": None,
        }):
            from prc_sdk import lightning as lg_module
            import importlib
            lg_module._LazyLightningCallback._cls = None
            cb_instance = lg_module._make_prc_lightning_callback()(monitor)
        return cb_instance, monitor

    def test_on_train_batch_end_logs_metrics(self):
        cb, monitor = self._build_callback()
        import torch
        trainer = _make_trainer(
            global_step=5,
            callback_metrics={"train_loss": MagicMock(item=lambda: 0.42)},
        )
        cb.on_train_batch_end(trainer, None, None, None, 0)
        log_calls = [c for c in monitor.calls if c[0] == "log"]
        self.assertEqual(len(log_calls), 1)
        self.assertAlmostEqual(log_calls[0][1]["train_loss"], 0.42)

    def test_non_rank0_skips_logging(self):
        cb, monitor = self._build_callback()
        trainer = _make_trainer(is_global_zero=False)
        cb.on_train_batch_end(trainer, None, None, None, 0)
        self.assertEqual(monitor.calls, [])

    def test_on_validation_epoch_end_emits_log_and_epoch_finished(self):
        cb, monitor = self._build_callback()
        trainer = _make_trainer(
            current_epoch=2,
            callback_metrics={"val_loss": MagicMock(item=lambda: 0.55)},
        )
        cb.on_validation_epoch_end(trainer, None)
        names = monitor.called_names()
        self.assertIn("log", names)
        self.assertIn("epoch_finished", names)

    def test_on_exception_marks_run_failed(self):
        cb, monitor = self._build_callback()
        trainer = _make_trainer()
        cb.on_exception(trainer, None, RuntimeError("CUDA OOM"))
        finish_calls = [c for c in monitor.calls if c[0] == "finish"]
        self.assertEqual(len(finish_calls), 1)
        self.assertEqual(finish_calls[0][1]["status"], "failed")

    def test_on_save_checkpoint_logs_checkpoint(self):
        cb, monitor = self._build_callback()
        trainer = _make_trainer(global_step=50, current_epoch=3)
        cb.on_save_checkpoint(trainer, None, {})
        ckpt_calls = [c for c in monitor.calls if c[0] == "log_checkpoint"]
        self.assertEqual(len(ckpt_calls), 1)

    def test_setup_extracts_config_for_fit_stage(self):
        cb, monitor = self._build_callback()
        trainer = _make_trainer()
        cb.setup(trainer, None, stage="fit")
        self.assertIn("max_epochs", monitor.config)
        self.assertIn("hparam_lr", monitor.config)

    def test_log_every_n_steps_respected(self):
        """With log_every_n_steps=3, only every third step should emit a log."""
        cb, monitor = self._build_callback()
        cb._log_every_n_steps = 3
        trainer = _make_trainer(
            callback_metrics={"train_loss": MagicMock(item=lambda: 0.1)},
        )
        for step in range(9):
            trainer.global_step = step
            cb.on_train_batch_end(trainer, None, None, None, step)

        log_calls = [c for c in monitor.calls if c[0] == "log"]
        # Steps 0, 3, 6 → 3 calls
        self.assertEqual(len(log_calls), 3)


if __name__ == "__main__":
    unittest.main()
