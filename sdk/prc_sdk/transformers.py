"""
Optional Hugging Face Transformers integration for the prc SDK.

Provides `PrcHfCallback`, a drop-in `transformers.TrainerCallback` that gives
you live metric tracking, anomaly detection, and dashboard monitoring for
any model fine-tuned via the Hugging Face `Trainer` — including LLMs (Llama,
Mistral, Qwen, Phi), encoder models (BERT, RoBERTa), and diffusion pipelines
built on `Trainer`.

Zero-touch usage::

    from prc_sdk import Monitor
    from prc_sdk.transformers import PrcHfCallback

    monitor = Monitor(project="llama-finetune", run_name="experiment-01")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[PrcHfCallback(monitor)],
    )
    trainer.train()
    monitor.finish()

Key design decisions
--------------------
* **Rank-0 only** — In DDP / FSDP / DeepSpeed multi-GPU runs, only the
  primary process (``state.is_world_process_zero``) emits events to prc.
  This prevents 8× duplicate events and conflicting step counters.

* **Lazy import** — importing this module does NOT require `transformers` to
  be installed. The base class is resolved at first instantiation through the
  same lazy pattern used by `prc_sdk.tensorflow`.

* **Scalar filter** — HF ``logs`` dicts can contain non-numeric entries
  (``total_flos``, string labels, etc.). Only ``float``/``int`` values are
  forwarded.

* **Metric normalisation** — HF uses ``loss`` / ``eval_loss``; prc's anomaly
  detectors look for ``train_loss`` / ``val_loss``. Both key sets are emitted
  so the dashboard and the analytics engine both see the right names.

* **Fail-safe** — every callback method is wrapped in a try/except. A broken
  prc connection never kills a long-running fine-tune.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("prc.transformers")


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("prc.transformers: monitoring call failed (non-fatal)")
        return None


def _is_scalar(v) -> bool:
    """Return True for plain int/float values (excludes str, lists, …)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _normalize_logs(logs: Dict[str, Any]) -> Dict[str, float]:
    """
    Filter to scalars and rename HF keys to prc-canonical names.

    HF key      ->  prc key
    ----------      -------
    loss            train_loss  (+ keep loss)
    eval_loss       val_loss    (+ keep eval_loss)
    """
    out: Dict[str, float] = {}
    for k, v in logs.items():
        if not _is_scalar(v):
            continue
        out[k] = float(v)

    # Canonical renames for anomaly detectors.
    if "loss" in out and "train_loss" not in out:
        out["train_loss"] = out["loss"]
    if "eval_loss" in out and "val_loss" not in out:
        out["val_loss"] = out["eval_loss"]

    return out


def _extract_hf_config(training_args) -> Dict[str, Any]:
    """
    Pull the most diagnostic hyperparameters from a TrainingArguments object.
    Uses `getattr` everywhere so it's forward/backward compatible with
    different transformers versions.
    """
    fields = [
        "learning_rate",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "num_train_epochs",
        "warmup_steps",
        "warmup_ratio",
        "weight_decay",
        "lr_scheduler_type",
        "optim",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "fp16",
        "bf16",
        "seed",
        "max_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
    ]
    config: Dict[str, Any] = {}
    for f in fields:
        val = getattr(training_args, f, None)
        if val is not None:
            # lr_scheduler_type is a SchedulerType enum — stringify it.
            config[f] = str(val) if not isinstance(val, (int, float, bool, str)) else val
    return config


# ---------------------------------------------------------------------------
# Lazy callback builder (same pattern as PrcKerasCallback in tensorflow.py)
# ---------------------------------------------------------------------------

def _make_prc_hf_callback():
    """Build PrcHfCallback inheriting from the real TrainerCallback class."""
    try:
        from transformers import TrainerCallback  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "prc: 'transformers' is not installed. "
            "Install it with: pip install transformers   "
            "or: pip install prc[transformers]"
        ) from exc

    class PrcHfCallback(TrainerCallback):
        """
        Hugging Face TrainerCallback that streams training events to prc.

        Attach it to your Trainer with::

            trainer = Trainer(
                ...,
                callbacks=[PrcHfCallback(monitor)],
            )

        Parameters
        ----------
        monitor:
            An initialised ``prc_sdk.Monitor`` instance.
        log_model_config:
            If True (default), extract and log ``TrainingArguments`` as the
            run config on ``on_train_begin``.
        """

        def __init__(self, monitor, log_model_config: bool = True) -> None:
            self._monitor = monitor
            self._log_model_config = log_model_config
            self._last_epoch: int = 0

        # -- helpers ----------------------------------------------------------

        def _is_rank0(self, state) -> bool:
            """Only log from the primary process in multi-GPU runs."""
            return getattr(state, "is_world_process_zero", True)

        # -- lifecycle callbacks ----------------------------------------------

        def on_train_begin(self, args, state, control, **kwargs):
            if not self._is_rank0(state):
                return
            try:
                if self._log_model_config:
                    config = _extract_hf_config(args)
                    # Merge into the monitor's existing config (non-destructive).
                    self._monitor.config.update(config)
                _safe_call(self._monitor.epoch_started, 0)
            except Exception:
                logger.exception("prc.transformers: on_train_begin failed (non-fatal)")

        def on_epoch_begin(self, args, state, control, **kwargs):
            if not self._is_rank0(state):
                return
            epoch = int(getattr(state, "epoch", 0) or 0)
            self._last_epoch = epoch
            _safe_call(self._monitor.epoch_started, epoch)

        def on_log(self, args, state, control, logs=None, **kwargs):
            """
            HF calls this at every ``logging_steps`` and after every eval.
            ``logs`` contains train loss, eval loss/metrics, and LR.
            """
            if not self._is_rank0(state):
                return
            if not logs:
                return

            step = int(getattr(state, "global_step", 0) or 0)
            epoch = int(getattr(state, "epoch", 0) or 0)
            self._last_epoch = epoch

            metrics = _normalize_logs(logs)
            if metrics:
                _safe_call(self._monitor.log, step=step, epoch=epoch, **metrics)

        def on_epoch_end(self, args, state, control, **kwargs):
            if not self._is_rank0(state):
                return
            epoch = int(getattr(state, "epoch", 0) or 0)
            _safe_call(self._monitor.epoch_finished, epoch)

        def on_save(self, args, state, control, **kwargs):
            """A checkpoint was saved to disk."""
            if not self._is_rank0(state):
                return
            step = int(getattr(state, "global_step", 0) or 0)
            epoch = self._last_epoch
            output_dir = getattr(args, "output_dir", "checkpoint")
            checkpoint_path = f"{output_dir}/checkpoint-{step}"
            _safe_call(
                self._monitor.log_checkpoint,
                step=step,
                epoch=epoch,
                path=checkpoint_path,
                metrics={},
            )

        def on_train_end(self, args, state, control, **kwargs):
            # We intentionally do NOT call monitor.finish() here so the user
            # keeps the same explicit-finish pattern as in the PyTorch and
            # Keras examples. If they want auto-finish, they can pass
            # auto_finish=True (see __init__).
            pass

    return PrcHfCallback


class _LazyHfCallback:
    """Defers building the real TrainerCallback subclass until first use."""

    _cls = None

    def __call__(self, *args, **kwargs):
        if _LazyHfCallback._cls is None:
            _LazyHfCallback._cls = _make_prc_hf_callback()
        return _LazyHfCallback._cls(*args, **kwargs)

    def __instancecheck__(self, instance):
        if _LazyHfCallback._cls is None:
            return False
        return isinstance(instance, _LazyHfCallback._cls)


PrcHfCallback = _LazyHfCallback()
