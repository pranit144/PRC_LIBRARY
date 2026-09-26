"""
Optional PyTorch Lightning integration for the prc SDK.

Provides `PrcLightningCallback`, a ``lightning.pytorch.Callback`` (or
``pytorch_lightning.Callback`` for legacy Lightning) that gives you live
metric tracking, anomaly detection, and dashboard monitoring for any
Lightning training loop — from a simple single-GPU `Trainer` to
multi-node DDP.

Zero-touch usage::

    from prc_sdk import Monitor
    from prc_sdk.lightning import PrcLightningCallback

    monitor = Monitor(project="my-model", run_name="experiment-01")

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[PrcLightningCallback(monitor)],
    )
    trainer.fit(model, train_loader, val_loader)
    monitor.finish()

Key design decisions
--------------------
* **Rank-0 only** — checked via ``trainer.is_global_zero`` so only the
  primary process emits events in multi-GPU / multi-node runs.

* **Dual import path** — tries ``lightning.pytorch`` first (modern
  Lightning ≥ 2.0), falls back to ``pytorch_lightning`` (legacy).
  A clear ImportError is raised if neither is installed.

* **Metric source** — reads ``trainer.callback_metrics`` which is the
  Lightning-standard dict that always holds the most recently logged values
  for every metric key (train_loss, val_loss, etc.) as tensors. Values are
  converted to float scalars before forwarding.

* **Metric normalisation** — Lightning typically uses ``train_loss_step`` /
  ``val_loss`` key names. We pass them through verbatim; prc's analytics
  detectors accept both ``train_loss`` and ``val_loss`` already.

* **Fail-safe** — every callback method is try/excepted; a broken prc
  connection never kills training.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("prc.lightning")


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("prc.lightning: monitoring call failed (non-fatal)")
        return None


def _to_scalar(v) -> Optional[float]:
    """Convert a Lightning callback_metrics value (often a Tensor) to float."""
    try:
        # Tensor.item() works for both torch.Tensor and numpy scalars.
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except (TypeError, ValueError):
        return None


def _harvest_metrics(trainer) -> Dict[str, float]:
    """
    Collect all numeric scalar metrics from trainer.callback_metrics.
    Silently skips non-scalars (e.g. epoch-level histogram tensors).
    """
    out: Dict[str, float] = {}
    try:
        for k, v in trainer.callback_metrics.items():
            scalar = _to_scalar(v)
            if scalar is not None:
                out[k] = scalar
    except Exception:
        pass
    return out


def _extract_lightning_config(trainer) -> Dict[str, Any]:
    """
    Pull common hyperparameters from the Lightning Trainer and the
    LightningModule's hparams dict (if it exposes one).
    """
    config: Dict[str, Any] = {}

    # Trainer-level settings.
    for attr in ("max_epochs", "max_steps", "precision", "strategy",
                 "num_nodes", "accumulate_grad_batches", "gradient_clip_val",
                 "gradient_clip_algorithm", "log_every_n_steps"):
        val = getattr(trainer, attr, None)
        if val is not None:
            config[attr] = str(val) if not isinstance(val, (int, float, bool, str)) else val

    # LightningModule hparams (user-defined hyperparameters).
    try:
        hparams = trainer.lightning_module.hparams
        if hparams:
            for k, v in dict(hparams).items():
                if isinstance(v, (int, float, bool, str)):
                    config[f"hparam_{k}"] = v
    except Exception:
        pass

    return config


# ---------------------------------------------------------------------------
# Lazy callback builder
# ---------------------------------------------------------------------------

def _get_callback_base():
    """
    Return the Callback base class, preferring modern ``lightning.pytorch``
    over the legacy ``pytorch_lightning`` package.
    """
    try:
        from lightning.pytorch import Callback  # type: ignore
        return Callback
    except ImportError:
        pass
    try:
        from pytorch_lightning import Callback  # type: ignore
        return Callback
    except ImportError:
        pass
    raise ImportError(
        "prc: neither 'lightning' nor 'pytorch_lightning' is installed. "
        "Install with: pip install lightning   "
        "or: pip install pytorch-lightning"
    )


def _make_prc_lightning_callback():
    Base = _get_callback_base()

    class PrcLightningCallback(Base):
        """
        PyTorch Lightning Callback that streams training events to prc.

        Attach to your Trainer::

            trainer = pl.Trainer(
                callbacks=[PrcLightningCallback(monitor)],
            )

        Parameters
        ----------
        monitor:
            An initialised ``prc_sdk.Monitor`` instance.
        log_model_config:
            If True (default), extract Trainer + LightningModule hparams
            and merge them into the run config on ``setup``.
        log_every_n_steps:
            Log metrics every N training steps. Default: 1 (every step
            where Lightning emits a log). Set higher for very fast loops.
        """

        def __init__(
            self,
            monitor,
            log_model_config: bool = True,
            log_every_n_steps: int = 1,
        ) -> None:
            super().__init__()
            self._monitor = monitor
            self._log_model_config = log_model_config
            self._log_every_n_steps = max(1, log_every_n_steps)
            self._last_epoch: int = 0
            self._global_step: int = 0

        # -- helpers ----------------------------------------------------------

        def _is_rank0(self, trainer) -> bool:
            return getattr(trainer, "is_global_zero", True)

        # -- lifecycle --------------------------------------------------------

        def setup(self, trainer, pl_module, stage: str) -> None:
            """Called once per stage (fit / validate / test / predict)."""
            if stage != "fit":
                return
            if not self._is_rank0(trainer):
                return
            if self._log_model_config:
                config = _extract_lightning_config(trainer)
                self._monitor.config.update(config)

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            if not self._is_rank0(trainer):
                return
            epoch = trainer.current_epoch
            self._last_epoch = epoch
            _safe_call(self._monitor.epoch_started, epoch)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            if not self._is_rank0(trainer):
                return

            self._global_step = trainer.global_step

            if self._global_step % self._log_every_n_steps != 0:
                return

            metrics = _harvest_metrics(trainer)
            if metrics:
                _safe_call(
                    self._monitor.log,
                    step=self._global_step,
                    epoch=self._last_epoch,
                    **metrics,
                )

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if not self._is_rank0(trainer):
                return

            metrics = _harvest_metrics(trainer)
            epoch = trainer.current_epoch
            self._last_epoch = epoch

            if metrics:
                # Emit validation metrics as a metric event so anomaly
                # detectors (overfitting, plateau) can evaluate them.
                _safe_call(
                    self._monitor.log,
                    step=self._global_step,
                    epoch=epoch,
                    **metrics,
                )
            _safe_call(self._monitor.epoch_finished, epoch, metrics or None)

        def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
            if not self._is_rank0(trainer):
                return
            step = trainer.global_step
            epoch = trainer.current_epoch
            ckpt_path = getattr(trainer, "checkpoint_callback", None)
            path = getattr(ckpt_path, "last_model_path", None) or f"checkpoint-epoch{epoch}-step{step}"
            _safe_call(
                self._monitor.log_checkpoint,
                step=step,
                epoch=epoch,
                path=str(path),
                metrics=_harvest_metrics(trainer),
            )

        def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
            if not self._is_rank0(trainer):
                return
            logger.warning("prc.lightning: training raised %s — marking run as failed", type(exception).__name__)
            _safe_call(self._monitor.finish, status="failed")

        def on_fit_end(self, trainer, pl_module) -> None:
            # We intentionally do NOT call monitor.finish() here — same
            # pattern as PrcKerasCallback and PrcHfCallback: the user
            # calls monitor.finish() explicitly after trainer.fit().
            pass

    return PrcLightningCallback


class _LazyLightningCallback:
    """Defers building the real Callback subclass until first use."""

    _cls = None

    def __call__(self, *args, **kwargs):
        if _LazyLightningCallback._cls is None:
            _LazyLightningCallback._cls = _make_prc_lightning_callback()
        return _LazyLightningCallback._cls(*args, **kwargs)

    def __instancecheck__(self, instance):
        if _LazyLightningCallback._cls is None:
            return False
        return isinstance(instance, _LazyLightningCallback._cls)


PrcLightningCallback = _LazyLightningCallback()
