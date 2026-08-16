"""
Optional TensorFlow / Keras integration. Import lazily so the core SDK
never requires tensorflow to be installed.

Two ways to use it, matching how most Keras code is actually written:

1. Standard `model.fit()` — attach `PrcKerasCallback`. You get live
   loss/accuracy/learning-rate metrics, epoch/checkpoint events, and
   best-effort GPU stats, with zero changes to your training loop.

    from prc_sdk import Monitor
    from prc_sdk.tensorflow import PrcKerasCallback

    monitor = Monitor(project="mnist-tf", run_name="experiment-01")
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        callbacks=[PrcKerasCallback(monitor)],
    )
    monitor.finish()

2. Custom `tf.GradientTape` loop — call `gradient_stats()` /
   `parameter_stats()` yourself, the same way the PyTorch integration's
   `TorchMonitorHook` works, since Keras's `fit()` callbacks don't expose
   per-batch gradients directly.

    from prc_sdk.tensorflow import gradient_stats, parameter_stats, gpu_stats

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)

    if step % 20 == 0:
        monitor.log_gradient_stats(step=step, epoch=epoch,
                                    stats=gradient_stats(model.trainable_variables, grads))
        monitor.log_activation_stats(step=step, epoch=epoch,
                                      stats={"parameters": parameter_stats(model.trainable_variables)})
        monitor.log_system_metrics(step=step, stats=gpu_stats())

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("prc")


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("prc.tensorflow: monitoring call failed (non-fatal)")
        return None


# -- gradient / parameter stats (for custom GradientTape loops) ----------

def gradient_stats(trainable_variables: Sequence, gradients: Sequence) -> Dict[str, Any]:
    """
    trainable_variables / gradients: the parallel lists you get from
    `tape.gradient(loss, model.trainable_variables)`. Some entries in
    `gradients` may be None (e.g. unused variables) - those are skipped.
    """
    import tensorflow as tf

    norms: List[float] = []
    max_abs = 0.0
    min_abs = float("inf")
    n_with_grad = 0

    for g in gradients:
        if g is None:
            continue
        g = tf.convert_to_tensor(g)
        n_with_grad += 1
        norm = float(tf.norm(g).numpy())
        norms.append(norm)
        gabs = tf.abs(g)
        if tf.size(gabs).numpy() > 0:
            max_abs = max(max_abs, float(tf.reduce_max(gabs).numpy()))
            min_abs = min(min_abs, float(tf.reduce_min(gabs).numpy()))

    if not norms:
        return {"num_params_with_grad": 0}

    total_norm = sum(n ** 2 for n in norms) ** 0.5
    return {
        "num_params_with_grad": n_with_grad,
        "grad_norm_total": total_norm,
        "grad_norm_mean": sum(norms) / len(norms),
        "grad_norm_max": max(norms),
        "grad_norm_min": min(norms),
        "grad_abs_max": max_abs,
        "grad_abs_min": min_abs if min_abs != float("inf") else 0.0,
    }


def parameter_stats(trainable_variables: Sequence) -> Dict[str, Any]:
    import tensorflow as tf

    norms = [float(tf.norm(v).numpy()) for v in trainable_variables]
    if not norms:
        return {}
    return {
        "param_norm_total": sum(n ** 2 for n in norms) ** 0.5,
        "param_norm_mean": sum(norms) / len(norms),
        "param_norm_max": max(norms),
        "param_norm_min": min(norms),
    }


def gpu_stats() -> Dict[str, Any]:
    """Best-effort GPU utilization/memory. Returns {} if no GPU / tensorflow missing."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return {}
        gpu = gpus[0]
        info = tf.config.experimental.get_memory_info(gpu.name.replace("/physical_device:", "").lower() or "GPU:0")
        return {
            "gpu_name": gpu.name,
            "gpu_memory_current_mb": info.get("current", 0) / (1024 ** 2),
            "gpu_memory_peak_mb": info.get("peak", 0) / (1024 ** 2),
        }
    except Exception:
        return {}


def log_gradient_and_parameter_stats(monitor, trainable_variables: Sequence, gradients: Sequence,
                                      step: int, epoch: int) -> None:
    """Convenience wrapper for custom GradientTape loops, mirroring
    prc_sdk.pytorch.log_gradient_stats / log_parameter_stats."""
    g_stats = _safe_call(gradient_stats, trainable_variables, gradients)
    if g_stats:
        monitor.log_gradient_stats(step=step, epoch=epoch, stats=g_stats)
    p_stats = _safe_call(parameter_stats, trainable_variables)
    if p_stats:
        monitor.log_activation_stats(step=step, epoch=epoch, stats={"parameters": p_stats})


# -- Keras Callback (for standard model.fit()) ----------------------------

def _get_keras_callback_base():
    import tensorflow as tf
    return tf.keras.callbacks.Callback


def _make_prc_keras_callback():
    Base = _get_keras_callback_base()

    class PrcKerasCallback(Base):
        """
        Attach to `model.fit(callbacks=[...])` for zero-touch metric,
        epoch, and checkpoint logging. Does not include gradient/parameter
        stats — Keras's fit() callbacks don't expose per-batch gradients.
        For that level of detail, use a custom GradientTape loop with
        `gradient_stats()` / `parameter_stats()` instead (see module
        docstring).
        """

        def __init__(self, monitor, log_every_n_batches: int = 1,
                     checkpoint_metric: str = "val_loss", lower_is_better: bool = True):
            super().__init__()
            self.monitor = monitor
            self.log_every_n_batches = max(1, log_every_n_batches)
            self.checkpoint_metric = checkpoint_metric
            self.lower_is_better = lower_is_better
            self._step = 0
            self._current_epoch = 0
            self._best_metric: Optional[float] = None

        def on_epoch_begin(self, epoch, logs=None):
            self._current_epoch = epoch
            _safe_call(self.monitor.epoch_started, epoch)

        def on_train_batch_end(self, batch, logs=None):
            logs = logs or {}
            if self._step % self.log_every_n_batches == 0:
                metrics = _normalize_metric_names(
                    {k: float(v) for k, v in logs.items() if _is_scalar_number(v)}
                )
                lr = _current_learning_rate(self.model)
                if lr is not None:
                    metrics["learning_rate"] = lr
                _safe_call(self.monitor.log, step=self._step, epoch=self._current_epoch, **metrics)
            self._step += 1

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            metrics = _normalize_metric_names({k: float(v) for k, v in logs.items() if _is_scalar_number(v)})

            # Keras only reports val_* metrics at epoch end. Emit them as a
            # 'metric' event too (not just 'epoch_finished'), since that's
            # what prc's analytics detectors (overfitting, plateau, ...)
            # scan for train/val pairs.
            _safe_call(self.monitor.log, step=self._step, epoch=epoch, **metrics)
            _safe_call(self.monitor.epoch_finished, epoch, metrics)

            gpu = _safe_call(gpu_stats)
            if gpu:
                _safe_call(self.monitor.log_system_metrics, step=self._step, stats=gpu)

            metric_value = logs.get(self.checkpoint_metric)
            if metric_value is not None:
                is_better = (
                    self._best_metric is None
                    or (metric_value < self._best_metric if self.lower_is_better else metric_value > self._best_metric)
                )
                if is_better:
                    self._best_metric = float(metric_value)
                    _safe_call(
                        self.monitor.log_checkpoint,
                        step=self._step, epoch=epoch, path=f"epoch_{epoch}",
                        metrics=metrics,
                    )

        def on_train_end(self, logs=None):
            pass  # user calls monitor.finish() explicitly, matching the PyTorch example

    return PrcKerasCallback


def _is_scalar_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_metric_names(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Keras emits 'loss' / 'val_loss' by default, but prc's analytics
    detectors (overfitting, plateau, instability) look for 'train_loss' /
    'val_loss' - the same convention used by the PyTorch integration and
    the raw SDK examples. Rename so anomaly detection works out of the
    box, without discarding the original Keras-style keys.
    """
    out = dict(metrics)
    if "loss" in out and "train_loss" not in out:
        out["train_loss"] = out["loss"]
    return out


def _current_learning_rate(model) -> Optional[float]:
    try:
        import tensorflow as tf

        lr = model.optimizer.learning_rate
        if callable(lr):
            lr = lr(model.optimizer.iterations)
        return float(tf.keras.backend.get_value(lr))
    except Exception:
        return None


class _LazyCallback:
    """Defers building the real tf.keras.callbacks.Callback subclass until
    first use, so importing this module doesn't require tensorflow."""

    _cls = None

    def __call__(self, *args, **kwargs):
        if _LazyCallback._cls is None:
            _LazyCallback._cls = _make_prc_keras_callback()
        return _LazyCallback._cls(*args, **kwargs)


PrcKerasCallback = _LazyCallback()
