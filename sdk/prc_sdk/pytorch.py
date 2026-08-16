"""
Optional PyTorch integration. Import lazily so the core SDK never requires
torch to be installed.

    from prc_sdk import Monitor
    from prc_sdk.pytorch import log_gradient_stats, log_gpu_stats

    monitor = Monitor(project="mnist")
    ...
    loss.backward()
    log_gradient_stats(monitor, model, step=step, epoch=epoch)
    optimizer.step()
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("prc")


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("prc.pytorch: monitoring call failed (non-fatal)")
        return None


def gradient_stats(model) -> Dict[str, Any]:
    """Compute per-run gradient magnitude/norm summary for a torch model.
    Safe to call even if some parameters have no gradient yet."""
    import torch

    norms = []
    max_abs = 0.0
    min_abs = float("inf")
    n_params_with_grad = 0

    for _, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n_params_with_grad += 1
        norm = float(torch.norm(g).item())
        norms.append(norm)
        gabs = g.abs()
        if gabs.numel() > 0:
            max_abs = max(max_abs, float(gabs.max().item()))
            min_abs = min(min_abs, float(gabs.min().item()))

    if not norms:
        return {"num_params_with_grad": 0}

    total_norm = sum(n ** 2 for n in norms) ** 0.5
    return {
        "num_params_with_grad": n_params_with_grad,
        "grad_norm_total": total_norm,
        "grad_norm_mean": sum(norms) / len(norms),
        "grad_norm_max": max(norms),
        "grad_norm_min": min(norms),
        "grad_abs_max": max_abs,
        "grad_abs_min": min_abs if min_abs != float("inf") else 0.0,
    }


def parameter_stats(model) -> Dict[str, Any]:
    import torch

    norms = []
    for _, p in model.named_parameters():
        norms.append(float(torch.norm(p.detach()).item()))
    if not norms:
        return {}
    return {
        "param_norm_total": sum(n ** 2 for n in norms) ** 0.5,
        "param_norm_mean": sum(norms) / len(norms),
        "param_norm_max": max(norms),
        "param_norm_min": min(norms),
    }


def gpu_stats() -> Dict[str, Any]:
    """Best-effort GPU utilization/memory. Returns {} if no GPU / torch missing."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        idx = torch.cuda.current_device()
        mem_alloc = torch.cuda.memory_allocated(idx)
        mem_reserved = torch.cuda.memory_reserved(idx)
        total = torch.cuda.get_device_properties(idx).total_memory
        return {
            "gpu_index": idx,
            "gpu_name": torch.cuda.get_device_properties(idx).name,
            "gpu_memory_allocated_mb": mem_alloc / (1024 ** 2),
            "gpu_memory_reserved_mb": mem_reserved / (1024 ** 2),
            "gpu_memory_total_mb": total / (1024 ** 2),
            "gpu_memory_utilization_pct": (mem_reserved / total * 100) if total else 0.0,
        }
    except Exception:
        return {}


def log_gradient_stats(monitor, model, step: int, epoch: int) -> None:
    stats = _safe_call(gradient_stats, model)
    if stats:
        monitor.log_gradient_stats(step=step, epoch=epoch, stats=stats)


def log_parameter_stats(monitor, model, step: int, epoch: int) -> None:
    stats = _safe_call(parameter_stats, model)
    if stats:
        monitor.log_activation_stats(step=step, epoch=epoch, stats={"parameters": stats})


def log_gpu_stats(monitor, step: int) -> None:
    stats = _safe_call(gpu_stats)
    if stats:
        monitor.log_system_metrics(step=step, stats=stats)


class TorchMonitorHook:
    """Convenience wrapper bundling gradient/parameter/GPU logging so users
    don't need to remember to call each function individually."""

    def __init__(self, monitor, model, log_every_n_steps: int = 50):
        self.monitor = monitor
        self.model = model
        self.log_every_n_steps = log_every_n_steps

    def maybe_log(self, step: int, epoch: int) -> None:
        if step % self.log_every_n_steps != 0:
            return
        log_gradient_stats(self.monitor, self.model, step, epoch)
        log_parameter_stats(self.monitor, self.model, step, epoch)
        log_gpu_stats(self.monitor, step)
