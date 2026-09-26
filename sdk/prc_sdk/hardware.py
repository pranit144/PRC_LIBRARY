"""
Hardware telemetry collector for the prc SDK.

Runs as a lightweight background daemon thread that samples CPU, RAM,
and GPU statistics at a configurable interval and pushes them to the
Monitor via `log_system_metrics`.

Usage (automatic when you pass enable_hardware_monitoring=True to Monitor):

    monitor = Monitor(project="my-model", enable_hardware_monitoring=True)
    # ... training ...
    monitor.finish()  # sampler thread stops automatically

Usage (manual, if you need finer control):

    from prc_sdk.hardware import HardwareSampler

    sampler = HardwareSampler(monitor, interval_seconds=5)
    sampler.start()
    # ... training ...
    sampler.stop()

Fallback chain for GPU metrics:
  1. pynvml  -- gives true GPU compute utilisation % + temperature
  2. torch.cuda -- gives VRAM allocated/reserved if torch is installed
  3. (nothing) -- CPU/RAM still reported, GPU fields simply absent

The sampler never raises into user code; all errors are logged and
swallowed to preserve the fail-safe guarantee of the core SDK.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("prc.hardware")


# ---------------------------------------------------------------------------
# Low-level collectors
# ---------------------------------------------------------------------------

def _collect_cpu_ram() -> Dict[str, Any]:
    """Collect CPU utilisation and system RAM via psutil (optional dep)."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "cpu_utilization_pct": psutil.cpu_percent(interval=None),
            "ram_used_mb": vm.used / (1024 ** 2),
            "ram_total_mb": vm.total / (1024 ** 2),
            "ram_utilization_pct": vm.percent,
        }
    except ImportError:
        return {}
    except Exception:
        logger.debug("prc.hardware: cpu/ram collection failed", exc_info=True)
        return {}


def _collect_gpu_pynvml() -> Dict[str, Any]:
    """
    Try pynvml first -- gives true GPU compute utilisation % and temperature
    on NVIDIA hardware. Returns {} if pynvml is not available or fails.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        if n == 0:
            return {}

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")

        result: Dict[str, Any] = {
            "gpu_index": 0,
            "gpu_name": name,
            "gpu_count": n,
            "gpu_utilization_pct": util.gpu,
            "gpu_memory_used_mb": mem.used / (1024 ** 2),
            "gpu_memory_total_mb": mem.total / (1024 ** 2),
            "gpu_memory_utilization_pct": (mem.used / mem.total * 100) if mem.total else 0.0,
        }

        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            result["gpu_temperature_c"] = temp
        except Exception:
            pass

        return result

    except ImportError:
        return {}
    except Exception:
        logger.debug("prc.hardware: pynvml collection failed", exc_info=True)
        return {}


def _collect_gpu_torch() -> Dict[str, Any]:
    """
    Fallback GPU collector using torch.cuda. Only gives VRAM, not true
    compute utilisation. Returns {} if torch is not installed or no CUDA GPU.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        mem_alloc = torch.cuda.memory_allocated(idx)
        mem_reserved = torch.cuda.memory_reserved(idx)
        total = props.total_memory
        return {
            "gpu_index": idx,
            "gpu_name": props.name,
            "gpu_memory_allocated_mb": mem_alloc / (1024 ** 2),
            "gpu_memory_reserved_mb": mem_reserved / (1024 ** 2),
            "gpu_memory_total_mb": total / (1024 ** 2),
            "gpu_memory_utilization_pct": (mem_reserved / total * 100) if total else 0.0,
        }
    except ImportError:
        return {}
    except Exception:
        logger.debug("prc.hardware: torch.cuda collection failed", exc_info=True)
        return {}


def collect_hardware_snapshot() -> Dict[str, Any]:
    """
    Collect a full hardware snapshot: CPU+RAM plus GPU (pynvml preferred,
    torch.cuda fallback). Returns an empty dict if nothing is available.
    """
    stats: Dict[str, Any] = {}
    stats.update(_collect_cpu_ram())

    gpu = _collect_gpu_pynvml()
    if not gpu:
        gpu = _collect_gpu_torch()
    stats.update(gpu)

    return stats


# ---------------------------------------------------------------------------
# Background sampler thread
# ---------------------------------------------------------------------------

class HardwareSampler:
    """
    Background daemon thread that samples hardware metrics at a fixed
    interval and forwards them to a Monitor instance.

    The thread is a daemon so it never prevents the Python interpreter from
    exiting even if the user forgets to call stop().

    Args:
        monitor:           A prc_sdk.Monitor instance.
        interval_seconds:  How often to sample. Default: 5 s.
    """

    def __init__(self, monitor, interval_seconds: float = 5.0) -> None:
        self._monitor = monitor
        self._interval = max(0.5, interval_seconds)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background sampling thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="prc-hardware-sampler",
            daemon=True,
        )
        self._thread.start()
        logger.debug("prc.hardware: sampler started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        """
        Signal the sampler to stop and wait briefly for it to exit.
        Safe to call multiple times or even if start() was never called.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None
        logger.debug("prc.hardware: sampler stopped")

    def _run(self) -> None:
        """Main loop executed in the daemon thread."""
        step = 0
        while not self._stop_event.wait(timeout=self._interval):
            try:
                snapshot = collect_hardware_snapshot()
                if snapshot:
                    self._monitor.log_system_metrics(step=step, stats=snapshot)
                    step += 1
            except Exception:
                logger.debug("prc.hardware: sample tick failed", exc_info=True)

