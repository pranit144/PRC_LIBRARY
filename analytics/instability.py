from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


def detect_instability(
    metrics: List[Dict[str, Any]],
    metric_key: str = "train_loss",
    window: int = 10,
    cv_threshold: float = 0.35,
) -> Optional[Dict[str, Any]]:
    """
    Flags large oscillations using coefficient of variation (stdev/mean)
    of the recent window as a simple, explainable instability signal.
    """
    values = [m[metric_key] for m in metrics if metric_key in m][-window:]
    if len(values) < max(4, window // 2):
        return None

    m = mean(values)
    if m == 0:
        return None
    cv = pstdev(values) / abs(m)

    if cv > cv_threshold:
        return {
            "type": "unstable_training",
            "severity": "high" if cv > cv_threshold * 2 else "medium",
            "confidence": round(min(0.9, 0.4 + cv), 2),
            "message": (
                f"'{metric_key}' is oscillating significantly over recent steps. "
                "This is consistent with unstable training, possibly related to "
                "learning rate or batch size."
            ),
        }
    return None
