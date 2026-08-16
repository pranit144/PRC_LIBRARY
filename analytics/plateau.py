from __future__ import annotations

from typing import Any, Dict, List, Optional


def detect_plateau(
    metrics: List[Dict[str, Any]],
    metric_key: str = "val_loss",
    window: int = 10,
    min_relative_improvement: float = 0.005,
) -> Optional[Dict[str, Any]]:
    """
    Detect insufficient improvement in `metric_key` over the last `window`
    observations. Assumes lower is better (loss-style metric); for
    accuracy-style metrics pass values already negated by the caller if
    "higher is better" plateau detection is needed.
    """
    values = [m[metric_key] for m in metrics if metric_key in m][-window:]
    if len(values) < window:
        return None

    best_before = min(values[:-1])
    best_now = min(values)
    improvement = (best_before - best_now) / (abs(best_before) + 1e-9)

    if improvement < min_relative_improvement:
        return {
            "type": "plateau",
            "severity": "low" if improvement > 0 else "medium",
            "confidence": round(min(0.9, 0.5 + (min_relative_improvement - improvement) * 20), 2),
            "message": (
                f"'{metric_key}' has shown little to no improvement over the last "
                f"{window} observations. This may indicate the model has reached a plateau."
            ),
        }
    return None
