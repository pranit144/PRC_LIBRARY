"""
Deterministic overfitting detector.

Pattern: training loss trending down while validation loss trends up
over a recent window. Never claims certainty — output is always phrased
with "may", "consistent with", etc., and carries a confidence score.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _trend(values: List[float]) -> float:
    """Simple slope estimate via least-squares over index -> value."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def detect_overfitting(metrics: List[Dict[str, Any]], window: int = 10) -> Optional[Dict[str, Any]]:
    """
    metrics: list of dicts with at least 'train_loss' and 'val_loss', in
    chronological order.
    """
    recent = [m for m in metrics if "train_loss" in m and "val_loss" in m][-window:]
    if len(recent) < max(4, window // 2):
        return None

    train_losses = [m["train_loss"] for m in recent]
    val_losses = [m["val_loss"] for m in recent]

    train_slope = _trend(train_losses)
    val_slope = _trend(val_losses)

    if train_slope < 0 and val_slope > 0:
        # Normalize slopes into a rough confidence score.
        magnitude = min(abs(train_slope), abs(val_slope))
        spread = (max(val_losses) - min(val_losses)) or 1e-9
        confidence = max(0.5, min(0.95, magnitude / spread * 5))

        severity = "high" if val_slope > abs(train_slope) else "medium"

        return {
            "type": "overfitting",
            "severity": severity,
            "confidence": round(confidence, 2),
            "message": (
                "Validation loss has been increasing while training loss "
                "continues to decrease. This pattern may indicate overfitting."
            ),
        }
    return None
