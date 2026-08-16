from __future__ import annotations

from typing import Any, Dict, List, Optional


def detect_gradient_anomaly(
    gradient_history: List[Dict[str, Any]],
    vanish_threshold: float = 1e-6,
    explode_threshold: float = 1e3,
    window: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    gradient_history: list of dicts with a 'grad_norm_total' key (as
    produced by prc_sdk.pytorch.gradient_stats), chronological order.
    """
    recent = [g["grad_norm_total"] for g in gradient_history if "grad_norm_total" in g][-window:]
    if not recent:
        return None

    avg = sum(recent) / len(recent)

    if avg < vanish_threshold:
        return {
            "type": "vanishing_gradients",
            "severity": "high",
            "confidence": 0.8,
            "message": (
                "Gradient magnitude has dropped to a very small value across recent steps. "
                "This may indicate vanishing gradients."
            ),
        }
    if avg > explode_threshold:
        return {
            "type": "exploding_gradients",
            "severity": "high",
            "confidence": 0.8,
            "message": (
                "Gradient magnitude has grown very large across recent steps. "
                "This may indicate exploding gradients."
            ),
        }
    return None
