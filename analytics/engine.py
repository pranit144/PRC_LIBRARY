from __future__ import annotations

from typing import Any, Dict, List, Optional

from .overfitting import detect_overfitting
from .plateau import detect_plateau
from .instability import detect_instability
from .gradient_anomaly import detect_gradient_anomaly


class AnalyticsEngine:
    """Runs all deterministic detectors over a run's metric/gradient
    history and returns any anomalies found. Kept dependency-free and
    explainable by design (Phase 7 baseline) — no black-box ML here."""

    def analyze(
        self,
        metrics: List[Dict[str, Any]],
        gradient_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        results = []

        overfit = detect_overfitting(metrics)
        if overfit:
            results.append(overfit)

        plateau = detect_plateau(metrics)
        if plateau:
            results.append(plateau)

        instability = detect_instability(metrics)
        if instability:
            results.append(instability)

        if gradient_history:
            grad_anomaly = detect_gradient_anomaly(gradient_history)
            if grad_anomaly:
                results.append(grad_anomaly)

        return results
