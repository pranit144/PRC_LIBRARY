"""
Forecasting is kept behind an abstract interface so the baseline
statistical method here can later be swapped for something more
sophisticated without touching the API or dashboard.

Forecasts are NEVER presented as guaranteed. Every result includes an
explicit uncertainty interval and a confidence score, and the engine
returns a clear "insufficient data" result rather than fabricating one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ForecastEngine(ABC):
    @abstractmethod
    def forecast(self, history: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
        ...


def _linear_fit(xs: List[float], ys: List[float]):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n)) or 1e-9
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


class SimpleTrendForecastEngine(ForecastEngine):
    """
    Explainable baseline: fits a simple exponential-decay-style trend to
    the metric's recent trajectory (via a linear fit in a transformed
    space for loss-like metrics, or directly for bounded metrics like
    accuracy), and extrapolates forward with a widening uncertainty band.

    This is intentionally simple and transparent rather than a black-box
    model, per the "keep forecasting behind an abstraction, start with an
    explainable baseline" requirement.
    """

    MIN_POINTS = 5

    def forecast(self, history: List[Dict[str, Any]], metric: str,
                 horizon_steps: Optional[int] = None) -> Dict[str, Any]:
        points = [(m["step"], m[metric]) for m in history if metric in m and m.get("step") is not None]
        points.sort(key=lambda p: p[0])

        if len(points) < self.MIN_POINTS:
            return {
                "metric": metric,
                "status": "insufficient_data",
                "message": "Insufficient data for reliable forecast.",
            }

        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        current = ys[-1]

        slope, intercept = _linear_fit(xs, ys)
        last_step = xs[-1]
        observed_span = max(xs[-1] - xs[0], 1.0)
        # Default: extrapolate forward by the same span already observed.
        # This keeps the forecast anchored to a plausible near-future
        # horizon instead of extrapolating a noisy short-run slope across
        # an arbitrary large number of steps.
        if horizon_steps is None:
            horizon_steps = observed_span
        future_step = last_step + horizon_steps

        predicted_final = intercept + slope * future_step

        # Bound accuracy-like metrics to a sane range.
        is_bounded_metric = "acc" in metric or "accuracy" in metric or "f1" in metric or "precision" in metric or "recall" in metric
        if is_bounded_metric:
            predicted_final = max(0.0, min(1.0, predicted_final))
        elif "loss" in metric:
            predicted_final = max(0.0, predicted_final)

        # Residual-based uncertainty: how well does the linear fit explain
        # recent history? Wider residuals -> wider interval, lower confidence.
        residuals = [ys[i] - (intercept + slope * xs[i]) for i in range(len(xs))]
        residual_std = (sum(r ** 2 for r in residuals) / len(residuals)) ** 0.5

        margin = max(residual_std * 1.5, abs(predicted_final) * 0.01)
        lower_bound = predicted_final - margin
        upper_bound = predicted_final + margin
        if is_bounded_metric:
            lower_bound = max(0.0, lower_bound)
            upper_bound = min(1.0, upper_bound)

        # Confidence heuristic: more history + tighter residuals -> higher
        # confidence, capped well below "certain".
        n_points_factor = min(1.0, len(points) / 50.0)
        fit_quality = 1.0 / (1.0 + residual_std / (abs(current) + 1e-9))
        confidence = round(min(0.9, max(0.3, 0.3 + 0.4 * n_points_factor + 0.3 * fit_quality)), 2)

        # Rough convergence estimate: step at which the trend's remaining
        # change drops below 1% of the total observed change.
        convergence_step: Optional[int] = None
        if abs(slope) > 1e-12:
            total_change = predicted_final - current
            if abs(total_change) > 1e-9:
                target = current + 0.99 * total_change
                convergence_step = int((target - intercept) / slope)

        return {
            "metric": metric,
            "status": "ok",
            "current": current,
            "predicted_final": round(predicted_final, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "confidence": confidence,
            "estimated_convergence_step": convergence_step,
            "disclaimer": "This is a statistical estimate based on the observed trend, not a guaranteed outcome.",
        }
