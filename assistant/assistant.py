"""
Training assistant abstraction (Phase 9).

For the MVP this is a deterministic explanation engine — no external LLM
required. Later, an LLM adapter can consume the same structured context
(metrics, anomalies, forecast, timeline, hyperparameters) to produce
richer natural-language answers, without changing this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TrainingAssistant(ABC):
    @abstractmethod
    def explain_run(self, run: Dict[str, Any], metrics: List[Dict[str, Any]],
                     anomalies: List[Dict[str, Any]], forecast: Optional[Dict[str, Any]]) -> str:
        ...

    @abstractmethod
    def best_epoch(self, metrics: List[Dict[str, Any]], metric_key: str = "val_loss",
                    lower_is_better: bool = True) -> Optional[Dict[str, Any]]:
        ...


class DeterministicAssistant(TrainingAssistant):
    """Rule-based explanations built directly from detected anomalies and
    forecasts — transparent and cheap, no API key required."""

    def best_epoch(self, metrics: List[Dict[str, Any]], metric_key: str = "val_loss",
                    lower_is_better: bool = True) -> Optional[Dict[str, Any]]:
        candidates = [m for m in metrics if metric_key in m]
        if not candidates:
            return None
        key_fn = (lambda m: m[metric_key]) if lower_is_better else (lambda m: -m[metric_key])
        return min(candidates, key=key_fn)

    def explain_run(self, run: Dict[str, Any], metrics: List[Dict[str, Any]],
                     anomalies: List[Dict[str, Any]], forecast: Optional[Dict[str, Any]]) -> str:
        lines = []
        best = self.best_epoch(metrics)
        if best:
            lines.append(f"Best validation checkpoint so far: epoch {best.get('epoch')} (step {best.get('step')}).")

        if anomalies:
            top = anomalies[-1]
            lines.append(top.get("message", ""))
        else:
            lines.append("No significant anomalies detected in this run so far.")

        if forecast and forecast.get("status") == "ok":
            lines.append(
                f"Forecast: {forecast['metric']} is projected to reach "
                f"{forecast['predicted_final']} (range {forecast['lower_bound']}–{forecast['upper_bound']}, "
                f"confidence {int(forecast['confidence'] * 100)}%)."
            )
        elif forecast and forecast.get("status") == "insufficient_data":
            lines.append("Not enough data yet for a reliable forecast.")

        return " ".join(lines)

    def answer(self, question: str, run: Dict[str, Any], metrics: List[Dict[str, Any]],
               anomalies: List[Dict[str, Any]], forecast: Optional[Dict[str, Any]]) -> str:
        """Very small deterministic keyword router for common questions.
        This is a placeholder baseline — swap in an LLM adapter for
        open-ended natural-language Q&A."""
        q = question.lower()

        if "best epoch" in q or "best checkpoint" in q:
            best = self.best_epoch(metrics)
            if not best:
                return "No validation metrics recorded yet."
            return f"The best epoch so far is epoch {best.get('epoch')} (step {best.get('step')})."

        if "stop" in q:
            overfit = next((a for a in anomalies if a.get("type") == "overfitting"), None)
            if overfit:
                return (
                    "There are signs consistent with overfitting. You may want to consider "
                    "early stopping around the best validation checkpoint, though this is a "
                    "judgment call, not a guarantee."
                )
            return "No strong signal to stop yet based on current detectors."

        if "unstable" in q or "instability" in q:
            unstable = next((a for a in anomalies if a.get("type") == "unstable_training"), None)
            return unstable["message"] if unstable else "No instability detected in the recent window."

        if "gpu" in q:
            return "GPU utilization is tracked under system metrics; check the hardware panel for current values."

        if "next" in q or "try" in q:
            suggestions = []
            if any(a.get("type") == "overfitting" for a in anomalies):
                suggestions += ["early stopping", "more regularization", "data augmentation", "reduce model capacity"]
            if any(a.get("type") == "plateau" for a in anomalies):
                suggestions += ["adjust learning rate", "try a learning rate schedule"]
            if not suggestions:
                return "Training looks healthy; no specific interventions suggested right now."
            return "Possible things to try: " + ", ".join(dict.fromkeys(suggestions)) + "."

        return self.explain_run(run, metrics, anomalies, forecast)
