import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forecasting import SimpleTrendForecastEngine


def make_history(values):
    return [{"step": i, "val_loss": v} for i, v in enumerate(values)]


def test_insufficient_data_returns_explicit_status():
    engine = SimpleTrendForecastEngine()
    history = make_history([1.0, 0.9])
    result = engine.forecast(history, "val_loss")
    assert result["status"] == "insufficient_data"
    assert "message" in result


def test_forecast_with_enough_data_returns_bounds_and_confidence():
    engine = SimpleTrendForecastEngine()
    values = [1.0 - i * 0.02 for i in range(20)]
    history = make_history(values)
    result = engine.forecast(history, "val_loss")
    assert result["status"] == "ok"
    assert result["lower_bound"] <= result["predicted_final"] <= result["upper_bound"]
    assert 0 <= result["confidence"] <= 1
    assert "disclaimer" in result


def test_forecast_never_claims_certainty():
    engine = SimpleTrendForecastEngine()
    values = [1.0 - i * 0.05 for i in range(30)]
    history = make_history(values)
    result = engine.forecast(history, "val_loss")
    assert result["confidence"] < 1.0


def test_accuracy_metric_bounded_between_0_and_1():
    engine = SimpleTrendForecastEngine()
    history = [{"step": i, "accuracy": min(0.99, 0.5 + i * 0.03)} for i in range(20)]
    result = engine.forecast(history, "accuracy")
    assert 0.0 <= result["predicted_final"] <= 1.0
    assert 0.0 <= result["lower_bound"] <= 1.0
    assert 0.0 <= result["upper_bound"] <= 1.0


def test_loss_metric_not_negative():
    engine = SimpleTrendForecastEngine()
    values = [1.0 - i * 0.06 for i in range(20)]  # would go negative if unclamped
    history = make_history(values)
    result = engine.forecast(history, "val_loss")
    assert result["predicted_final"] >= 0.0
