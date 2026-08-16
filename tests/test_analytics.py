import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analytics import detect_overfitting, detect_plateau, detect_instability, detect_gradient_anomaly
from analytics.engine import AnalyticsEngine


def make_metrics(train_losses, val_losses):
    return [{"step": i, "epoch": 0, "train_loss": t, "val_loss": v}
            for i, (t, v) in enumerate(zip(train_losses, val_losses))]


def test_detect_overfitting_positive():
    train = [1.0 - i * 0.05 for i in range(12)]
    val = [0.9 - i * 0.01 if i < 6 else 0.84 + (i - 6) * 0.03 for i in range(12)]
    metrics = make_metrics(train, val)
    result = detect_overfitting(metrics)
    assert result is not None
    assert result["type"] == "overfitting"
    assert 0 < result["confidence"] <= 1


def test_detect_overfitting_negative_when_both_improving():
    train = [1.0 - i * 0.05 for i in range(12)]
    val = [0.9 - i * 0.04 for i in range(12)]
    metrics = make_metrics(train, val)
    assert detect_overfitting(metrics) is None


def test_detect_overfitting_insufficient_data():
    metrics = make_metrics([1.0, 0.9], [0.9, 0.8])
    assert detect_overfitting(metrics) is None


def test_detect_plateau_positive():
    values = [0.5, 0.499, 0.498, 0.4975, 0.497, 0.4968, 0.4966, 0.4965, 0.4964, 0.4963]
    metrics = [{"step": i, "val_loss": v} for i, v in enumerate(values)]
    result = detect_plateau(metrics, metric_key="val_loss", window=10)
    assert result is not None
    assert result["type"] == "plateau"


def test_detect_plateau_negative_when_improving():
    values = [1.0 - i * 0.1 for i in range(10)]
    metrics = [{"step": i, "val_loss": v} for i, v in enumerate(values)]
    assert detect_plateau(metrics, metric_key="val_loss", window=10) is None


def test_detect_instability_positive():
    values = [1.0, 0.2, 1.1, 0.1, 1.2, 0.15, 1.05, 0.3, 1.15, 0.2]
    metrics = [{"step": i, "train_loss": v} for i, v in enumerate(values)]
    result = detect_instability(metrics)
    assert result is not None
    assert result["type"] == "unstable_training"


def test_detect_instability_negative_when_smooth():
    values = [1.0 - i * 0.05 for i in range(10)]
    metrics = [{"step": i, "train_loss": v} for i, v in enumerate(values)]
    assert detect_instability(metrics) is None


def test_detect_gradient_anomaly_vanishing():
    history = [{"grad_norm_total": 1e-8} for _ in range(5)]
    result = detect_gradient_anomaly(history)
    assert result["type"] == "vanishing_gradients"


def test_detect_gradient_anomaly_exploding():
    history = [{"grad_norm_total": 1e6} for _ in range(5)]
    result = detect_gradient_anomaly(history)
    assert result["type"] == "exploding_gradients"


def test_detect_gradient_anomaly_normal():
    history = [{"grad_norm_total": 1.5} for _ in range(5)]
    assert detect_gradient_anomaly(history) is None


def test_analytics_engine_aggregates_detectors():
    train = [1.0 - i * 0.05 for i in range(12)]
    val = [0.9 - i * 0.01 if i < 6 else 0.84 + (i - 6) * 0.03 for i in range(12)]
    metrics = make_metrics(train, val)
    engine = AnalyticsEngine()
    results = engine.analyze(metrics)
    assert any(r["type"] == "overfitting" for r in results)
