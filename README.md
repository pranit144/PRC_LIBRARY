# prc

**See the training. Understand the model. Forecast what comes next.**

prc is a real-time AI/ML training observability, diagnosis, and
forecasting platform. Instead of watching raw training logs scroll by,
you get a live dashboard that shows what your model is doing right now,
explains unusual behavior in plain language, and forecasts where
training is headed — always with clearly labeled uncertainty, never as
a guarantee.

## 1. What prc is

Four things, in order:

1. **Observe** — live metrics, gradients, activations, hardware usage
2. **Understand** — plain-language explanations of what's happening and why
3. **Detect** — deterministic anomaly detection (overfitting, plateau,
   vanishing/exploding gradients, instability)
4. **Forecast** — projected final metrics with explicit confidence and
   uncertainty bounds

## 2. Why it exists

Training logs are dense and easy to stare at without actually
understanding. prc's job is to turn that stream of numbers into
something you can reason about — the same way an oscilloscope turns a
voltage into a waveform you can read at a glance.

## 3. Architecture

```
PyTorch training script
        │
        ▼
   prc SDK (prc_sdk)          local buffering, fail-safe logging
        │
        ▼
   Event protocol             versioned, framework-independent JSON events
        │
        ├──► Local storage (JSONL buffer, always written first)
        │
        └──► prc server (FastAPI)
                  │
                  ├──► SQLite storage (behind a Storage abstraction —
                  │     PostgreSQL can be swapped in later)
                  │
                  ├──► Analytics engine (deterministic detectors)
                  │
                  ├──► Forecast engine (explainable trend baseline,
                  │     behind a ForecastEngine abstraction)
                  │
                  ├──► Assistant (deterministic Q&A, LLM-optional)
                  │
                  └──► WebSocket stream ──► React dashboard
```

Key design principles (see `AGENTS.md`-style notes below if you're
extending this):

- The SDK never crashes user training code — every public method
  catches and logs its own errors.
- Storage, forecasting, and the assistant are all behind small
  abstract interfaces so implementations can be swapped without
  touching callers.
- Anomalies and forecasts are always phrased with uncertainty
  ("may indicate", "consistent with") and carry a confidence score.
  Nothing is presented as a guaranteed fact.

## 4. Installation

Requires Python 3.11+ and Node 20+ (for the dashboard).

```bash
git clone <this repo>
cd prc
pip install -e ".[dev]"          # SDK + server + storage + analytics + forecasting + assistant
pip install -e ".[pytorch]"      # optional, for the PyTorch integration
```

## 5. Quick start

**Single-port mode (recommended — works locally, in Colab, Kaggle, or on a remote box):**

```bash
cd dashboard && npm install && npm run build && cd ..
uvicorn server.main:app --reload
```

That's it — one process, one port (`:8000`), serving both the API and
the dashboard. Then just run any training script using the SDK; the
moment `Monitor(...)` is created it prints a live link:

```python
from prc_sdk import Monitor

monitor = Monitor(project="my-model", run_name="experiment-01")
# 🔴 Live monitoring: http://localhost:8000/runs/run_abc123
```

The SDK detects where it's running and adjusts the link automatically:

| Environment | Behavior |
|---|---|
| Local script / plain Jupyter | prints `http://localhost:8000/runs/{run_id}` |
| Google Colab | auto-detects and prints a working proxied URL via Colab's port-proxy, plus renders a clickable link inline in the notebook |
| Kaggle | prints the local URL with a note that Kaggle doesn't support automatic arbitrary-port proxying (you may need your own tunnel) |
| Remote / SSH session | prints the local URL plus a one-line `ssh -L` port-forward hint |

The dashboard's browser tab also switches to `● Live Monitoring —
{run_name}` while a run is active, and back to normal once it finishes.

**Dev mode (separate dashboard dev server with hot reload):**

```bash
uvicorn server.main:app --reload          # terminal 1, API on :8000
cd dashboard && npm run dev                # terminal 2, dashboard on :5173
```

In this mode set `VITE_PRC_SERVER_URL=http://localhost:8000` if the
dashboard needs to reach an API on a different host/port than its own
origin.

**Send it some training data:**

```python
from prc_sdk import Monitor

monitor = Monitor(project="my-model", run_name="experiment-01")
for step in range(100):
    monitor.log(step=step, epoch=step // 20, train_loss=..., val_loss=...)
monitor.finish()
```

Or run the full working example: `python examples/mnist/train.py` (see
`examples/mnist/README.md`).

## 6. PyTorch example

```python
from prc_sdk import Monitor
from prc_sdk.pytorch import TorchMonitorHook

monitor = Monitor(project="image-classifier", run_name="experiment-01")
hook = TorchMonitorHook(monitor, model, log_every_n_steps=20)

for epoch in range(10):
    for step, batch in enumerate(train_loader):
        loss = train_step(batch)
        loss.backward()
        hook.maybe_log(step, epoch)     # gradient / parameter / GPU stats
        optimizer.step()
        monitor.log(step=step, epoch=epoch, train_loss=float(loss),
                    learning_rate=optimizer.param_groups[0]["lr"])
monitor.finish()
```

See `examples/mnist/train.py` for a complete, runnable script.

## 7. Dashboard

The dashboard is served from the same port as the API in production
mode (see Quick Start), which is what makes the live-monitoring link
work anywhere — including through Colab's or Kaggle's proxy layer,
which can only cleanly proxy a single port.

The run page shows:

- **Header** — project, run name, live/done status, epoch/step, a live
  "pulse" sparkline of validation loss
- **Metric cards** — train/val loss, accuracy, learning rate, GPU memory
- **Training chart** — observed curves in amber, forecasted continuation
  in dashed cyan, so predicted values are never mistaken for observed ones
- **Timeline** — chronological events, severity-colored
- **Model health** — a one-line beginner summary with an expandable
  detail view (progressive disclosure, so beginners aren't overwhelmed)
- **Forecast panel** — projected final metric, confidence, uncertainty
  range, always with a disclaimer that it's an estimate
- **Assistant** — ask questions like "should I stop training?" and get
  a deterministic, explainable answer grounded in detected anomalies
  and the current forecast

## 8. Forecasting

`forecasting.SimpleTrendForecastEngine` is an explainable statistical
baseline: it fits a linear trend to the recent metric history and
extrapolates forward by the same span already observed, with an
uncertainty band derived from how well that trend actually fits the
recent data. If there isn't enough history, it returns
`status: "insufficient_data"` rather than fabricating a number.

The engine is defined behind the `ForecastEngine` abstract interface so
a more sophisticated model can be dropped in later without touching the
API or dashboard.

## 9. Anomaly detection

Four deterministic detectors ship in the MVP (`analytics/`):

| Detector | Signal |
|---|---|
| Overfitting | train loss trending down, val loss trending up |
| Plateau | insufficient relative improvement over a window |
| Gradient anomaly | average gradient norm outside a normal range |
| Instability | high coefficient of variation in recent loss values |

All results carry a `severity`, a `confidence` (0–1), and a plain-
language `message` that avoids asserting certainty.

## 10. Roadmap

- **v0.1 (this repo)** — PyTorch SDK, live dashboard, deterministic
  anomaly detection, baseline forecasting, deterministic assistant
- **v0.2** — Keras / Hugging Face integrations, better forecasting,
  experiment comparison
- **v0.3** — LLM-backed assistant, counterfactual experiment
  forecasting, dataset analysis
- **v0.4** — example-level debugging, historical run intelligence, team
  collaboration
- **v1.0** — a complete AI training intelligence platform

## 11. Development setup

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The dashboard has its own toolchain:

```bash
cd dashboard
npm install
npm run dev      # dev server
npm run build    # typecheck + production build
```

Or run everything with Docker:

```bash
docker compose up --build
```

## 12. Contributing

This is an early-stage MVP. Useful contributions right now:

- Additional framework integrations (Keras, Hugging Face)
- More detectors in `analytics/`
- A PostgreSQL implementation of `storage.Storage`
- An LLM-backed implementation of `assistant.TrainingAssistant`

Please add tests for new functionality under `tests/` — see the
existing suite for the patterns used (fixtures, `TestClient` for API
tests, etc).

## Project layout

```
prc/
├── sdk/prc_sdk/        Python SDK (Monitor, event schema, PyTorch hooks)
├── server/              FastAPI app: REST + WebSocket
├── storage/              Storage abstraction + SQLite implementation
├── analytics/            Deterministic anomaly detectors
├── forecasting/          Forecast engine abstraction + baseline impl
├── assistant/            Deterministic training assistant
├── dashboard/            React + TypeScript frontend
├── examples/mnist/       End-to-end runnable example
├── tests/                pytest suite (36 tests)
├── pyproject.toml
├── docker-compose.yml
└── README.md
```
