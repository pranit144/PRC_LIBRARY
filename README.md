## PRC

<img width="44" height="44" alt="image" src="https://github.com/user-attachments/assets/799df09c-cf3d-4e7c-bb0d-c2e200fc8a2b" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

**See the training. Understand the model. Forecast what comes next.**

prc is a real-time AI/ML training observability, diagnosis, and
forecasting platform. Instead of watching raw training logs scroll by,
you get a live dashboard that shows what your model is doing right now,
explains unusual behavior in plain language, and forecasts where
training is headed — always with clearly labeled uncertainty, never as
a guarantee.

## Contents

1. [What prc is](#1-what-prc-is)
2. [Why it exists](#2-why-it-exists)
3. [Architecture](#3-architecture)
4. [Installation](#4-installation)
5. [Quick start](#5-quick-start)
6. [PyTorch example](#6-pytorch-example)
7. [TensorFlow / Keras example](#7-tensorflow--keras-example)
8. [Hugging Face Trainer example](#8-hugging-face-trainer-example)
9. [PyTorch Lightning example](#9-pytorch-lightning-example)
10. [Hardware telemetry](#10-hardware-telemetry)
11. [Dashboard](#11-dashboard)
12. [Forecasting](#12-forecasting)
13. [Anomaly detection](#13-anomaly-detection)
14. [Roadmap](#14-roadmap)
15. [Development setup](#15-development-setup)
16. [Contributing](#16-contributing)
17. [Project layout](#project-layout)

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
PyTorch / Keras training script
        │
        ▼
   prc SDK (prc_sdk)          local buffering, fail-safe logging
        │
        ▼
   Event protocol             versioned, framework-independent JSON events
        │
        ├──► Local storage (JSONL buffer, always written first)
        │
        └──► prc server (FastAPI, single port serves API + dashboard)
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

Key design principles — see [`docs/KT_NOTES.md`](docs/KT_NOTES.md) for
a full code-level walkthrough if you're extending this:

- The SDK never crashes user training code — every public method
  catches and logs its own errors.
- Storage, forecasting, and the assistant are all behind small
  abstract interfaces so implementations can be swapped without
  touching callers.
- Anomalies and forecasts are always phrased with uncertainty
  ("may indicate", "consistent with") and carry a confidence score.
  Nothing is presented as a guaranteed fact.
- The API and dashboard are served from a single port, so live-monitoring
  links work the same way locally, in Colab, in Kaggle, or behind SSH.

## 4. Installation

Requires Python 3.11+ and Node 20+ (for building the dashboard).

```bash
git clone https://github.com/YOUR_USERNAME/prc.git
cd prc
pip install -e ".[dev]"            # SDK + server + storage + analytics + forecasting + assistant
pip install -e ".[pytorch]"        # optional — PyTorch hooks (TorchMonitorHook)
pip install -e ".[tensorflow]"     # optional — TensorFlow/Keras callback
pip install -e ".[transformers]"   # optional — Hugging Face TrainerCallback
pip install -e ".[lightning]"      # optional — PyTorch Lightning callback
pip install -e ".[hardware]"       # optional — background CPU/RAM/GPU telemetry (psutil + pynvml)
pip install -e ".[tunnel]"         # optional — ngrok fallback for Kaggle/remote sessions
pip install -e ".[all]"            # everything above at once
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
| Kaggle | tries an ngrok tunnel if `pip install pyngrok` + `NGROK_AUTHTOKEN` are set; otherwise prints the local URL with a note that Kaggle doesn't proxy arbitrary ports |
| Remote / SSH session | tries an ngrok tunnel if available; otherwise prints the local URL plus a one-line `ssh -L` port-forward hint |

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

Or run one of the full working examples:

```bash
python examples/mnist/train.py          # PyTorch (see examples/mnist/README.md)
python examples/keras_mnist/train.py    # TensorFlow/Keras
```

## 6. PyTorch example

PyTorch has no built-in training-loop hook system, so the integration
is a set of helper functions you call yourself, plus a small wrapper
for convenience:

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

## 7. TensorFlow / Keras example

Keras's `model.fit()` already has a callback system, so this
integration is zero-touch — no changes to your training loop:

```python
from prc_sdk import Monitor
from prc_sdk.tensorflow import PrcKerasCallback

monitor = Monitor(project="image-classifier", run_name="experiment-01")

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    callbacks=[PrcKerasCallback(monitor, log_every_n_batches=5)],
)
monitor.finish()
```

`PrcKerasCallback` gives you metrics, epoch/checkpoint events, and
best-effort GPU stats automatically. It can't expose raw gradients
(Keras's `fit()` callbacks don't have access to them) — for that level
of detail, drop down to a custom `tf.GradientTape` loop and call the
module-level `gradient_stats()` / `parameter_stats()` helpers yourself,
mirroring the PyTorch path. See `prc_sdk/tensorflow.py` for details.

See `examples/keras_mnist/train.py` for a complete, runnable script.

## 8. Hugging Face Trainer example

Works with any model fine-tuned via `transformers.Trainer` — LLMs (Llama,
Mistral, Qwen, Phi), encoder models (BERT, RoBERTa), vision transformers,
diffusion pipelines built on Trainer, and more.

```python
from prc_sdk import Monitor
from prc_sdk.transformers import PrcHfCallback

monitor = Monitor(project="llama-finetune", run_name="experiment-01")
# 🔴 Live monitoring: http://localhost:8000/runs/run_abc123

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[PrcHfCallback(monitor)],
)
trainer.train()
monitor.finish()
```

`PrcHfCallback` automatically:
- Extracts `TrainingArguments` (learning rate, batch size, scheduler, optimizer,
  warmup, seed, …) as the run config.
- Maps HF's `loss` / `eval_loss` to prc's canonical `train_loss` / `val_loss`
  keys so anomaly detectors work out of the box.
- Filters out non-scalar log entries (`total_flos`, string fields, …).
- Logs checkpoint paths on every `on_save` event.
- **In DDP / FSDP / DeepSpeed multi-GPU runs, only rank 0 emits events** —
  no duplicate data.

Install the extra dependency:
```bash
pip install -e ".[transformers]"
# or: pip install transformers>=4.35
```

## 9. PyTorch Lightning example

Works with modern Lightning (`lightning.pytorch`) and legacy
`pytorch_lightning` — the callback detects which is installed automatically.

```python
import lightning.pytorch as pl
from prc_sdk import Monitor
from prc_sdk.lightning import PrcLightningCallback

monitor = Monitor(project="my-model", run_name="experiment-01")
# 🔴 Live monitoring: http://localhost:8000/runs/run_abc123

trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[PrcLightningCallback(monitor)],
)
trainer.fit(model, train_loader, val_loader)
monitor.finish()
```

`PrcLightningCallback` automatically:
- Harvests `trainer.callback_metrics` (train loss, val loss, any logged metric)
  at every batch and validation epoch end.
- Extracts Trainer settings (`max_epochs`, `precision`, `strategy`, …) and
  `LightningModule.hparams` as the run config.
- Calls `monitor.finish(status="failed")` automatically if an exception
  terminates training.
- Logs checkpoint paths via `on_save_checkpoint`.
- **Only the rank-0 process emits events** in multi-GPU / multi-node runs.
- Accepts a `log_every_n_steps` argument to thin out step-level logging for
  very fast inner loops.

Install the extra dependency:
```bash
pip install -e ".[lightning]"
# or: pip install lightning>=2.0
```

## 10. Hardware telemetry

prc can automatically collect CPU, RAM, and GPU metrics in a background
daemon thread so you can diagnose hardware bottlenecks and memory leaks
without changing your training loop.

```python
from prc_sdk import Monitor

monitor = Monitor(
    project="my-model",
    run_name="experiment-01",
    enable_hardware_monitoring=True,   # default: False
    hardware_interval_seconds=5.0,    # sample every 5 s
)
# ... training ...
monitor.finish()  # sampler thread stops cleanly here
```

Metrics collected and what provides them:

| Metric | Provider |
|---|---|
| `cpu_utilization_pct` | `psutil` |
| `ram_used_mb`, `ram_total_mb`, `ram_utilization_pct` | `psutil` |
| `gpu_utilization_pct` | `pynvml` (NVIDIA) |
| `gpu_memory_used_mb`, `gpu_memory_total_mb`, `gpu_memory_utilization_pct` | `pynvml` or `torch.cuda` |
| `gpu_temperature_c` | `pynvml` (where supported) |
| `gpu_name`, `gpu_index`, `gpu_count` | `pynvml` or `torch.cuda` |

**Fallback chain:** `pynvml` (full compute % + temperature) → `torch.cuda`
(VRAM only) → nothing (CPU/RAM still reported). The sampler never crashes
training regardless of what is or isn't installed.

You can also use it manually for finer control:

```python
from prc_sdk.hardware import HardwareSampler, collect_hardware_snapshot

# One-shot snapshot
print(collect_hardware_snapshot())
# {'cpu_utilization_pct': 12.3, 'ram_used_mb': 4096.0, ...}

# Or run the background thread yourself
sampler = HardwareSampler(monitor, interval_seconds=3)
sampler.start()
# ... training ...
sampler.stop()
```

Install the extra dependencies:
```bash
pip install -e ".[hardware]"
# or: pip install psutil>=5.9 pynvml>=11.0
```

## 11. Dashboard

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

## 12. Forecasting

`forecasting.SimpleTrendForecastEngine` is an explainable statistical
baseline: it fits a linear trend to the recent metric history and
extrapolates forward by the same span already observed, with an
uncertainty band derived from how well that trend actually fits the
recent data. If there isn't enough history, it returns
`status: "insufficient_data"` rather than fabricating a number.

The engine is defined behind the `ForecastEngine` abstract interface so
a more sophisticated model can be dropped in later without touching the
API or dashboard.

## 13. Anomaly detection

Four deterministic detectors ship in the MVP (`analytics/`):

| Detector | Signal |
|---|---|
| Overfitting | train loss trending down, val loss trending up |
| Plateau | insufficient relative improvement over a window |
| Gradient anomaly | average gradient norm outside a normal range |
| Instability | high coefficient of variation in recent loss values |

All results carry a `severity`, a `confidence` (0–1), and a plain-
language `message` that avoids asserting certainty.

## 14. Roadmap

- **v0.1 (this repo)** — PyTorch + TensorFlow/Keras SDKs, live dashboard,
  deterministic anomaly detection, baseline forecasting, deterministic
  assistant, environment-aware live links (local, Colab, Kaggle, SSH, ngrok
  fallback), **Hugging Face Trainer callback**, **PyTorch Lightning callback**,
  **background hardware telemetry** (CPU/RAM/GPU via pynvml + psutil)
- **v0.2** — experiment comparison / run leaderboard, better forecasting
  (non-linear curve fitting), webhook notifications (Slack/Discord)
- **v0.3** — LLM-backed assistant (Gemini / OpenAI / Ollama), NaN/Inf crash
  watchdog, counterfactual experiment forecasting
- **v0.4** — example-level debugging, historical run intelligence, team
  collaboration, PostgreSQL storage adapter
- **v1.0** — a complete AI training intelligence platform

## 15. Development setup

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

CI runs both of these on every push/PR (`.github/workflows/ci.yml`).

Or run everything with Docker (single container, single port, same as
production mode above):

```bash
docker compose up --build
```

## 16. Contributing

This is an early-stage project. Useful contributions right now:

- More detectors in `analytics/` (NaN/Inf watchdog, dead neuron detector, LR auditor)
- A PostgreSQL implementation of `storage.Storage`
- An LLM-backed implementation of `assistant.TrainingAssistant`
- Non-linear forecasting engine (`AsymptoticTrendForecastEngine`)
- Multi-run comparison dashboard view
- Webhook notification sink (Slack / Discord / email)

Please add tests for new functionality under `tests/` — see the
existing suite for the patterns used (fixtures, `TestClient` for API
tests, etc). See [`docs/KT_NOTES.md`](docs/KT_NOTES.md) for a full
code-level walkthrough of how the pieces fit together before you start.

## Project layout

```
prc/
├── sdk/prc_sdk/
│   ├── monitor.py           Monitor — the main user-facing entry point
│   ├── events.py            Event protocol (versioned, framework-independent)
│   ├── transport.py         LocalBuffer + HttpSender (fail-safe I/O)
│   ├── live_url.py          Environment-aware live link (local/Colab/Kaggle/SSH)
│   ├── pytorch.py           PyTorch hooks: gradient/parameter/GPU stats, TorchMonitorHook
│   ├── tensorflow.py        TensorFlow/Keras callback (PrcKerasCallback)
│   ├── transformers.py      Hugging Face TrainerCallback (PrcHfCallback)  ← new
│   ├── lightning.py         PyTorch Lightning callback (PrcLightningCallback) ← new
│   └── hardware.py          Background hardware sampler: CPU/RAM/GPU telemetry ← new
├── server/                  FastAPI app: REST + WebSocket + dashboard static serving
├── storage/                 Storage abstraction + SQLite implementation
├── analytics/               Deterministic anomaly detectors (overfitting, plateau,
│                              gradient anomaly, instability)
├── forecasting/             Forecast engine abstraction + linear baseline impl
├── assistant/               Deterministic training assistant
├── dashboard/               React + TypeScript frontend
├── examples/
│   ├── mnist/               PyTorch end-to-end example
│   └── keras_mnist/         TensorFlow/Keras end-to-end example
├── tests/                   pytest suite (59 tests)
├── docs/KT_NOTES.md         In-depth code walkthrough / knowledge transfer notes
├── .github/workflows/ci.yml
├── pyproject.toml
├── docker-compose.yml
└── README.md
```
