# prc — Knowledge Transfer Notes

This is a walkthrough of how prc is actually built, written the way I'd
explain it to someone taking over the codebase. It traces real code, not
just describes concepts abstractly. Read this alongside the repo open in
an editor.

---

## 1. The one sentence version

**A training script calls `monitor.log(...)`. That call writes to a
local file, then (best-effort) sends the same data over HTTP to a
server, which stores it in SQLite, runs some math on it, and pushes it
to a browser tab over a WebSocket.** Everything else in the repo exists
to make that one sentence robust, explainable, and portable.

---

## 2. The full request lifecycle, traced through actual files

Let's trace one `monitor.log(step=5, epoch=0, train_loss=0.4, val_loss=0.5)`
call from your training script all the way to a pixel changing on your
screen.

### Step 1 — `sdk/prc_sdk/monitor.py` → `Monitor.log()`

```python
@_safe
def log(self, step, epoch=0, **metrics):
    self._last_step = step
    self._last_epoch = epoch
    self._emit(metric_event(self.run_id, step, epoch, metrics))
```

Two things worth noticing:

- **`@_safe`** — every public method on `Monitor` is wrapped in a
  decorator (defined just above the class) that catches *any* exception
  and logs it instead of raising. This is the "monitoring must never
  crash training" requirement from the spec, implemented as a single
  reusable decorator rather than try/except scattered through every
  method.
- **`_emit()`** is the only place that actually does I/O. `log()` just
  builds an `Event` object and hands it off.

### Step 2 — `sdk/prc_sdk/events.py` → `metric_event()`

```python
def metric_event(run_id, step, epoch, metrics):
    return Event(run_id=run_id, event="metric", step=step, epoch=epoch,
                 data={"metrics": metrics})
```

`Event` is a `@dataclass` — the single source of truth for "what a
training event looks like." It's deliberately generic: every event type
(`metric`, `checkpoint`, `anomaly`, `run_started`, ...) uses the same
five fields (`run_id`, `event`, `step`, `epoch`, `data`), with `data`
being a free-form dict whose shape depends on `event`. This is why
adding a new event type later (e.g. `data_drift_detected` in a future
version) doesn't require touching the schema — just a new value in
`EVENT_TYPES` and a constructor function.

### Step 3 — `sdk/prc_sdk/monitor.py` → `Monitor._emit()`

```python
def _emit(self, event):
    if self._buffer is not None:
        self._buffer.append(event)
    if self._sender is not None:
        self._sender.send(event)
```

Every event goes to **two places**, and the order matters:

1. **`self._buffer`** (`transport.LocalBuffer`) — appends the event as
   one line of JSON to a local `.jsonl` file
   (`.prc/{project}/{run_id}.jsonl`). This write is synchronous and
   happens first, so even if the process crashes immediately after, the
   data survives on disk.
2. **`self._sender`** (`transport.HttpSender`) — hands the event to a
   background thread via a queue and returns immediately. `log()` never
   blocks on network I/O.

### Step 4 — `sdk/prc_sdk/transport.py` → `HttpSender`

```python
def send(self, event):
    if not self.server_url:
        return
    try:
        self._queue.put_nowait(event)
    except queue.Full:
        logger.warning(...)   # drop, don't block training
```

A dedicated background thread (`_worker`) pulls events off this queue
and POSTs them one at a time to `{server_url}/api/events`. If the
server is unreachable, `_post()` catches the exception and does
nothing — the event is already safe in the local JSONL buffer, so this
is by design, not a bug.

### Step 5 — `server/main.py` → `POST /api/events`

```python
@api.post("/events")
async def post_event(event: EventIn):
    event_dict = event.model_dump()
    storage.save_event(event_dict)

    if event.event == "metric":
        metrics = storage.list_metrics(event.run_id)
        found = analytics_engine.analyze(metrics, gradient_history)
        for anomaly in found:
            if anomaly["type"] not in existing_types:
                storage.save_anomaly(event.run_id, anomaly)
                await ws_manager.broadcast(event.run_id, {"event": "anomaly", "data": anomaly})

    await ws_manager.broadcast(event.run_id, event_dict)
    return {"status": "accepted"}
```

Three things happen on every metric event, in order:

1. **Persist** — `storage.save_event()` writes the raw event to SQLite.
2. **Analyze** — pull the run's full metric history back out and run it
   through every detector in `analytics/`. This is intentionally
   re-computed on every single event rather than incrementally updated,
   because the detectors are cheap (pure Python over a list of dicts,
   no ML) and correctness-by-recomputation is much easier to reason
   about than maintaining rolling state.
3. **Broadcast** — push the raw event, and any newly-found anomaly, out
   to every WebSocket currently subscribed to this `run_id`.

### Step 6 — `storage/sqlite_storage.py` → `save_event()`

This is the one place that understands the *meaning* of different event
types (everywhere else just treats events as opaque JSON):

```python
if etype == "run_started":
    cur.execute("INSERT ... INTO projects ...")
    cur.execute("INSERT ... INTO runs ...")
elif etype == "run_finished":
    cur.execute("UPDATE runs SET status = ? ...")
elif etype == "checkpoint":
    cur.execute("INSERT INTO checkpoints ...")
```

Every event, regardless of type, also gets inserted into a generic
`events` table (the full audit log), *and* certain event types update
denormalized tables (`runs.last_step`, `runs.status`, etc.) so the
dashboard's "get me this run's current state" query is a single cheap
row lookup rather than "scan every event for this run and derive
state."

### Step 7 — `analytics/engine.py` + the four detector files

```python
class AnalyticsEngine:
    def analyze(self, metrics, gradient_history=None):
        results = []
        if overfit := detect_overfitting(metrics): results.append(overfit)
        if plateau := detect_plateau(metrics): results.append(plateau)
        ...
        return results
```

Each detector (`overfitting.py`, `plateau.py`, `instability.py`,
`gradient_anomaly.py`) is a **pure function**: list of metric dicts in,
`Optional[dict]` out. No classes, no state, no ML models — just
statistics (linear regression slope, coefficient of variation, moving
windows). This was a deliberate choice: the spec explicitly says
"explain what happened, don't just show numbers" and "never claim
certainty" — a hand-written, inspectable `if slope_a < 0 and slope_b >
0` is much easier to trust and debug than a black-box classifier, and
for an MVP it's genuinely sufficient.

### Step 8 — `forecasting/engine.py` → `SimpleTrendForecastEngine`

Only called on-demand (`GET /api/runs/{id}/forecast`), not on every
event — forecasting is more expensive and less urgent than anomaly
detection, so it's pull-based rather than push-based.

```python
slope, intercept = _linear_fit(xs, ys)
future_step = last_step + horizon_steps   # horizon = span already observed
predicted_final = intercept + slope * future_step
```

The one non-obvious design point here: the forecast horizon defaults to
**the same span of steps already observed**, not some fixed large
number. My first version defaulted to `horizon_steps=100000`, which
produced nonsense (extrapolating a 30-step trend 100,000 steps into the
future gives values like `425.8` for a loss that should end near `0.3`).
I caught this in testing — see the confidence/uncertainty math right
below it for how the interval widens based on how well the linear fit
actually explains recent history (`residual_std`).

### Step 9 — `server/ws_manager.py` → `ConnectionManager.broadcast()`

```python
async def broadcast(self, run_id, message):
    conns = self._connections.get(run_id, [])
    for ws in conns:
        await ws.send_json(message)
```

Simple in-memory dict of `run_id -> [WebSocket, ...]`. No message queue,
no Redis, no pub/sub system — this is a single-process local server, so
an in-memory dict is the right amount of infrastructure. (If you ever
need multiple server processes/replicas, this is the first thing that
would need to become Redis pub/sub or similar — it's the one piece of
state that doesn't live in SQLite.)

### Step 10 — `dashboard/src/lib/api.ts` → `connectRunSocket()`

```ts
ws = new WebSocket(`${WS_BASE_URL}/api/ws/runs/${runId}`);
ws.onmessage = (evt) => onMessage(JSON.parse(evt.data));
```

### Step 11 — `dashboard/src/pages/RunPage.tsx`

```ts
const disconnect = connectRunSocket(runId, (msg) => {
  if (msg.event === 'metric') {
    setMetrics((prev) => [...prev, { step: msg.step, ...msg.data.metrics }]);
  } else if (msg.event === 'anomaly') {
    setAnomalies((prev) => [...prev, msg.data]);
  }
});
```

React state update → re-render → `TrainingChart` redraws with the new
point → pixel changes on your screen. This is also backed by a 4-second
poll (`setInterval(refresh, 4000)`) as a fallback in case the WebSocket
drops, so the dashboard degrades gracefully instead of silently going
stale.

---

## 3. Why things are split into separate top-level packages

```
sdk/prc_sdk/   storage/   analytics/   forecasting/   assistant/   server/
```

This isn't arbitrary — it mirrors the spec's explicit requirement to
"keep storage behind an abstraction," "keep forecasting behind an
abstraction," and "keep the SDK independent from the dashboard." Each
package has exactly one abstract interface that everything else depends
on:

| Package | Abstract interface | Concrete implementation | Why it's separated |
|---|---|---|---|
| `storage/` | `Storage` (ABC, `base.py`) | `SqliteStorage` | Swap in Postgres later without touching `server/main.py` |
| `forecasting/` | `ForecastEngine` (ABC) | `SimpleTrendForecastEngine` | Swap in a real ML model later without touching the API |
| `assistant/` | `TrainingAssistant` (ABC) | `DeterministicAssistant` | Swap in an LLM-backed version later (v0.3) without touching the API |

`server/main.py` only ever imports the concrete classes at the top — if
you're adding a Postgres backend, you write a new
`PostgresStorage(Storage)` class and change one import line.

---

## 4. The SDK's fail-safe pattern, in full

This shows up in three layers, worth understanding as one coherent
design rather than three separate tricks:

1. **`@_safe` decorator** (`monitor.py`) — catches exceptions raised
   *inside* SDK methods (e.g. a bad metric value, a full disk).
2. **Local-buffer-first** (`_emit()`) — even if the network layer is
   completely broken, data isn't lost, just not yet visible on the
   dashboard.
3. **Bounded background queue** (`HttpSender`) — if the server is slow
   or down, the queue fills up and *new* events get dropped (with a
   warning log) rather than blocking the training loop or growing
   memory unboundedly.

The PyTorch/TensorFlow integration files (`pytorch.py`, `tensorflow.py`)
add a fourth layer: `_safe_call()`, a lighter version of the same idea
for framework-specific helper functions like `gradient_stats()`, since
those touch tensor internals that can fail in framework-specific ways
(e.g. `p.grad is None` before the first backward pass).

---

## 5. The framework integrations aren't symmetric, and that's intentional

**PyTorch** (`pytorch.py`): no high-level training loop exists in
PyTorch to hook into, so the integration is a set of helper functions
(`gradient_stats`, `parameter_stats`, `gpu_stats`) plus a
`TorchMonitorHook` convenience wrapper that you call manually inside
your own loop:

```python
hook.maybe_log(step, epoch)   # you call this yourself, every step
```

**TensorFlow/Keras** (`tensorflow.py`): Keras's `model.fit()` *does*
have a high-level hook system (`keras.callbacks.Callback`), so the
primary integration is `PrcKerasCallback`, which you pass into
`fit(callbacks=[...])` and never call anything manually. But this
comes with a real limitation: Keras's callback API gives you `logs`
dicts of *already-computed metrics*, not raw gradients — so
`PrcKerasCallback` cannot give you gradient/parameter stats the way
`TorchMonitorHook` can. For that level of detail in Keras, you drop
down to a custom `tf.GradientTape` loop and call the module-level
`gradient_stats()` / `parameter_stats()` functions yourself, mirroring
the PyTorch path. This is documented at the top of `tensorflow.py`.

One bug I found and fixed while actually testing this: Keras reports
`loss`/`val_loss` in its logs dict, but the analytics detectors
(`overfitting.py` etc.) look for `train_loss`/`val_loss` specifically
(matching the PyTorch example's naming). `_normalize_metric_names()` in
`tensorflow.py` renames `loss` → `train_loss` without discarding the
original key, so anomaly detection works automatically for Keras users
without them needing to know about this internal convention.

A second bug found in the same test pass: Keras only reports `val_*`
metrics inside `on_epoch_end`, not per-batch. My first version of the
callback only forwarded those into an `epoch_finished` event, which the
analytics detectors never scan (they only read `metric` events) — so
overfitting detection silently never fired for Keras runs. Fixed by
also emitting a `metric` event at epoch end, not just `epoch_finished`.

---

## 6. The single-port refactor (why `/api` exists)

Originally the dashboard (Vite dev server, `:5173`) and the API
(FastAPI, `:8000`) were two separate processes. This breaks portability:
Colab and Kaggle can only cleanly proxy one arbitrary port per notebook
cell, and juggling two proxied ports is fragile everywhere. The fix,
in `server/main.py`:

```python
api = APIRouter(prefix="/api")
# ... all endpoints defined on `api`, not `app` ...
app.include_router(api)

if DASHBOARD_DIST.exists():
    app.mount("/assets", StaticFiles(directory=...))
    @app.get("/{full_path:path}")
    async def serve_dashboard(full_path: str):
        candidate = DASHBOARD_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_file)   # SPA fallback
```

The API had to move to `/api/*` because the dashboard's own
client-side route for a run page is `/runs/:id` — the exact same path
shape the API uses for `GET /runs/{run_id}`. Without the prefix, they'd
collide on the same server. The catch-all route at the bottom is what
makes React Router's client-side navigation work: any path that isn't a
real static file falls back to `index.html`, and React Router takes
over from there in the browser.

The dashboard's `lib/api.ts` defaults to `window.location.origin` for
this exact reason — whatever URL you're actually viewing the dashboard
at (localhost, a Colab proxy URL, whatever), API calls go to the same
origin automatically, no configuration needed.

---

## 7. Where to look when you want to change something

| I want to... | Start here |
|---|---|
| Add a new event type (e.g. `data_drift_detected`) | `sdk/prc_sdk/events.py` — add to `EVENT_TYPES` + a constructor function |
| Add a new anomaly detector | `analytics/` — copy the shape of `plateau.py` (pure function, list-of-dicts in, `Optional[dict]` out), then register it in `analytics/engine.py` |
| Change what the dashboard shows | `dashboard/src/pages/RunPage.tsx` is the page; it composes components from `dashboard/src/components/` |
| Swap SQLite for Postgres | Implement `storage.Storage` (see `storage/base.py`), swap the import in `server/main.py` |
| Add a real LLM-backed assistant | Implement `assistant.TrainingAssistant`, swap the import in `server/main.py` — `DeterministicAssistant.answer()` shows the context shape (metrics, anomalies, forecast) an LLM adapter would need |
| Add a new framework integration (Keras done, JAX/Lightning next) | New file in `sdk/prc_sdk/`, following the pattern in `tensorflow.py` (lazy `import` inside functions so the core SDK doesn't require the framework) |
| Change how forecasts are computed | `forecasting/engine.py` — implement `ForecastEngine.forecast()` |
| Change the live-URL / environment detection | `sdk/prc_sdk/live_url.py` |

---

## 8. What's tested vs. what's verified-by-hand

`tests/` (36 tests, all passing) covers: event schema round-tripping,
SDK fail-safety, SQLite storage correctness, all four analytics
detectors (positive and negative cases), the forecasting engine
(including the "insufficient data" and bounded-metric edge cases), and
the full FastAPI app via `TestClient` (REST + WebSocket).

What I additionally ran by hand, outside the automated suite, to sanity
check things `TestClient` can't fully represent: a real `uvicorn`
process, a real Keras `model.fit()`, real HTTP over an actual TCP
socket end-to-end (not ASGI in-process), confirming the single-port
static file serving + `/api` routing actually work together, and that
the live-URL line prints correctly before training even starts.

What is **not** yet tested end-to-end: the PyTorch MNIST example
(`examples/mnist/train.py`) — torch/torchvision aren't installed in the
environment I built this in, so it's syntax-checked but not executed.
The Colab/Kaggle branches of `live_url.py` are similarly untested in a
real Colab/Kaggle session (I don't have access to one) — the logic is
straightforward but hasn't been run against the real `google.colab`
module.
