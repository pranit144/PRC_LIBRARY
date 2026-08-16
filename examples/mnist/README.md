# prc MNIST example

Demonstrates the full prc pipeline end-to-end: a real PyTorch training
loop instrumented with the prc SDK, streaming to a local prc server and
visualized live in the dashboard.

## Requirements

```bash
pip install torch torchvision
pip install -e ../..   # installs the prc_sdk / server / storage / analytics packages
```

## 1. Start the server

```bash
cd ../..
uvicorn server.main:app --reload
```

This starts the API + WebSocket server at `http://localhost:8000` and
creates a local `prc.db` SQLite file.

## 2. Start the dashboard (optional but recommended)

```bash
cd ../../dashboard
npm install
npm run dev
```

## 3. Run the training script

```bash
python train.py
```

By default this trains a small CNN on a 3,000-image subset of MNIST for
20 epochs — small enough to overfit quickly, so you can watch prc's
overfitting detector and forecast panel react in real time.

What this demonstrates:

1. **Starting prc** — `Monitor(project="mnist-demo", ...)`
2. **Training the model** — a standard PyTorch loop
3. **Sending metrics** — `monitor.log(...)` every 10 steps, plus gradient/
   parameter/GPU stats every 20 steps via `TorchMonitorHook`
4. **Viewing the dashboard** — live loss/accuracy curves, GPU panel, timeline
5. **Detecting overfitting** — because the training subset is small, val
   loss will start climbing while train loss keeps falling after a few
   epochs; the dashboard should flag this
6. **Generating a forecast** — the forecast panel updates as new points
   arrive
7. **Saving the run** — checkpoints are written to `./checkpoints` and
   logged as `checkpoint` events whenever validation loss improves

Useful flags:

```bash
python train.py --epochs 30 --lr 0.0005 --train-subset 2000
```
