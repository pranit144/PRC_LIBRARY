"""
End-to-end prc example: train a small CNN on MNIST while streaming
metrics, gradient stats, and checkpoints to prc.

Usage:
    # 1. Start the prc server in another terminal:
    #      uvicorn server.main:app --reload
    # 2. Run this script:
    python examples/mnist/train.py
    # 3. Open the dashboard (see dashboard/README.md) and select this run.

This intentionally trains a slightly over-parameterized model for a few
extra epochs on a subset of MNIST so that overfitting is easy to observe
in the dashboard - it's meant as a demo, not best training practice.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from prc_sdk import Monitor
from prc_sdk.pytorch import TorchMonitorHook


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_loaders(data_dir: str, train_subset: int, val_subset: int, batch_size: int):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_full = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

    train_ds = Subset(train_full, range(train_subset))
    val_ds = Subset(test_full, range(val_subset))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-subset", type=int, default=3000, help="small subset to induce overfitting quickly")
    parser.add_argument("--val-subset", type=int, default=1000)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader = get_loaders(args.data_dir, args.train_subset, args.val_subset, args.batch_size)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    monitor = Monitor(
        project="mnist-demo",
        run_name=f"cnn-lr{args.lr}-{int(time.time())}",
        config={"lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
                "train_subset": args.train_subset},
        server_url=args.server_url,
    )
    hook = TorchMonitorHook(monitor, model, log_every_n_steps=20)

    best_val_loss = float("inf")
    step = 0

    try:
        for epoch in range(args.epochs):
            monitor.epoch_started(epoch)
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()

                hook.maybe_log(step, epoch)
                optimizer.step()

                if step % 10 == 0:
                    val_loss, val_acc = evaluate(model, val_loader, device)
                    monitor.log(
                        step=step, epoch=epoch,
                        train_loss=float(loss.item()),
                        val_loss=val_loss,
                        accuracy=val_acc,
                        learning_rate=optimizer.param_groups[0]["lr"],
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt_path = str(Path(args.checkpoint_dir) / f"best_step{step}.pt")
                        torch.save(model.state_dict(), ckpt_path)
                        monitor.log_checkpoint(step=step, epoch=epoch, path=ckpt_path,
                                                metrics={"val_loss": val_loss, "accuracy": val_acc})

                step += 1

            monitor.epoch_finished(epoch)
            print(f"epoch {epoch} done (step {step})")

        monitor.finish(status="completed")
    except Exception:
        monitor.finish(status="failed")
        raise

    print(f"Run finished: {monitor.run_id}. View it in the dashboard.")


if __name__ == "__main__":
    main()
