"""
End-to-end prc example using Keras / TensorFlow and a real MNIST subset.

Unlike the PyTorch example (which uses TorchMonitorHook + a manual loop),
this one uses the standard `model.fit()` path with `PrcKerasCallback` —
the way most Keras users actually train.

Usage:
    # 1. Start the prc server in another terminal:
    #      uvicorn server.main:app --reload
    # 2. Run this script:
    python examples/keras_mnist/train.py
    # 3. Open the dashboard and select project "mnist-keras-demo".

Trains on a small subset for many epochs so overfitting shows up quickly
in the dashboard - this is a demo, not best practice.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from prc_sdk import Monitor
from prc_sdk.tensorflow import PrcKerasCallback


def build_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-subset", type=int, default=1500, help="small subset to induce overfitting quickly")
    parser.add_argument("--val-subset", type=int, default=1000)
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train[: args.train_subset].astype("float32") / 255.0)[..., None]
    y_train = y_train[: args.train_subset]
    x_val = (x_test[: args.val_subset].astype("float32") / 255.0)[..., None]
    y_val = y_test[: args.val_subset]

    model = build_model()

    monitor = Monitor(
        project="mnist-keras-demo",
        run_name=f"keras-cnn-{int(time.time())}",
        config={"batch_size": args.batch_size, "epochs": args.epochs, "train_subset": args.train_subset},
        server_url=args.server_url,
    )

    try:
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[PrcKerasCallback(monitor, log_every_n_batches=5)],
            verbose=2,
        )
        monitor.finish(status="completed")
    except Exception:
        monitor.finish(status="failed")
        raise

    print(f"Run finished: {monitor.run_id}. View it in the dashboard under 'mnist-keras-demo'.")


if __name__ == "__main__":
    main()
