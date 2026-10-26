"""
train.py
--------
Training loop for the MLP using mini-batch SGD.

Supports:
- Mini-batch / stochastic SGD (batch_size=1 → pure SGD)
- Early stopping on validation loss
- Epoch-level history recording
"""

import os
import json
import time

import numpy as np

from src.model import MLP, one_hot

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models")
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "metrics")


def train(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    epochs:    int  = 200,
    batch_size: int = 32,
    patience:   int = 20,
    min_delta: float = 1e-4,
    verbose:   bool  = True,
    experiment_name: str = "exp",
):
    """
    Trains the MLP with mini-batch SGD and optional early stopping.

    Parameters
    ----------
    model        : MLP instance (already constructed)
    X_train, y_train : training data (arrays)
    X_val, y_val     : validation data (arrays)
    epochs       : maximum number of training epochs
    batch_size   : number of samples per SGD update
                   (1 = pure stochastic, N = batch GD)
    patience     : early-stopping patience (epochs without val-loss improvement)
    min_delta    : minimum improvement to reset patience counter
    verbose      : print epoch summaries if True
    experiment_name : used for saving history

    Returns
    -------
    history : dict with train_loss, val_loss, train_acc, val_acc lists
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    N = X_train.shape[0]
    num_classes = model.num_classes

    best_val_loss = np.inf
    patience_counter = 0
    best_weights = None
    best_biases  = None

    # Reset history
    model.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start = time.time()

    for epoch in range(1, epochs + 1):
        # ── Shuffle training data each epoch ──────────────────────────────────
        perm = np.random.permutation(N)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        # ── Mini-batch SGD ─────────────────────────────────────────────────────
        for start_idx in range(0, N, batch_size):
            X_batch = X_shuffled[start_idx: start_idx + batch_size]
            y_batch = y_shuffled[start_idx: start_idx + batch_size]
            y_oh    = one_hot(y_batch, num_classes)

            # Forward pass
            model.forward(X_batch)
            # Backward pass
            grads_W, grads_b = model.backward(y_oh)
            # Weight update
            model.update_weights(grads_W, grads_b)

        # ── Epoch metrics ─────────────────────────────────────────────────────
        train_loss = model.loss(X_train, y_train)
        val_loss   = model.loss(X_val,   y_val)
        train_acc  = model.accuracy(X_train, y_train)
        val_acc    = model.accuracy(X_val,   y_val)

        model.history["train_loss"].append(float(train_loss))
        model.history["val_loss"].append(float(val_loss))
        model.history["train_acc"].append(float(train_acc))
        model.history["val_acc"].append(float(val_acc))

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Deep-copy best weights
            best_weights = [w.copy() for w in model.W]
            best_biases  = [b.copy() for b in model.b]
        else:
            patience_counter += 1

        if verbose and (epoch % 20 == 0 or epoch == 1 or epoch == epochs):
            elapsed = time.time() - start
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                  f"[{elapsed:.1f}s]")

        if patience_counter >= patience:
            if verbose:
                print(f"  [Early stopping at epoch {epoch}]")
            break

    # Restore best weights
    if best_weights is not None:
        model.W = best_weights
        model.b = best_biases

    # Save history
    hist_path = os.path.join(METRICS_DIR, f"{experiment_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(model.history, f, indent=2)
    if verbose:
        print(f"  History saved → {hist_path}")

    return model.history
