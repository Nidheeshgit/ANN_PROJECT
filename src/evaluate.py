"""
evaluate.py
-----------
Evaluation utilities: metrics, confusion matrix, comparison table.
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "metrics")

CLASS_ORDER = ["Low", "Medium", "High"]


# ──────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label_names=None):
    """
    Computes accuracy, macro-F1, macro-recall, per-class recall, confusion matrix.

    Returns a dict with all metric values.
    """
    if label_names is None:
        label_names = CLASS_ORDER

    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_pre = precision_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy":       float(acc),
        "macro_f1":       float(macro_f1),
        "macro_recall":   float(macro_rec),
        "macro_precision": float(macro_pre),
        "per_class_recall": {
            label_names[i]: float(per_class_recall[i])
            for i in range(len(label_names))
        },
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def print_metrics(metrics, title="Evaluation Results"):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1       : {metrics['macro_f1']:.4f}")
    print(f"  Macro-Recall   : {metrics['macro_recall']:.4f}")
    print(f"  Macro-Precision: {metrics['macro_precision']:.4f}")
    print(f"  Per-class Recall:")
    for cls, rec in metrics["per_class_recall"].items():
        print(f"    {cls:8s}: {rec:.4f}")
    print(f"{'='*55}\n")


def save_metrics(metrics, name):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, f"{name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [Metrics saved → {path}]")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, experiment_name="exp", save=True):
    """Plots training vs validation accuracy and loss curves."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#ff7f0e", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title(f"Training vs Validation Loss\n({experiment_name})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", color="#2ca02c")
    axes[1].plot(epochs, history["val_acc"],   label="Val Accuracy",   color="#d62728", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Training vs Validation Accuracy\n({experiment_name})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"{experiment_name}_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Plot saved → {path}]")
    plt.close()
    return path if save else None


def plot_confusion_matrix(cm, label_names, experiment_name="exp", save=True):
    """Plots and optionally saves a confusion matrix heatmap."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cm_arr = np.array(cm)
    # Normalised version (row-wise)
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Raw counts
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title(f"Confusion Matrix (Counts)\n{experiment_name}")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Normalised (recall per cell)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1], linewidths=0.5,
        vmin=0.0, vmax=1.0,
    )
    axes[1].set_title(f"Confusion Matrix (Normalised)\n{experiment_name}")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"{experiment_name}_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Plot saved → {path}]")
    plt.close()
    return path if save else None


def plot_g3_distribution(df, save=True):
    """Plots G3 raw distribution and binned class distribution."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Raw G3 histogram
    axes[0].hist(df["G3"], bins=range(0, 22), color="#4C72B0", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("G3 (Final Grade)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of G3 (Raw Grades)")
    axes[0].set_xticks(range(0, 21))
    axes[0].grid(True, alpha=0.3)

    # Class distribution bar chart
    import pandas as pd
    bins_   = [-1, 9, 14, 20]
    labels_ = ["Low", "Medium", "High"]
    classes = pd.cut(df["G3"], bins=bins_, labels=labels_)
    counts  = classes.value_counts()[labels_]
    colors  = ["#e74c3c", "#f39c12", "#2ecc71"]
    axes[1].bar(labels_, counts.values, color=colors, edgecolor="black", alpha=0.85)
    for i, (lbl, cnt) in enumerate(zip(labels_, counts.values)):
        axes[1].text(i, cnt + 1, str(cnt), ha="center", fontsize=11)
    axes[1].set_xlabel("Performance Class")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Binned Performance Class Distribution")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "g3_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Plot saved → {path}]")
    plt.close()
    return path if save else None


# ──────────────────────────────────────────────────────────────────────────────
# Summary comparison table
# ──────────────────────────────────────────────────────────────────────────────

def build_comparison_table(all_results):
    """
    Builds and prints a comparison table from a list of result dicts.

    Each result dict should have keys:
        name, accuracy, macro_f1, macro_recall, notes
    """
    header = f"{'Model':<30} {'Accuracy':>9} {'Macro-F1':>9} {'Macro-Rec':>10} {'Notes'}"
    print("\n" + "=" * 85)
    print("  EXPERIMENT COMPARISON TABLE")
    print("=" * 85)
    print(header)
    print("-" * 85)
    for r in all_results:
        print(f"  {r['name']:<28} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f} "
              f"{r['macro_recall']:>10.4f}  {r.get('notes', '')}")
    print("=" * 85)

    # Save as JSON
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, "comparison_table.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  [Comparison table saved → {path}]")
    return path
