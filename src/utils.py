"""
utils.py
--------
Utility helpers: error analysis, misclassified sample inspection, etc.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")
CLASS_ORDER = ["Low", "Medium", "High"]


def error_analysis(X_test_raw, y_test, y_pred, feature_names,
                   label_names=None, n_samples=5):
    """
    Identifies misclassified samples and prints their feature values.

    Parameters
    ----------
    X_test_raw   : raw (unscaled) test features as a DataFrame or ndarray
    y_test       : true integer labels
    y_pred       : predicted integer labels
    feature_names: list of feature names
    label_names  : class names
    n_samples    : number of misclassified samples to display
    """
    if label_names is None:
        label_names = CLASS_ORDER

    misclassified_idx = np.where(y_test != y_pred)[0]
    total_mis = len(misclassified_idx)
    print(f"\n-- Error Analysis --")
    print(f"  Total misclassified: {total_mis} / {len(y_test)}")

    if total_mis == 0:
        print("  (No misclassifications!)")
        return

    # Show confusion breakdown
    print(f"\n  Misclassification breakdown:")
    for true_cls in range(len(label_names)):
        for pred_cls in range(len(label_names)):
            if true_cls == pred_cls:
                continue
            count = np.sum((y_test == true_cls) & (y_pred == pred_cls))
            if count > 0:
                print(f"    True={label_names[true_cls]:6s} → Pred={label_names[pred_cls]:6s}: {count}")

    # Display n sample misclassified rows
    sample_idx = misclassified_idx[:n_samples]
    print(f"\n  First {len(sample_idx)} misclassified samples:")
    print("-" * 60)

    if isinstance(X_test_raw, pd.DataFrame):
        X_display = X_test_raw.iloc[sample_idx]
    else:
        X_display = pd.DataFrame(X_test_raw[sample_idx], columns=feature_names)

    for i, idx in enumerate(sample_idx):
        row = X_display.iloc[i]
        true_lbl = label_names[y_test[idx]]
        pred_lbl = label_names[y_pred[idx]]
        print(f"\n  Sample {idx}  (True: {true_lbl}, Predicted: {pred_lbl})")
        # Show a selection of informative features
        informative = ["studytime", "failures", "absences", "higher", "goout",
                       "Dalc", "Walc", "health", "Medu", "Fedu"]
        for feat in informative:
            if feat in row.index:
                print(f"    {feat:15s}: {row[feat]}")
    print("-" * 60)


def plot_feature_importances_proxy(X_train, y_train, feature_names, top_n=15, save=True):
    """
    Computes a simple proxy for feature importance: absolute correlation
    of each feature with the integer target (Pearson r).
    Not a true MLP importance, but informative for analysis.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    correlations = []
    for i in range(X_train.shape[1]):
        r = np.corrcoef(X_train[:, i], y_train)[0, 1]
        correlations.append(abs(r) if not np.isnan(r) else 0.0)

    correlations = np.array(correlations)
    top_idx = np.argsort(correlations)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = correlations[top_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_names)))[::-1]
    bars = ax.barh(range(len(top_names)), top_vals, color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("|Pearson r| with target class")
    ax.set_title(f"Top {top_n} Feature Correlations with Performance Class")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "feature_importances.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Feature importance plot saved → {path}]")
        plt.close()
        return path
    plt.close()
    return None


def set_random_seed(seed=42):
    """Sets numpy random seed for reproducibility."""
    np.random.seed(seed)


def load_json(path):
    with open(path) as f:
        return json.load(f)
