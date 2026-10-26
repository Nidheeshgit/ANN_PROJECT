"""
main.py
-------
Orchestrates the complete pipeline:
1. Preprocessing
2. Baseline model
3. Four MLP experiments (E1–E4)
4. Final evaluation, plots, and comparison table

Run:
    python main.py

All results are saved in results/figures/, results/metrics/, results/models/.
"""

import os
import sys
import json
import numpy as np

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import run_preprocessing
from src.model import MLP
from src.train import train
from src.evaluate import (
    compute_metrics, print_metrics, save_metrics,
    plot_training_history, plot_confusion_matrix,
    plot_g3_distribution, build_comparison_table,
)
from src.utils import (
    error_analysis, plot_feature_importances_proxy, set_random_seed,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

METRICS_DIR = os.path.join(os.path.dirname(__file__), "results", "metrics")
CLASS_ORDER  = ["Low", "Medium", "High"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one MLP experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(name, layer_sizes, lr, batch_size, momentum,
                   epochs, patience, data):
    print(f"\n{'━'*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Architecture : {layer_sizes}")
    print(f"  LR           : {lr}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Momentum     : {momentum}")
    print(f"  Max epochs   : {epochs}")
    print(f"{'━'*60}")

    set_random_seed(42)
    model = MLP(layer_sizes, learning_rate=lr, momentum=momentum, random_state=42)
    model.summary()

    history = train(
        model,
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=True,
        experiment_name=name,
    )

    # Test evaluation
    y_pred = model.predict(data["X_test"])
    metrics = compute_metrics(data["y_test"], y_pred, CLASS_ORDER)
    print_metrics(metrics, title=f"{name} — Test Set")
    save_metrics(metrics, name)

    # Plots
    plot_training_history(history, experiment_name=name)
    plot_confusion_matrix(
        metrics["confusion_matrix"], CLASS_ORDER,
        experiment_name=name,
    )

    return model, metrics, history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  STUDENT PERFORMANCE PREDICTION — MLP PROJECT")
    print("=" * 70)

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    print("\n[STEP 1] Data Preprocessing")
    data = run_preprocessing()
    info = data["info"]

    X_train = data["X_train"];  y_train = data["y_train"]
    X_val   = data["X_val"];    y_val   = data["y_val"]
    X_test  = data["X_test"];   y_test  = data["y_test"]
    le      = data["label_encoder"]
    feature_names = data["feature_names"]
    raw_df  = data["raw_df"]

    input_dim   = X_train.shape[1]
    num_classes = 3

    # Distribution plots
    plot_g3_distribution(raw_df)
    plot_feature_importances_proxy(X_train, y_train, feature_names)

    print(f"\n  Input dimension  : {input_dim}")
    print(f"  Classes          : {CLASS_ORDER}")
    print(f"  Train / Val / Test: {len(X_train)} / {len(X_val)} / {len(X_test)}")

    # ── 2. Baseline models ────────────────────────────────────────────────────
    print("\n[STEP 2] Baseline Models")
    all_results = []

    # Majority-class baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    dummy_metrics = compute_metrics(y_test, y_dummy, CLASS_ORDER)
    print_metrics(dummy_metrics, "Majority-Class Baseline — Test Set")
    save_metrics(dummy_metrics, "baseline_majority")
    all_results.append({
        "name": "Majority-Class Baseline",
        "accuracy":     dummy_metrics["accuracy"],
        "macro_f1":     dummy_metrics["macro_f1"],
        "macro_recall": dummy_metrics["macro_recall"],
        "notes": "Predicts most frequent class only",
    })

    # Logistic Regression baseline
    lr_clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    lr_clf.fit(X_train, y_train)
    y_lr = lr_clf.predict(X_test)
    lr_metrics = compute_metrics(y_test, y_lr, CLASS_ORDER)
    print_metrics(lr_metrics, "Logistic Regression Baseline — Test Set")
    save_metrics(lr_metrics, "baseline_logreg")
    all_results.append({
        "name": "Logistic Regression",
        "accuracy":     lr_metrics["accuracy"],
        "macro_f1":     lr_metrics["macro_f1"],
        "macro_recall": lr_metrics["macro_recall"],
        "notes": "Linear baseline, no hidden layers",
    })

    # ── 3. MLP Experiments ────────────────────────────────────────────────────
    print("\n[STEP 3] MLP Experiments")

    experiments = [
        # name, layer_sizes, lr, batch_size, momentum, epochs, patience
        ("E1_Baseline_MLP",
         [input_dim, 64, 32, num_classes], 0.01, 32, 0.9, 300, 30),

        ("E2_Deeper_MLP",
         [input_dim, 128, 64, 32, num_classes], 0.01, 32, 0.9, 300, 30),

        ("E3_HighLR_MLP",
         [input_dim, 64, 32, num_classes], 0.05, 32, 0.9, 300, 30),

        ("E4_SmallBatch_MLP",
         [input_dim, 64, 32, num_classes], 0.01, 8, 0.9, 300, 30),
    ]

    experiment_notes = {
        "E1_Baseline_MLP":   "Baseline: 2 hidden layers [64,32], lr=0.01, bs=32",
        "E2_Deeper_MLP":     "Deeper arch: 3 hidden layers [128,64,32], lr=0.01, bs=32",
        "E3_HighLR_MLP":     "Higher LR: same arch as E1, lr=0.05, bs=32",
        "E4_SmallBatch_MLP": "Smaller batch: same arch as E1, lr=0.01, bs=8",
    }

    best_exp_name = None
    best_f1 = -1
    best_model_obj = None

    for exp_args in experiments:
        name      = exp_args[0]
        arch      = exp_args[1]
        lr_val    = exp_args[2]
        bs        = exp_args[3]
        mom       = exp_args[4]
        ep        = exp_args[5]
        pat       = exp_args[6]

        model_obj, exp_metrics, exp_history = run_experiment(
            name, arch, lr_val, bs, mom, ep, pat, data
        )

        row = {
            "name":         name,
            "accuracy":     exp_metrics["accuracy"],
            "macro_f1":     exp_metrics["macro_f1"],
            "macro_recall": exp_metrics["macro_recall"],
            "notes":        experiment_notes.get(name, ""),
        }
        all_results.append(row)

        if exp_metrics["macro_f1"] > best_f1:
            best_f1 = exp_metrics["macro_f1"]
            best_exp_name = name
            best_model_obj = model_obj

    # ── 4. Error Analysis on best model ───────────────────────────────────────
    print(f"\n[STEP 4] Error Analysis — Best Model: {best_exp_name}")
    y_pred_best = best_model_obj.predict(X_test)

    # Reconstruct raw (unscaled) test features for display
    # We stored original indices via stratified split; use scaled for now
    error_analysis(
        X_test, y_test, y_pred_best,
        feature_names=feature_names,
        label_names=CLASS_ORDER,
        n_samples=5,
    )

    # ── 5. Comparison table ───────────────────────────────────────────────────
    print("\n[STEP 5] Final Comparison Table")
    build_comparison_table(all_results)

    # ── 6. Save master results JSON ───────────────────────────────────────────
    os.makedirs(METRICS_DIR, exist_ok=True)
    master = {
        "preprocessing_info": info,
        "all_results": all_results,
        "best_experiment": best_exp_name,
    }
    with open(os.path.join(METRICS_DIR, "master_results.json"), "w") as f:
        json.dump(master, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  PROJECT COMPLETE!")
    print(f"  Best experiment (by Macro-F1): {best_exp_name} (F1={best_f1:.4f})")
    print(f"  All results saved in results/")
    print(f"{'='*70}\n")

    return all_results, info


if __name__ == "__main__":
    main()
