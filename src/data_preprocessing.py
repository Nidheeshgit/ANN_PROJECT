"""
data_preprocessing.py
---------------------
Handles loading, inspecting, and preprocessing the UCI Student Performance dataset.

Key design decisions:
- G1 and G2 (first- and second-period grades) are EXCLUDED from features to prevent
  target leakage and to make the model useful for early-warning prediction.
- Target variable is G3 binned into three classes:
      Low    : G3 in [0,  9]   (failing / very low)
      Medium : G3 in [10, 14]  (passing / average)
      High   : G3 in [15, 20]  (good / excellent)
- Preprocessing transformations are fit ONLY on the training split and then
  applied to validation and test splits to prevent data leakage.
"""

import os
import requests
import zipfile
import io
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# Class boundaries (inclusive on both ends)
CLASS_BOUNDARIES = {
    "Low":    (0,  9),
    "Medium": (10, 14),
    "High":   (15, 20),
}
CLASS_ORDER = ["Low", "Medium", "High"]

# Features excluded due to leakage risk (later-term grades)
LEAKAGE_FEATURES = ["G1", "G2"]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "metrics")


# ──────────────────────────────────────────────────────────
# Dataset download & loading
# ──────────────────────────────────────────────────────────

def download_dataset():
    """
    Downloads the UCI Student Performance dataset.
    Tries the official UCI API first; falls back to direct CSV fetch.
    Returns the path to student-mat.csv.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "student-mat.csv")

    if os.path.exists(csv_path):
        print(f"[INFO] Dataset already present at {csv_path}")
        return csv_path

    # Try UCI ML Repository API (zip download)
    print("[INFO] Attempting to download from UCI ML Repository …")
    try:
        url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The archive contains a nested zip "student.zip"
            inner_names = z.namelist()
            print(f"[DEBUG] Archive contents: {inner_names}")
            if "student.zip" in inner_names:
                inner_bytes = z.read("student.zip")
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_z:
                    inner_z.extract("student-mat.csv", DATA_DIR)
            elif "student-mat.csv" in inner_names:
                z.extract("student-mat.csv", DATA_DIR)
            else:
                # Try extracting all and look for it
                z.extractall(DATA_DIR)
                # Search recursively
                for root, dirs, files in os.walk(DATA_DIR):
                    for f in files:
                        if f == "student-mat.csv":
                            full = os.path.join(root, f)
                            if full != csv_path:
                                import shutil
                                shutil.move(full, csv_path)
                            break
        if os.path.exists(csv_path):
            print(f"[INFO] Dataset downloaded to {csv_path}")
            return csv_path
    except Exception as e:
        print(f"[WARN] UCI zip download failed: {e}")

    # Fallback: direct semicolon-separated CSV from a known mirror
    print("[INFO] Trying direct CSV fetch …")
    fallback_urls = [
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv",
        "https://raw.githubusercontent.com/selva86/datasets/master/student-mat.csv",
    ]
    for url in fallback_urls:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(csv_path, "wb") as f:
                f.write(r.content)
            # Quick sanity check
            df_test = pd.read_csv(csv_path, sep=";")
            if "G3" in df_test.columns and len(df_test) > 100:
                print(f"[INFO] Dataset fetched from fallback: {url}")
                return csv_path
        except Exception as e2:
            print(f"[WARN] Fallback {url} failed: {e2}")

    raise RuntimeError(
        "Could not download the dataset automatically.\n"
        "Please download 'student-mat.csv' manually from\n"
        "https://archive.ics.uci.edu/dataset/320/student+performance\n"
        f"and place it in: {DATA_DIR}"
    )


def load_raw_data(csv_path=None):
    """Loads the raw CSV and returns a DataFrame."""
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "student-mat.csv")
    df = pd.read_csv(csv_path, sep=";")
    return df


# ──────────────────────────────────────────────────────────
# Dataset inspection
# ──────────────────────────────────────────────────────────

def inspect_dataset(df):
    """Prints a comprehensive dataset inspection report."""
    print("=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"Shape           : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Duplicate rows  : {df.duplicated().sum()}")
    print(f"Missing values  : {df.isnull().sum().sum()}")
    print()

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical features  ({len(num_cols)}): {num_cols}")
    print()

    print("-- G3 (final grade) distribution --")
    print(df["G3"].value_counts().sort_index())
    print(f"G3 min={df['G3'].min()}, max={df['G3'].max()}, "
          f"mean={df['G3'].mean():.2f}, std={df['G3'].std():.2f}")
    print()

    print("-- Target class distribution after binning --")
    df_tmp = df.copy()
    df_tmp["performance"] = pd.cut(
        df_tmp["G3"],
        bins=[-1, 9, 14, 20],
        labels=CLASS_ORDER,
    )
    counts = df_tmp["performance"].value_counts().sort_index()
    total = len(df_tmp)
    for cls in CLASS_ORDER:
        c = counts.get(cls, 0)
        print(f"  {cls:8s}: {c:4d}  ({100*c/total:.1f}%)")
    print()
    return df


# ──────────────────────────────────────────────────────────
# Target construction
# ──────────────────────────────────────────────────────────

def create_target(df):
    """
    Creates the multiclass target variable from G3.
    Returns a Series of string labels: 'Low', 'Medium', 'High'.
    """
    labels = pd.cut(
        df["G3"],
        bins=[-1, 9, 14, 20],
        labels=CLASS_ORDER,
    ).astype(str)
    return labels


# ──────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────

def prepare_features(df, drop_leakage=True):
    """
    Drops the target column G3 and optionally G1/G2 (leakage).
    Returns a feature DataFrame (X) and raw target series (y_raw).
    """
    y_raw = create_target(df)

    drop_cols = ["G3"]
    if drop_leakage:
        drop_cols += LEAKAGE_FEATURES
        print(f"[INFO] Dropping leakage features: {LEAKAGE_FEATURES}")

    X = df.drop(columns=drop_cols)
    return X, y_raw


# ──────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────

BINARY_MAP = {"yes": 1, "no": 0}

# Ordinal-like features that have a natural order (kept as integer ranks)
ORDINAL_FEATURES = {
    "Medu": [0, 1, 2, 3, 4],
    "Fedu": [0, 1, 2, 3, 4],
    "traveltime": [1, 2, 3, 4],
    "studytime": [1, 2, 3, 4],
    "famrel": [1, 2, 3, 4, 5],
    "freetime": [1, 2, 3, 4, 5],
    "goout": [1, 2, 3, 4, 5],
    "Dalc": [1, 2, 3, 4, 5],
    "Walc": [1, 2, 3, 4, 5],
    "health": [1, 2, 3, 4, 5],
    "failures": [0, 1, 2, 3],
}

# Features with exactly two unique values → binary 0/1
BINARY_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]

# Nominal features → one-hot encoded
NOMINAL_FEATURES = ["Mjob", "Fjob", "reason", "guardian"]


def encode_features(X):
    """
    Applies deterministic encoding (no fitting required for these rules).
    Returns the encoded DataFrame with only numerical columns.
    """
    X = X.copy()

    # Binary features
    for col in BINARY_FEATURES:
        if col in X.columns:
            # school: GP→0, MS→1 ; sex: F→0, M→1 ; address: R→0, U→1
            # famsize: GT3→0, LE3→1 ; Pstatus: A→0, T→1
            # yes/no columns
            if X[col].isin(["yes", "no"]).all():
                X[col] = X[col].map(BINARY_MAP).astype(int)
            else:
                uniq = sorted(X[col].unique())
                mapping = {v: i for i, v in enumerate(uniq)}
                X[col] = X[col].map(mapping).astype(int)

    # Ordinal features are already integers – just ensure correct dtype
    for col in ORDINAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(int)

    # Nominal → one-hot (drop_first=False keeps interpretability)
    nom_present = [c for c in NOMINAL_FEATURES if c in X.columns]
    if nom_present:
        X = pd.get_dummies(X, columns=nom_present, drop_first=False)
        # Convert boolean dummies to int
        bool_cols = X.select_dtypes(include=["bool"]).columns
        X[bool_cols] = X[bool_cols].astype(int)

    return X


# ──────────────────────────────────────────────────────────
# Train / Val / Test split
# ──────────────────────────────────────────────────────────

def split_data(X, y, val_size=0.15, test_size=0.15, random_state=RANDOM_SEED):
    """
    Stratified 70 / 15 / 15 split.
    Returns X_train, X_val, X_test, y_train, y_val, y_test (all numpy arrays).
    Also returns label_encoder fitted on training labels.
    """
    # First split off the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    # Then split validation from remaining
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_trainval,
    )

    print(f"[INFO] Split sizes  → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"[INFO] Random seed  : {random_state}")

    # Encode labels to integers (0, 1, 2) in CLASS_ORDER
    le = LabelEncoder()
    le.fit(CLASS_ORDER)          # Fit on canonical order so 0=Low,1=Medium,2=High
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    return (
        X_train.values, X_val.values, X_test.values,
        y_train_enc, y_val_enc, y_test_enc,
        le,
    )


# ──────────────────────────────────────────────────────────
# Scaling (fit only on training data)
# ──────────────────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    """
    Fits StandardScaler on X_train only, then transforms all splits.
    Returns scaled arrays and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    print("[INFO] StandardScaler fitted on training data only (no leakage).")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ──────────────────────────────────────────────────────────
# Full pipeline entry point
# ──────────────────────────────────────────────────────────

def run_preprocessing(csv_path=None, verbose=True):
    """
    Runs the complete preprocessing pipeline.

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test : np.ndarray (scaled)
        y_train, y_val, y_test : np.ndarray (integer labels 0/1/2)
        label_encoder          : sklearn LabelEncoder
        scaler                 : sklearn StandardScaler
        feature_names          : list[str]
        raw_df                 : original DataFrame
        info                   : dict of dataset statistics
    """
    # 1. Download / load
    if csv_path is None:
        csv_path = download_dataset()
    df = load_raw_data(csv_path)

    if verbose:
        inspect_dataset(df)

    # 2. Features & target
    X_df, y_raw = prepare_features(df, drop_leakage=True)

    # 3. Encode
    X_enc = encode_features(X_df)
    feature_names = X_enc.columns.tolist()
    print(f"[INFO] Features after encoding: {len(feature_names)}")

    # Remove duplicate rows that are exactly identical in X (keep first)
    combined = pd.concat([X_enc.reset_index(drop=True),
                          y_raw.reset_index(drop=True).rename("target")], axis=1)
    n_before = len(combined)
    combined = combined.drop_duplicates()
    n_after = len(combined)
    if n_before != n_after:
        print(f"[INFO] Dropped {n_before - n_after} exact duplicate rows.")
    X_enc = combined.drop(columns=["target"])
    y_raw = combined["target"]

    # 4. Split
    X_train, X_val, X_test, y_train, y_val, y_test, le = split_data(X_enc, y_raw)

    # 5. Scale
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # 6. Save preprocessing info
    os.makedirs(METRICS_DIR, exist_ok=True)
    info = {
        "n_samples_total": int(n_after),
        "n_features_raw": int(df.shape[1]),
        "n_features_after_encoding": int(len(feature_names)),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "random_seed": RANDOM_SEED,
        "class_order": CLASS_ORDER,
        "class_boundaries": {k: list(v) for k, v in CLASS_BOUNDARIES.items()},
        "leakage_features_dropped": LEAKAGE_FEATURES,
        "feature_names": feature_names,
        "g3_stats": {
            "min": int(df["G3"].min()),
            "max": int(df["G3"].max()),
            "mean": float(df["G3"].mean()),
            "std": float(df["G3"].std()),
        },
        "class_counts_train": {
            cls: int(np.sum(y_train == i))
            for i, cls in enumerate(CLASS_ORDER)
        },
        "class_counts_val": {
            cls: int(np.sum(y_val == i))
            for i, cls in enumerate(CLASS_ORDER)
        },
        "class_counts_test": {
            cls: int(np.sum(y_test == i))
            for i, cls in enumerate(CLASS_ORDER)
        },
    }
    with open(os.path.join(METRICS_DIR, "preprocessing_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "label_encoder": le,
        "scaler":        scaler,
        "feature_names": feature_names,
        "raw_df":        df,
        "info":          info,
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print("\nPreprocessing complete.")
    print(f"  Train : {result['X_train'].shape}")
    print(f"  Val   : {result['X_val'].shape}")
    print(f"  Test  : {result['X_test'].shape}")
