# Student Performance Prediction Using Multiclass MLP

## Project Overview

This project implements a **Multiclass Multilayer Perceptron (MLP)** trained with **Stochastic Gradient Descent (SGD)** to predict student academic performance from the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance). The system classifies students into three performance categories — **Low**, **Medium**, and **High** — based on their demographic, social, and academic background features (excluding later-term grades G1 and G2 to prevent target leakage).

---

## Problem Statement

Early identification of students at risk of academic failure is a critical educational challenge. Traditional approaches rely on end-of-term grades — by which time it may be too late to intervene. This project builds an early-warning prediction system that uses only pre-term and term-entry attributes to predict a student's likely final performance category, enabling timely academic support.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | UCI Student Performance (Math subject) |
| **Source** | https://archive.ics.uci.edu/dataset/320/student+performance |
| **Records** | 395 students |
| **Raw features** | 33 (17 categorical, 16 numerical) |
| **Features used** | 43 (after encoding, excluding G1/G2/G3) |
| **Target** | G3 (final grade) → binned into 3 classes |

### Target Classes (G3 binning)

| Class | G3 Range | Count | % |
|---|---|---|---|
| Low | 0–9 | 130 | 32.9% |
| Medium | 10–14 | 192 | 48.6% |
| High | 15–20 | 73 | 18.5% |

### Key Feature Groups

- **Demographic**: school, sex, age, address, family size
- **Family**: parents' education (Medu, Fedu), parents' jobs, guardian
- **Academic**: study time, past failures, school support, paid tuition
- **Lifestyle**: going out (goout), alcohol consumption (Dalc, Walc), health, absences
- **Aspirations**: desire for higher education (higher)

> **G1 and G2 excluded** — including mid-term grades would introduce target leakage, making the model only useful after grades are available, which defeats the early-warning purpose.

---

## Methodology

### Preprocessing Pipeline

| Step | Method |
|---|---|
| Missing values | None (clean dataset) |
| Duplicate rows | None found |
| Binary features | Mapped to 0/1 |
| Ordinal features | Kept as integers |
| Nominal features | One-hot encoded |
| Feature scaling | StandardScaler (fit on train only) |

### Data Split

| Split | Size | Proportion |
|---|---|---|
| Train | 275 | 70% |
| Validation | 60 | 15% |
| Test | 60 | 15% |

- **Random seed**: 42
- **Stratified** split to preserve class proportions

### MLP Architecture (Baseline E1)

```
Input (43)  →  Dense(64) + ReLU  →  Dense(32) + ReLU  →  Dense(3) + Softmax
```

- **Activation**: ReLU (hidden), Softmax (output)
- **Loss**: Categorical cross-entropy
- **Optimizer**: SGD with momentum (0.9)
- **Initialisation**: He initialisation

### SGD Training

- **Batch size**: 32 (mini-batch SGD)
- **Learning rate**: 0.01 (E1 baseline)
- **Max epochs**: 300 (with early stopping, patience=30)
- **Momentum**: 0.9

---

## Experiments

| ID | Architecture | LR | Batch Size | Momentum | Purpose |
|---|---|---|---|---|---|
| E1 | [43, 64, 32, 3] | 0.01 | 32 | 0.9 | Baseline MLP |
| E2 | [43, 128, 64, 32, 3] | 0.01 | 32 | 0.9 | Deeper architecture |
| E3 | [43, 64, 32, 3] | 0.05 | 32 | 0.9 | Higher learning rate |
| E4 | [43, 64, 32, 3] | 0.01 | 8 | 0.9 | Smaller batch size |

---

## Results

All results come from the actual trained models evaluated on the **same held-out test set (60 samples, seed=42)**.

### Final Comparison Table

| Model | Accuracy | Macro-F1 | Macro-Recall | Notes |
|---|---|---|---|---|
| Majority-Class Baseline | 0.4833 | 0.2172 | 0.3333 | Predicts Medium always |
| Logistic Regression | 0.5167 | 0.4839 | 0.4833 | Linear baseline |
| **E1 — Baseline MLP** | **0.4167** | **0.3937** | **0.3903** | 2 hidden layers |
| **E2 — Deeper MLP** | **0.4833** | **0.4202** | **0.4208** | 3 hidden layers ← Best MLP |
| **E3 — High LR** | **0.3500** | **0.2884** | **0.2912** | LR=0.05, severe overfitting |
| **E4 — Small Batch** | **0.4500** | **0.3426** | **0.3550** | Batch=8, overfitting |

### Best MLP Model (E2 — Deeper Architecture)

| Metric | Value |
|---|---|
| Accuracy | 0.4833 |
| Macro-F1 | 0.4202 |
| Macro-Recall | 0.4208 |
| Recall — Low | 0.2727 |
| Recall — Medium | 0.3000 |
| Recall — High | 0.6897 |

### Key Observations

- All MLP experiments exhibit **overfitting** on this small 395-sample dataset — training accuracy reaches >98% while validation accuracy stagnates around 45–55%.
- The **Logistic Regression baseline outperforms all MLP models**, which is a scientifically honest result for very small tabular datasets and is discussed in depth in the report.
- **E2 (deeper architecture)** achieves the best Macro-F1 among all MLPs (0.4202), showing that additional capacity helps slightly.
- **E3 (high LR = 0.05)** suffers from divergence and early stopping at epoch 31, confirming that an aggressive learning rate is harmful.
- The **High class** is best recalled (0.69 in E2) because its features (especially `higher=yes`, low `failures`, high parental education) are more distinctive.
- The **Low class** is hardest to predict cleanly because grade-0 students (course withdrawals) and genuinely low performers share similar feature profiles.

---

## Generated Figures

All figures are saved in `results/figures/`:

| File | Description |
|---|---|
| `g3_distribution.png` | Raw G3 histogram + binned class distribution |
| `feature_importances.png` | Top-15 feature correlations with target class |
| `E1_Baseline_MLP_training_curves.png` | Train/val loss and accuracy — E1 |
| `E1_Baseline_MLP_confusion_matrix.png` | Confusion matrix — E1 |
| `E2_Deeper_MLP_training_curves.png` | Train/val loss and accuracy — E2 |
| `E2_Deeper_MLP_confusion_matrix.png` | Confusion matrix — E2 |
| `E3_HighLR_MLP_training_curves.png` | Train/val loss and accuracy — E3 |
| `E3_HighLR_MLP_confusion_matrix.png` | Confusion matrix — E3 |
| `E4_SmallBatch_MLP_training_curves.png` | Train/val loss and accuracy — E4 |
| `E4_SmallBatch_MLP_confusion_matrix.png` | Confusion matrix — E4 |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
- Download the dataset automatically from UCI
- Run preprocessing
- Train both baselines
- Run all 4 MLP experiments
- Save all plots and metrics to `results/`

### 3. Individual modules

```bash
# Preprocessing only
python -m src.data_preprocessing

# Run from individual module
python -c "from src.data_preprocessing import run_preprocessing; run_preprocessing()"
```

---

## Project Structure

```
student-performance-mlp/
│
├── data/
│   └── student-mat.csv              # UCI dataset (auto-downloaded)
│
├── src/
│   ├── data_preprocessing.py        # Download, inspect, encode, split, scale
│   ├── model.py                     # Pure NumPy MLP implementation
│   ├── train.py                     # Mini-batch SGD training loop
│   ├── evaluate.py                  # Metrics, plots, comparison table
│   └── utils.py                     # Error analysis, feature importance
│
├── results/
│   ├── figures/                     # All PNG plots
│   ├── metrics/                     # JSON metrics and histories
│   └── models/                     # (Reserved for saved weights)
│
├── report/
│   └── ANN_Project_Report.md        # Full academic report
│
├── VIVA_PREPARATION.md              # Viva Q&A document
├── main.py                          # Orchestration script
├── requirements.txt
└── README.md
```

---

## Dependencies

| Package | Version |
|---|---|
| numpy | ≥1.24.0 |
| pandas | ≥2.0.0 |
| scikit-learn | ≥1.3.0 |
| matplotlib | ≥3.7.0 |
| seaborn | ≥0.12.0 |
| requests | ≥2.31.0 |

---

## Implementation Notes

- **No deep learning framework used** — the MLP is implemented from scratch in pure NumPy, including forward propagation, backpropagation, and SGD.
- **SGD with momentum** is used (not Adam or RMSProp), as required by the course specification.
- **He initialisation** is used for weight initialisation (appropriate for ReLU activations).
- **Early stopping** is used to prevent excessive overfitting (patience = 30 epochs).
- **Data leakage prevention**: StandardScaler is fitted only on training data.

---

## Dataset Citation

P. Cortez and A. Morais. *A Data Mining Approach to Predict Forest Fires using Meteorological Data* — **Note: the student performance dataset is cited as:**

> P. Cortez and A. M. G. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., *Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)*, pp. 5–12, Porto, Portugal, April 2008, EUROSIS, ISBN 978-9077381-39-7.

UCI ML Repository link: https://archive.ics.uci.edu/dataset/320/student+performance
