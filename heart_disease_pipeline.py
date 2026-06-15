"""
================================================================================
PMIT-6121: Machine Learning
Comparative Analysis of SVM, KNN, and Naive Bayes Classifiers
Benchmarking ML Classifiers for Heart Disease Prediction
--------------------------------------------------------------------------------
Author : Md. Tarequl Islam Mizi  (ID: 252008)
Dataset: CDC BRFSS 2020 - Personal Key Indicators of Heart Disease (Kaggle)
         https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
--------------------------------------------------------------------------------
End-to-end pipeline:
    1. Load + EDA
    2. Preprocessing (impute -> encode -> split -> scale)
    3. Train 10 model configurations (SVM x6, KNN x3, GNB x1)
    4. Evaluate (accuracy / precision / recall / F1 on positive class)
    5. Confusion matrices + comparison table

Usage:
    python heart_disease_pipeline.py --data heart_2020_cleaned.csv
================================================================================
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
SVM_SUBSAMPLE = 30_000   # SVM is O(n^2)-O(n^3); subsample for tractability


# ----------------------------------------------------------------------------
# 1. LOAD + EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------
def load_data(path):
    """Load the BRFSS 2020 cleaned CSV."""
    df = pd.read_csv(path)
    print(f"[LOAD] shape = {df.shape}")
    print(f"[LOAD] missing values total = {df.isnull().sum().sum()}")
    return df


def run_eda(df):
    """Print core EDA statistics referenced in the report."""
    counts = df["HeartDisease"].value_counts()
    total = len(df)
    print("\n[EDA] Class distribution:")
    for label, n in counts.items():
        print(f"       {label:>3}: {n:>7,}  ({100 * n / total:5.2f}%)")
    ratio = counts.get("No", 0) / max(counts.get("Yes", 1), 1)
    print(f"[EDA] Imbalance ratio (No:Yes) = {ratio:.1f} : 1")

    # Heart disease prevalence by age group
    if "AgeCategory" in df.columns:
        age_rate = (
            df.assign(hd=(df["HeartDisease"] == "Yes").astype(int))
            .groupby("AgeCategory")["hd"].mean().mul(100).round(1)
        )
        print("\n[EDA] HD prevalence by age group (%):")
        print(age_rate.to_string())

    # Prevalence by comorbidity
    print("\n[EDA] HD prevalence by comorbidity (%):")
    for col in ["Stroke", "KidneyDisease", "Diabetic", "SkinCancer", "Asthma"]:
        if col in df.columns:
            sub = df[df[col].astype(str).str.startswith("Yes")]
            if len(sub):
                rate = 100 * (sub["HeartDisease"] == "Yes").mean()
                print(f"       {col:<14}: {rate:5.1f}%")


# ----------------------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------------------
def preprocess(df):
    """
    Impute (defensive) -> encode target -> one-hot encode features -> split.
    Returns un-scaled splits; scaling is applied per-model downstream because
    GNB does not require it (see report Section 2.4).
    """
    df = df.copy()

    # --- Defensive imputation (Kaggle-cleaned data has 0 NaNs, but raw exports may not) ---
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols_all = df.select_dtypes(include="object").columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())          # median: robust to outliers
    for c in cat_cols_all:
        df[c] = df[c].fillna(df[c].mode()[0])          # mode for categoricals

    # --- Encode binary target: Yes -> 1, No -> 0 ---
    df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0})

    # --- One-hot encode all remaining categorical columns ---
    # drop_first=True avoids the dummy-variable trap (multicollinearity).
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"\n[PREP] feature matrix expanded to {df_enc.shape[1] - 1} numerical dimensions")

    X = df_enc.drop(columns="HeartDisease")
    y = df_enc["HeartDisease"]

    # --- Stratified 80/20 split preserves the 91.4% / 8.6% ratio ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[PREP] train = {len(X_train):,}  | test = {len(X_test):,}")
    print(f"[PREP] train positive = {100 * y_train.mean():.2f}%  "
          f"| test positive = {100 * y_test.mean():.2f}%")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    StandardScaler fitted on TRAIN ONLY, then applied to test (no leakage).
    Required for distance-based SVM and KNN.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit on train only
    X_test_sc = scaler.transform(X_test)          # transform only
    return X_train_sc, X_test_sc


# ----------------------------------------------------------------------------
# 3. MODEL CONFIGURATIONS
# ----------------------------------------------------------------------------
def build_models():
    """Return the 10 model configurations as (name, hyperparams, estimator, needs_scaling)."""
    configs = []

    # --- SVM: 2 kernels x 3 C values = 6 configs (class_weight balances 10.7:1 imbalance) ---
    for kernel in ("linear", "rbf"):
        for C in (0.1, 1.0, 10.0):
            configs.append((
                "SVM", f"{kernel.capitalize()}, C={C:g}",
                SVC(kernel=kernel, C=C, class_weight="balanced",
                    random_state=RANDOM_STATE),
                True,   # needs scaling
            ))

    # --- KNN: 3 neighbourhood sizes ---
    for k in (3, 5, 11):
        configs.append((
            "KNN", f"K={k}",
            KNeighborsClassifier(n_neighbors=k),
            True,   # needs scaling
        ))

    # --- Gaussian Naive Bayes: scale-invariant -> no scaling ---
    configs.append((
        "GNB", "Default",
        GaussianNB(var_smoothing=1e-9),
        False,  # does NOT need scaling
    ))
    return configs


# ----------------------------------------------------------------------------
# 4. TRAIN + EVALUATE
# ----------------------------------------------------------------------------
def evaluate_all(X_train, X_test, y_train, y_test):
    """Train every configuration and collect positive-class metrics."""
    X_train_sc, X_test_sc = scale_features(X_train, X_test)
    results = []

    for family, hp, model, needs_scaling in build_models():
        if needs_scaling:
            Xtr, Xte = X_train_sc, X_test_sc
        else:
            Xtr, Xte = X_train.values, X_test.values

        # SVM: subsample training set for computational tractability
        if family == "SVM" and len(Xtr) > SVM_SUBSAMPLE:
            rng = np.random.RandomState(RANDOM_STATE)
            idx = rng.choice(len(Xtr), SVM_SUBSAMPLE, replace=False)
            Xtr_fit, ytr_fit = Xtr[idx], y_train.values[idx]
        else:
            Xtr_fit, ytr_fit = Xtr, y_train.values

        model.fit(Xtr_fit, ytr_fit)
        y_pred = model.predict(Xte)

        results.append({
            "Model": family,
            "Hyperparameters": hp,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=1),
            "Recall": recall_score(y_test, y_pred, pos_label=1),
            "F1-Score": f1_score(y_test, y_pred, pos_label=1),
            "_cm": confusion_matrix(y_test, y_pred),
            "_pred": y_pred,
        })
        print(f"[EVAL] {family:<4} {hp:<14} "
              f"acc={results[-1]['Accuracy']:.4f}  "
              f"prec={results[-1]['Precision']:.4f}  "
              f"rec={results[-1]['Recall']:.4f}  "
              f"f1={results[-1]['F1-Score']:.4f}")
    return results


def print_comparison_table(results):
    df = pd.DataFrame(results)[
        ["Model", "Hyperparameters", "Accuracy", "Precision", "Recall", "F1-Score"]
    ]
    print("\n" + "=" * 78)
    print("CLASSIFICATION PERFORMANCE - ALL CONFIGURATIONS (positive class = Yes)")
    print("=" * 78)
    print(df.round(4).to_string(index=False))


def plot_confusion_matrices(results, out="confusion_matrices.png"):
    """Plot best config per family side-by-side."""
    best = {}
    for r in results:
        fam = r["Model"]
        if fam not in best or r["F1-Score"] > best[fam]["F1-Score"]:
            best[fam] = r
    fams = [f for f in ("SVM", "KNN", "GNB") if f in best]

    fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4.5))
    if len(fams) == 1:
        axes = [axes]
    for ax, fam in zip(axes, fams):
        r = best[fam]
        cm = r["_cm"]
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        ax.set_title(f"{fam} ({r['Hyperparameters']})\nF1={r['F1-Score']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[PLOT] saved {out}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Heart disease classifier benchmark")
    ap.add_argument("--data", default="heart_2020_cleaned.csv",
                    help="path to BRFSS 2020 cleaned CSV")
    args = ap.parse_args()

    df = load_data(args.data)
    run_eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    results = evaluate_all(X_train, X_test, y_train, y_test)
    print_comparison_table(results)
    plot_confusion_matrices(results)

    # Detailed classification report for the recommended screening model (GNB)
    gnb = next(r for r in results if r["Model"] == "GNB")
    print("\n[REPORT] Gaussian Naive Bayes - full classification report:")
    print(classification_report(y_test, gnb["_pred"],
                                target_names=["No HD", "HD"], digits=4))


if __name__ == "__main__":
    main()
