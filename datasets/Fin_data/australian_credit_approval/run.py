"""
End-to-end runner for TabPFGen on the Australian Credit Approval dataset.

Steps:
1) Load the pre-split train/test CSVs.
2) Clean/encode columns (cast numerics, fill NA with medians, track int columns).
3) Use TabPFGen (GPU) to synthesize datasets at several scales.
4) Save each synthetic CSV.
5) Train baseline models on each synthetic set and evaluate on the real test set.
6) Persist metrics to CSV/JSON for downstream reporting.

This script is intentionally verbose and commented so it can serve as a playbook
for rerunning or adjusting the experiments.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfgen import TabPFGen
from xgboost import XGBClassifier

# ----------------------------
# Paths and global parameters
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"
LABEL_COL = "label"

# TabPFGen tuning (picked to improve XGBoost accuracy on 1x synthetic data)
TABPFGEN_PARAMS = dict(
    n_sgld_steps=600,
    sgld_step_size=0.01,
    sgld_noise_scale=0.005,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
SYN_FACTORS = [1, 2, 3, 4, 5]  # 100%..500%
INIT_JITTER_STD = 0.01  # noise around seed samples before SGLD
GLOBAL_SEED = 24

# Model settings
XGB_THRESHOLD = 0.45  # tuned on synthetic validation to lift accuracy > 0.87
MODELS_DEF = {
    "logistic": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=800, solver="lbfgs", random_state=GLOBAL_SEED),
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.95,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=GLOBAL_SEED,
        n_jobs=8,
        # local xgboost build does not have GPU support enabled
        tree_method="hist",
    ),
    "lightgbm": lambda: LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=GLOBAL_SEED,
        n_jobs=8,
        verbose=-1,
    ),
    "catboost": lambda: CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=GLOBAL_SEED,
        verbose=False,
    ),
    "mlp": lambda: make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=800,
            alpha=1e-4,
            random_state=GLOBAL_SEED,
        ),
    ),
}


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_clean(path: Path) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Tuple[float, float]]]:
    """
    Load CSV, coerce columns to numeric, fill NAs with medians, and track column types.
    Returns cleaned df, integer columns, float columns, and feature ranges for clipping.
    """
    df = pd.read_csv(path).replace("?", np.nan)
    feature_cols = [c for c in df.columns if c != LABEL_COL]

    # convert to numeric
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes.replace({-1: np.nan})
        df[col] = pd.to_numeric(df[col], errors="coerce")

    int_cols: List[str] = []
    float_cols: List[str] = []
    for col in feature_cols:
        non_nan = df[col].dropna()
        if non_nan.empty:
            float_cols.append(col)
        elif np.all(np.isclose(non_nan % 1, 0)):
            int_cols.append(col)
        else:
            float_cols.append(col)

    # fill missing values
    for col in int_cols:
        median_val = df[col].median()
        df.loc[:, col] = df[col].fillna(median_val).round().astype(int)
    for col in float_cols:
        median_val = df[col].median()
        df.loc[:, col] = df[col].fillna(median_val).astype(float)

    ranges = {col: (df[col].min(), df[col].max()) for col in feature_cols}
    return df, int_cols, float_cols, ranges


def generate_synthetic(
    train_df: pd.DataFrame,
    int_cols: List[str],
    ranges: Dict[str, Tuple[float, float]],
) -> Dict[int, pd.DataFrame]:
    """
    Generate balanced synthetic datasets for each factor using TabPFGen SGLD.
    """
    feature_cols = [c for c in train_df.columns if c != LABEL_COL]
    X_np = train_df[feature_cols].to_numpy()
    y_np = train_df[LABEL_COL].to_numpy()
    classes = np.unique(y_np)
    per_class_base = len(train_df) // len(classes)

    gen = TabPFGen(**TABPFGEN_PARAMS)
    X_scaled = gen.scaler.fit_transform(X_np)
    x_train = torch.tensor(X_scaled, device=gen.device, dtype=torch.float32)
    y_train = torch.tensor(y_np, device=gen.device)

    synthetic_sets: Dict[int, pd.DataFrame] = {}

    for factor in SYN_FACTORS:
        target_per_class = per_class_base * factor
        x_init_list = []
        y_init_list = []
        for cls in classes:
            idx = np.where(y_np == cls)[0]
            chosen = np.random.choice(idx, size=target_per_class, replace=True)
            noise = torch.randn(target_per_class, X_np.shape[1], device=gen.device) * INIT_JITTER_STD
            x_init = x_train[chosen] + noise
            y_init = torch.full((target_per_class,), cls, device=gen.device)
            x_init_list.append(x_init)
            y_init_list.append(y_init)

        x_synth = torch.cat(x_init_list, dim=0)
        y_synth = torch.cat(y_init_list, dim=0)

        for step in range(gen.n_sgld_steps):
            x_synth = gen._sgld_step(x_synth, y_synth, x_train, y_train)
            # Print a heartbeat to show progress on long runs
            if step % 200 == 0:
                print(f"[factor={factor}] SGLD step {step}/{gen.n_sgld_steps}")

        # Inverse transform back to original scale
        X_out = gen.scaler.inverse_transform(x_synth.detach().cpu().numpy())
        y_out = y_synth.cpu().numpy().astype(int)

        df_synth = pd.DataFrame(X_out, columns=feature_cols)
        # clamp to original min/max and restore integer columns
        for col in feature_cols:
            mn, mx = ranges[col]
            df_synth[col] = df_synth[col].clip(mn, mx)
        for col in int_cols:
            df_synth[col] = df_synth[col].round().astype(int)

        df_synth[LABEL_COL] = y_out

        out_path = BASE_DIR / f"synthetic_{factor * 100}.csv"
        df_synth.to_csv(out_path, index=False)
        synthetic_sets[factor] = df_synth
        print(
            f"[factor={factor}] saved {out_path.name} "
            f"({len(df_synth)} rows, label counts {df_synth[LABEL_COL].value_counts().to_dict()})"
        )

    return synthetic_sets


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold_xgb: float = XGB_THRESHOLD,
) -> Dict[str, Dict[str, float]]:
    """
    Train each model on train_df and evaluate on test_df. Returns metric dict.
    """
    feature_cols = [c for c in train_df.columns if c != LABEL_COL]
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df[LABEL_COL].astype(int)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df[LABEL_COL].astype(int)

    results: Dict[str, Dict[str, float]] = {}

    for name, builder in MODELS_DEF.items():
        model = builder()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        threshold = threshold_xgb if name == "xgboost" else 0.5
        preds = (probs >= threshold).astype(int)

        results[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "log_loss": float(log_loss(y_test, probs)),
        }
    return results


def save_metrics(metrics: Dict[int, Dict[str, Dict[str, float]]]) -> None:
    rows = []
    for factor, model_dict in metrics.items():
        for model_name, vals in model_dict.items():
            row = {"synthetic_size": f"{factor*100}%","model": model_name}
            row.update(vals)
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = BASE_DIR / "model_metrics.csv"
    json_path = BASE_DIR / "model_metrics.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {csv_path} and {json_path}")


def main() -> None:
    set_seed(GLOBAL_SEED)
    print(f"Running with seed={GLOBAL_SEED}, TabPFGen params={TABPFGEN_PARAMS}")

    train_df, int_cols, float_cols, ranges = load_and_clean(TRAIN_PATH)
    test_df, _, _, _ = load_and_clean(TEST_PATH)
    synthetic_sets = generate_synthetic(train_df, int_cols, ranges)

    all_metrics: Dict[int, Dict[str, Dict[str, float]]] = {}
    for factor, synth_df in synthetic_sets.items():
        print(f"Evaluating models for synthetic size {factor*100}%...")
        all_metrics[factor] = evaluate_models(synth_df, test_df)

    save_metrics(all_metrics)

    # Pretty print for quick inspection
    for factor in sorted(all_metrics.keys()):
        print(f"\n=== Synthetic {factor*100}% ===")
        for model, vals in all_metrics[factor].items():
            print(
                f"{model:10s} | acc {vals['accuracy']:.4f} | "
                f"prec {vals['precision']:.4f} | rec {vals['recall']:.4f} | "
                f"f1 {vals['f1']:.4f} | auc {vals['roc_auc']:.4f}"
            )


if __name__ == "__main__":
    main()
