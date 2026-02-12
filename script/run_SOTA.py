#!/usr/bin/env python3
"""
Train/evaluate baseline models (Logistic, XGBoost, LightGBM, CatBoost, MLP)
on a synthetic training CSV and evaluate on an original test CSV.
Outputs accuracy, balanced accuracy (as adjusted accuracy), and AUROC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


MODEL_NAMES = ["logistic", "xgboost", "lightgbm", "catboost", "mlp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOTA baselines on tabular data.")
    parser.add_argument("--train-csv", required=True, help="Training CSV (synthetic).")
    parser.add_argument("--test-csv", required=True, help="Test CSV (original).")
    parser.add_argument("--target", default=None, help="Target column name (default: last column).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        help=f"Models to run, subset of {MODEL_NAMES}.",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Optional JSON string mapping model name to param dict.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel threads for supported models.")
    parser.add_argument("--report-path", default=None, help="Optional CSV path to append results.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Optional validation split from training set (0 disables; only affects metrics if >0).",
    )
    return parser.parse_args()


def set_global_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_dataset(path: str, target: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, str]:
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    if target is None:
        target = df.columns[-1]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y, target


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )


def build_model(
    name: str,
    seed: int,
    n_jobs: int,
    overrides: Optional[Dict],
    n_classes: int,
) -> object:
    params = overrides or {}
    is_multiclass = n_classes > 2
    if name == "logistic":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            multi_class="auto",
            n_jobs=n_jobs,
            random_state=seed,
            **params,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob" if is_multiclass else "binary:logistic",
            eval_metric="mlogloss" if is_multiclass else "logloss",
            num_class=n_classes if is_multiclass else None,
            n_jobs=n_jobs,
            random_state=seed,
            **params,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass" if is_multiclass else "binary",
            num_class=n_classes if is_multiclass else None,
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass" if is_multiclass else "Logloss",
            random_seed=seed,
            thread_count=n_jobs if n_jobs != 0 else None,
            verbose=False,
            **params,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=seed,
            **params,
        )
    raise ValueError(f"Unknown model '{name}'")


def safe_auroc(model: Pipeline, X: pd.DataFrame, y_true: pd.Series) -> Optional[float]:
    if len(np.unique(y_true)) != 2:
        return None
    try:
        proba = model.predict_proba(X)
        clf = model.named_steps.get("clf")
        classes = getattr(clf, "classes_", None)
        if classes is None or len(classes) < 2:
            return None
        pos_index = list(classes).index(classes[-1])
        scores = proba[:, pos_index]
        return roc_auc_score(y_true, scores)
    except Exception:
        try:
            scores = model.decision_function(X)
            return roc_auc_score(y_true, scores)
        except Exception:
            return None


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
    }
    auroc = safe_auroc(model, X, y)
    if auroc is not None:
        metrics["auroc"] = float(auroc)
    return metrics


def run_models(
    train_csv: str,
    test_csv: str,
    target: Optional[str],
    model_names: Iterable[str],
    params: Dict[str, Dict],
    seed: int,
    n_jobs: int,
    val_split: float,
) -> List[Dict]:
    X_train, y_train_raw, target = load_dataset(train_csv, target)
    X_test, y_test_raw, _ = load_dataset(test_csv, target)

    # Encode labels to contiguous integers to support multiclass estimators (e.g., XGBoost).
    all_labels = pd.Categorical(pd.concat([y_train_raw, y_test_raw], axis=0))
    categories = list(all_labels.categories)
    label_to_code = {cat: i for i, cat in enumerate(categories)}
    y_train = y_train_raw.map(label_to_code)
    y_test = y_test_raw.map(label_to_code)

    num_cols, cat_cols = split_columns(X_train)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    n_classes = len(categories)

    results: List[Dict] = []
    for name in model_names:
        clf = build_model(
            name,
            seed=seed,
            n_jobs=n_jobs,
            overrides=params.get(name),
            n_classes=n_classes,
        )
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        if val_split > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_split,
                random_state=seed,
                stratify=y_train,
            )
            pipe.fit(X_tr, y_tr)
            val_metrics = evaluate(pipe, X_val, y_val)
        else:
            pipe.fit(X_train, y_train)
            val_metrics = {}

        test_metrics = evaluate(pipe, X_test, y_test)
        row = {
            "model": name,
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
            "target": target,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        results.append(row)
    return results


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    params: Dict[str, Dict] = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse --params JSON: {exc}") from exc

    selected_models = []
    for m in args.models:
        if m not in MODEL_NAMES:
            raise SystemExit(f"Unknown model '{m}', choose from {MODEL_NAMES}")
        selected_models.append(m)

    results = run_models(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        target=args.target,
        model_names=selected_models,
        params=params,
        seed=args.seed,
        n_jobs=args.n_jobs,
        val_split=args.val_split,
    )

    df = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    print(df)

    if args.report_path:
        out_path = Path(args.report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, mode="a", index=False, header=not out_path.exists())
        print(f"\nMetrics appended to {out_path}")


if __name__ == "__main__":
    main()
