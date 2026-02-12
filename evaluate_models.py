#!/usr/bin/env python
# path: sync_data_proj/evaluate_models.py
"""模型评测脚本：在原始/合成数据上训练 Logistic、XGBoost、LightGBM、CatBoost。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "sync_data_proj" / "datasets"


def detect_columns(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != target]
    categorical = [c for c in feature_cols if df[c].dtype == object]
    numeric = [c for c in feature_cols if c not in categorical]
    return numeric, categorical


def build_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = []
    if numeric:
        transformers.append(("num", num_transformer, numeric))
    if categorical:
        transformers.append(("cat", cat_transformer, categorical))
    if not transformers:
        # 没有特征时回退到空转换
        transformers.append(("pass", "passthrough", []))
    return ColumnTransformer(transformers)


def build_models(seed: int) -> Dict[str, object]:
    return {
        "Logistic": LogisticRegression(
            max_iter=2000, solver="saga", penalty="l2", C=1.0, n_jobs=-1, random_state=seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            objective="binary:logistic",
            random_state=seed,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=seed,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            depth=8,
            iterations=800,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=seed,
        ),
    }


def get_probabilities(model, X_test):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        scores = np.asarray(scores)
        if scores.ndim > 1:
            scores = scores[:, 0]
        return scores
    return None


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    y_prob = get_probabilities(model, X_test)
    if y_prob is not None and len(np.unique(y_test)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def scenario_splits(df: pd.DataFrame, test_size: float, seed: int, stratify) -> Tuple:
    return train_test_split(
        df.drop(columns=["label"]),
        df["label"],
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )


def run_for_dataset(dataset_dir: Path, seed: int, test_size: float) -> None:
    print(f"[INFO] 开始评估数据集: {dataset_dir.name}")
    original = pd.read_csv(dataset_dir / "data.csv")
    synthetic = pd.read_csv(dataset_dir / "sync_data.csv")

    numeric_cols, categorical_cols = detect_columns(original, "label")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    base_models = build_models(seed)
    pipelines = {name: Pipeline([("prep", preprocessor), ("clf", model)]) for name, model in base_models.items()}

    results: List[Dict[str, object]] = []

    # 场景 1：原始数据 8:2
    X_train, X_test, y_train, y_test = scenario_splits(original, test_size, seed, original["label"])
    for model_name, pipeline in pipelines.items():
        print(f"[INFO] 场景1 - {model_name}")
        metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
        results.append({"scenario": "original_train_test", "model": model_name, **metrics})

    # 场景 2：合成数据 8:2
    X_syn_train, X_syn_test, y_syn_train, y_syn_test = scenario_splits(synthetic, test_size, seed, synthetic["label"])
    for model_name, pipeline in pipelines.items():
        print(f"[INFO] 场景2 - {model_name}")
        metrics = evaluate_model(pipeline, X_syn_train, y_syn_train, X_syn_test, y_syn_test)
        results.append({"scenario": "synthetic_train_test", "model": model_name, **metrics})

    # 场景 3：合成训练 -> 原始测试
    X_syn_full = synthetic.drop(columns=["label"])
    y_syn_full = synthetic["label"]
    X_orig_full = original.drop(columns=["label"])
    y_orig_full = original["label"]
    for model_name, pipeline in pipelines.items():
        print(f"[INFO] 场景3 - {model_name}")
        metrics = evaluate_model(pipeline, X_syn_full, y_syn_full, X_orig_full, y_orig_full)
        results.append({"scenario": "synthetic_train_original_test", "model": model_name, **metrics})

    df_results = pd.DataFrame(results)
    csv_path = dataset_dir / "model_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    with (dataset_dir / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] {dataset_dir.name} 评估完成，结果保存至 {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对原始/合成数据进行多模型评估")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="指定要评估的数据集目录名称（默认使用 datasets 下全部目录）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2, help="训练/测试划分比例 (默认 0.2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = args.datasets or [p.name for p in DATASETS_DIR.iterdir() if p.is_dir()]
    for name in dataset_names:
        dataset_dir = DATASETS_DIR / name
        if not dataset_dir.exists():
            print(f"[WARN] {dataset_dir} 不存在，跳过。")
            continue
        if not (dataset_dir / "sync_data.csv").exists():
            print(f"[WARN] {dataset_dir} 缺少 sync_data.csv，跳过。")
            continue
        run_for_dataset(dataset_dir, args.seed, args.test_size)


if __name__ == "__main__":
    main()
