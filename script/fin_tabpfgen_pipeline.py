"""
TabPFGen pipeline: 5-split TSTR with TabPFGen.

What this script does
- For each dataset folder, load `data.csv` (Fin) or `dataset.csv` (EHR).
- Ensure there is exactly one label column named `label` (defaults to last column).
- Run 5 splits (prefer StratifiedKFold=5 when feasible, else StratifiedShuffleSplit with 80/20).
- Train downstream models on:
  - real train -> evaluate on real test (baseline)
  - TabPFGen synthetic train -> evaluate on real test (TSTR, synthetic)
- Report mean ± std over the 5 splits.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Thread / BLAS safety
# ---------------------------------------------------------------------
# On high-core machines, OpenBLAS may try to spawn >128 threads and crash.
# These env vars must be set *before* importing NumPy.
_DEFAULT_THREADS = os.environ.get("SYNC_PIPELINE_CPU_THREADS", "32")
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, _DEFAULT_THREADS)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_POSSIBLE_SRC = [Path(__file__).resolve().parents[i] / "src" for i in range(1, 4)]
for p in _POSSIBLE_SRC:
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))
try:
    from generator import TabPFNConditionalGenerator  # type: ignore
except Exception:
    TabPFNConditionalGenerator = None  # type: ignore

LABEL_COL = "label"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]
MODELS = ["logistic", "xgboost", "lightgbm", "catboost", "mlp"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def find_dataset_csv(dataset_dir: Path) -> Optional[Path]:
    """Return the dataset CSV path for a dataset folder."""
    for name in ("data.csv", "dataset.csv"):
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    return None


def expand_dataset_dirs(raw_paths: List[str]) -> List[Path]:
    """Expand input paths into concrete dataset folders.

    Users may pass:
    - a dataset folder that directly contains `data.csv` or `dataset.csv`
    - a *root folder* that contains multiple dataset subfolders
      (e.g. `.../datasets/EHR_datasets/` or `.../datasets/Fin_data/`).

    This function keeps ordering deterministic and de-duplicates paths.
    """
    expanded: List[Path] = []
    seen: set[str] = set()

    for raw in raw_paths:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            print(f"[skip] {p} does not exist", file=sys.stderr)
            continue
        if not p.is_dir():
            print(f"[skip] {p} is not a directory", file=sys.stderr)
            continue

        if find_dataset_csv(p) is not None:
            key = str(p)
            if key not in seen:
                expanded.append(p)
                seen.add(key)
            continue

        children = [c for c in p.iterdir() if c.is_dir() and not c.name.startswith(".")]
        candidates = [c for c in children if find_dataset_csv(c) is not None]
        candidates.sort(key=lambda x: x.name)
        if not candidates:
            print(f"[skip] {p} has no data.csv/dataset.csv and no dataset subfolders", file=sys.stderr)
            continue
        for c in candidates:
            key = str(c)
            if key not in seen:
                expanded.append(c)
                seen.add(key)

    return expanded


def ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the label column is named `label`. If missing, treat the last column as label."""
    if LABEL_COL in df.columns:
        return df
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 1 feature column + 1 label column.")
    return df.rename(columns={df.columns[-1]: LABEL_COL})


def detect_int_float(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    int_cols: List[str] = []
    float_cols: List[str] = []
    for col in df.columns:
        if col == LABEL_COL:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        non_nan = series.dropna()
        if non_nan.empty:
            float_cols.append(col)
        elif np.all(np.isclose(non_nan % 1, 0)):
            int_cols.append(col)
        else:
            float_cols.append(col)
    return int_cols, float_cols


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Tuple[float, float]]]:
    df = df.replace("?", np.nan).copy()
    df = ensure_label_column(df)

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes.replace({-1: np.nan})
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Label: allow numeric or string labels (e.g. "ckd"/"notckd").
    label_raw = df[LABEL_COL]
    label_numeric = pd.to_numeric(label_raw, errors="coerce")
    if label_numeric.isna().all():
        codes, _ = pd.factorize(label_raw.astype(str), sort=True)
        df[LABEL_COL] = codes.astype(int)
    else:
        if label_numeric.isna().any():
            codes, _ = pd.factorize(label_raw.astype(str), sort=True)
            df[LABEL_COL] = codes.astype(int)
        else:
            df[LABEL_COL] = label_numeric.astype(int)

    fallback = int(df[LABEL_COL].mode(dropna=True).iloc[0]) if not df[LABEL_COL].mode(dropna=True).empty else 0
    df[LABEL_COL] = df[LABEL_COL].fillna(fallback).astype(int)

    int_cols, float_cols = detect_int_float(df)
    for col in int_cols:
        med = df[col].median()
        df[col] = df[col].fillna(med).round().astype(int)
    for col in float_cols:
        med = df[col].median()
        df[col] = df[col].fillna(med).astype(float)
    ranges = {c: (df[c].min(), df[c].max()) for c in feature_cols}
    return df, int_cols, float_cols, ranges


def postprocess(df: pd.DataFrame, int_cols: List[str], ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col in [c for c in out.columns if c != LABEL_COL]:
        mn, mx = ranges[col]
        out[col] = out[col].clip(mn, mx)
    for col in int_cols:
        out[col] = out[col].round().astype(int)
    return out


def make_models(seed: int, n_classes: int) -> Dict[str, Pipeline]:
    is_multiclass = n_classes > 2
    return {
        "logistic": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=800, solver="lbfgs", random_state=seed)),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob" if is_multiclass else "binary:logistic",
                        num_class=n_classes if is_multiclass else None,
                        eval_metric="mlogloss" if is_multiclass else "logloss",
                        random_state=seed,
                        n_jobs=8,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multiclass" if is_multiclass else "binary",
                        num_class=n_classes if is_multiclass else None,
                        random_state=seed,
                        n_jobs=8,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "catboost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    CatBoostClassifier(
                        iterations=400,
                        depth=6,
                        learning_rate=0.05,
                        loss_function="MultiClass" if is_multiclass else "Logloss",
                        random_seed=seed,
                        verbose=False,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=800,
                        alpha=1e-4,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def eval_models(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> Dict[str, Dict[str, float]]:
    y_train = train_df[LABEL_COL].to_numpy()
    y_test = test_df[LABEL_COL].to_numpy()
    classes = np.unique(y_train)
    n_classes = len(classes)
    models = make_models(seed, n_classes=n_classes)
    X_train = train_df.drop(columns=[LABEL_COL]).to_numpy()
    X_test = test_df.drop(columns=[LABEL_COL]).to_numpy()

    average = "binary" if n_classes == 2 else "macro"
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        metrics = {k: math.nan for k in METRICS}
        try:
            if n_classes < 2:
                results[name] = metrics
                continue
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["precision"] = float(precision_score(y_test, y_pred, average=average, zero_division=0))
            metrics["recall"] = float(recall_score(y_test, y_pred, average=average, zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, average=average, zero_division=0))
            try:
                proba = model.predict_proba(X_test)
                if n_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
                metrics["log_loss"] = float(log_loss(y_test, proba, labels=classes))
            except Exception:
                metrics["roc_auc"] = math.nan
                metrics["log_loss"] = math.nan
        except Exception:
            pass
        results[name] = metrics
    return results


def generate_tabpfgen(
    train_df: pd.DataFrame,
    int_cols: List[str],
    ranges: Dict[str, Tuple[float, float]],
    sgld_steps: int,
    step_size: float,
    noise_scale: float,
    jitter: float,
    factor: float,
    seed: int,
    energy_subsample: Optional[int],
) -> pd.DataFrame:
    try:
        from tabpfgen import TabPFGen
        import torch
    except ModuleNotFoundError as e:
        raise ImportError("tabpfgen is required for TabPFGen generator. Activate tabpfgen env.") from e
    set_seed(seed)
    feature_cols = [c for c in train_df.columns if c != LABEL_COL]
    X_np = train_df[feature_cols].to_numpy()
    y_np = train_df[LABEL_COL].to_numpy()
    classes, counts = np.unique(y_np, return_counts=True)
    probs = counts / counts.sum()

    gen = TabPFGen(
        n_sgld_steps=sgld_steps,
        sgld_step_size=step_size,
        sgld_noise_scale=noise_scale,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    X_scaled = gen.scaler.fit_transform(X_np)
    device = gen.device
    x_train = torch.tensor(X_scaled, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_np, device=device)

    # Optional speed knob: use a subset of the real train set when computing the SGLD energy.
    # This can drastically speed up large datasets, while keeping scaling + initialization based on all train rows.
    x_energy = x_train
    y_energy = y_train
    if energy_subsample is not None and energy_subsample > 0 and len(train_df) > energy_subsample:
        sub_idx = np.random.choice(len(train_df), size=energy_subsample, replace=False)
        x_energy = x_train[sub_idx]
        y_energy = y_train[sub_idx]

    target_size = int(round(len(train_df) * factor))
    target_counts = {cls: int(round(target_size * p)) for cls, p in zip(classes, probs)}
    diff = target_size - sum(target_counts.values())
    if diff != 0:
        target_counts[classes[0]] += diff

    x_list = []
    y_list = []
    for cls in classes:
        n = target_counts[cls]
        idx = np.where(y_np == cls)[0]
        chosen = np.random.choice(idx, size=n, replace=True)
        noise = torch.randn(n, X_np.shape[1], device=device) * jitter
        x_init = x_train[chosen] + noise
        y_init = torch.full((n,), cls, device=device)
        x_list.append(x_init)
        y_list.append(y_init)

    x_synth = torch.cat(x_list, dim=0)
    y_synth = torch.cat(y_list, dim=0)

    for _ in range(gen.n_sgld_steps):
        x_synth = gen._sgld_step(x_synth, y_synth, x_energy, y_energy)

    X_out = gen.scaler.inverse_transform(x_synth.detach().cpu().numpy())
    y_out = y_synth.cpu().numpy()

    syn_df = pd.DataFrame(X_out, columns=feature_cols)
    syn_df[LABEL_COL] = y_out.astype(int)
    syn_df = postprocess(syn_df, int_cols, ranges)
    return syn_df


def _safe_mean_std(values: np.ndarray) -> Tuple[float, float]:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return math.nan, math.nan
    if valid.size == 1:
        return float(valid.mean()), 0.0
    return float(valid.mean()), float(valid.std(ddof=1))


def aggregate_metrics_mean_std(per_split: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    agg: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for model in MODELS:
        rows = []
        for split_res in per_split:
            rows.append([split_res[model][m] for m in METRICS])
        arr = np.array(rows, dtype=float)
        agg[model] = {m: _safe_mean_std(arr[:, i]) for i, m in enumerate(METRICS)}
    return agg


def iter_splits(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> Iterable[Tuple[np.ndarray, np.ndarray, str]]:
    """Yield (train_idx, test_idx, split_name). Prefer 5-fold stratified CV; fallback to 5 random stratified splits."""
    # 5-fold stratified CV requires each class to appear at least n_splits times.
    min_class = int(pd.Series(y).value_counts().min())
    if min_class >= n_splits:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (tr, te) in enumerate(skf.split(X, y), start=1):
            yield tr, te, f"StratifiedKFold fold {i}/{n_splits}"
        return
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=seed)
    for i, (tr, te) in enumerate(sss.split(X, y), start=1):
        yield tr, te, f"StratifiedShuffleSplit split {i}/{n_splits} (min_class={min_class})"


def run_dataset(
    dataset_dir: Path,
    csv_path: Path,
    sgld_steps: int,
    step_size: float,
    noise_scale: float,
    jitter: float,
    factor: float,
    seed: int,
    energy_subsample: Optional[int],
    generators: List[str],
) -> Dict:
    raw_df = pd.read_csv(csv_path)
    df, int_cols, float_cols, ranges = clean_dataframe(raw_df)
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    folds = []
    fold_id = 0
    X_all = df[feature_cols].to_numpy()
    y_all = df[LABEL_COL].to_numpy()

    for train_idx, test_idx, split_name in iter_splits(X_all, y_all, seed=seed, n_splits=5):
        fold_id += 1
        real_train = df.iloc[train_idx].reset_index(drop=True)
        real_test = df.iloc[test_idx].reset_index(drop=True)

        baseline = eval_models(real_train, real_test, seed)
        fold_entry = {
            "fold": fold_id,
            "split": split_name,
            "baseline": baseline,
            "gens": {},
            "counts": {"real_train": len(real_train), "real_test": len(real_test)},
        }

        syn_cache: Dict[str, pd.DataFrame] = {}
        if "tabpfgen" in generators:
            print(f"  [fold {fold_id}/5] generating tabpfgen synthetic...", flush=True)
            syn_cache["tabpfgen"] = generate_tabpfgen(
                real_train,
                int_cols,
                ranges,
                sgld_steps,
                step_size,
                noise_scale,
                jitter,
                factor,
                seed + fold_id,
                energy_subsample=energy_subsample,
            )

        for gname, gtrain in syn_cache.items():
            fold_entry["counts"][f"{gname}_train"] = len(gtrain)
            synth_only = eval_models(gtrain, real_test, seed)
            fold_entry["gens"][gname] = {"synthetic": synth_only}
            print(f"  [fold {fold_id}/5] evaluated {gname} synthetic -> real test", flush=True)

        folds.append(fold_entry)

    gens_in_result = list(folds[0]["gens"].keys()) if folds and folds[0].get("gens") else []
    return {
        "dataset": dataset_dir.name,
        "path": str(dataset_dir),
        "data_file": str(csv_path),
        "samples": len(df),
        "folds": folds,
        "summary": {
            "baseline": aggregate_metrics_mean_std([f["baseline"] for f in folds]),
            "generators": {
                g: {
                    "synthetic": aggregate_metrics_mean_std([f["gens"][g]["synthetic"] for f in folds]),
                }
                for g in gens_in_result
            },
        },
        "params": {
            "tabpfgen": {
                "n_sgld_steps": sgld_steps,
                "sgld_step_size": step_size,
                "sgld_noise_scale": noise_scale,
                "jitter": jitter,
                "synthetic_factor": factor,
                "energy_subsample": energy_subsample,
            },
            "seed": seed,
        },
    }


def write_markdown(results: List[Dict], out_path: Path) -> None:
    title = "# TabPFGen Pipeline Results"

    def render_dataset_section(res: Dict) -> List[str]:
        lines: List[str] = []
        lines.append(f"## {res['dataset']}")
        lines.append(f"- path: `{res['path']}`")
        lines.append(f"- data: `{res['data_file']}`")
        lines.append(f"- samples: {res['samples']}")
        params = res["params"]
        lines.append(f"- TabPFGen: {params['tabpfgen']}")
        lines.append(f"- seed: {params['seed']}")
        lines.append("")
        lines.append("| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for model in MODELS:
            row = res["summary"]["baseline"][model]
            lines.append(
                f"| baseline | real | {model} | "
                f"{row['accuracy'][0]:.4f} ± {row['accuracy'][1]:.4f} | "
                f"{row['precision'][0]:.4f} ± {row['precision'][1]:.4f} | "
                f"{row['recall'][0]:.4f} ± {row['recall'][1]:.4f} | "
                f"{row['f1'][0]:.4f} ± {row['f1'][1]:.4f} | "
                f"{row['roc_auc'][0]:.4f} ± {row['roc_auc'][1]:.4f} | "
                f"{row['log_loss'][0]:.4f} ± {row['log_loss'][1]:.4f} |"
            )
        for gen_name, gen_res in res["summary"]["generators"].items():
            for model in MODELS:
                row = gen_res["synthetic"][model]
                lines.append(
                    f"| {gen_name} | synthetic | {model} | "
                    f"{row['accuracy'][0]:.4f} ± {row['accuracy'][1]:.4f} | "
                    f"{row['precision'][0]:.4f} ± {row['precision'][1]:.4f} | "
                    f"{row['recall'][0]:.4f} ± {row['recall'][1]:.4f} | "
                    f"{row['f1'][0]:.4f} ± {row['f1'][1]:.4f} | "
                    f"{row['roc_auc'][0]:.4f} ± {row['roc_auc'][1]:.4f} | "
                    f"{row['log_loss'][0]:.4f} ± {row['log_loss'][1]:.4f} |"
                )
        lines.append("")
        return lines

    def parse_existing(text: str) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
        lines = text.splitlines()
        header: List[str] = []
        sections: List[Tuple[str, List[str]]] = []
        current_name: Optional[str] = None
        current_lines: List[str] = []

        for line in lines:
            if line.startswith("## "):
                if current_name is None:
                    header = current_lines
                else:
                    sections.append((current_name, current_lines))
                current_name = line[3:].strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_name is None:
            header = current_lines
        else:
            sections.append((current_name, current_lines))
        return header, sections

    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines: List[str] = []
    sections_in_order: List[str] = []
    sections_map: Dict[str, List[str]] = {}

    if out_path.exists():
        existing_text = out_path.read_text(encoding="utf-8", errors="ignore")
        existing_header, existing_sections = parse_existing(existing_text)
        header_lines = existing_header if any(line.strip() for line in existing_header) else [title, ""]
        for name, sec_lines in existing_sections:
            sections_in_order.append(name)
            sections_map[name] = sec_lines
    else:
        header_lines = [title, ""]

    # Upsert new results (append new datasets; overwrite existing dataset section if present)
    for res in results:
        name = str(res["dataset"])
        sections_map[name] = render_dataset_section(res)
        if name not in sections_in_order:
            sections_in_order.append(name)

    # Rebuild
    out_lines: List[str] = list(header_lines)
    if out_lines and out_lines[-1].strip() != "":
        out_lines.append("")
    for name in sections_in_order:
        out_lines.extend(sections_map[name])
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")

    out_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabPFGen 5-fold TSTR pipeline.")
    parser.add_argument("--datasets", nargs="+", required=True, help="List of dataset folders containing data.csv or dataset.csv")
    parser.add_argument("--results-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/tabpfgen.md"))
    parser.add_argument("--sgld-steps", type=int, default=600)
    parser.add_argument("--sgld-step-size", type=float, default=0.01)
    parser.add_argument("--sgld-noise-scale", type=float, default=0.005)
    parser.add_argument("--jitter", type=float, default=0.01)
    parser.add_argument("--synthetic-factor", type=float, default=1.0)
    parser.add_argument(
        "--energy-subsample",
        type=int,
        default=None,
        help="Optional: subsample size of real train rows used in SGLD energy computation (speeds up large datasets).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["tabpfgen"],
        help="Subset of generators to run (tabpfgen).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    results = []
    for ds_path in expand_dataset_dirs(args.datasets):
        csv_path = find_dataset_csv(ds_path)
        if csv_path is None:
            print(f"[skip] {ds_path} has no data.csv/dataset.csv", file=sys.stderr)
            continue
        print(f"\n[dataset] {ds_path.name} ({csv_path.name})", flush=True)
        res = run_dataset(
            ds_path,
            csv_path=csv_path,
            sgld_steps=args.sgld_steps,
            step_size=args.sgld_step_size,
            noise_scale=args.sgld_noise_scale,
            jitter=args.jitter,
            factor=args.synthetic_factor,
            seed=args.seed,
            energy_subsample=args.energy_subsample,
            generators=args.generators,
        )
        results.append(res)
        # Write incrementally so long runs still produce partial output.
        write_markdown(results, args.results_md)
    if not results:
        print("No datasets processed.", file=sys.stderr)
        sys.exit(1)
    print(f"Done. Results written to {args.results_md}")


if __name__ == "__main__":
    main()
