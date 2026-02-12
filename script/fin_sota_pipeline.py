"""
SOTA generators pipeline (GaussianCopula, SDV CTGAN/TVAE, SynthCity DDPM, TabPFN conditional).

This script focuses on *non-TabPFGen* synthesizers and reports:
- TSTR (train on synthetic train, test on real test)
- mean ± std over 5 splits (80/20 each)

Note: some datasets (e.g. Arrhythmia) cannot support StratifiedKFold=5 due to rare classes.
We fallback to 5 stratified shuffle splits in that case.
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
# On high-core machines, OpenBLAS may try to spawn >128 threads and crash
# with "Program is Terminated... maximum of 128 threads" (sometimes ending
# in a segfault). These env vars must be set *before* importing NumPy.
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
      (e.g. `.../datasets/Fin_data/`).

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

        # Treat as a root folder: include direct children that are dataset folders.
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


def _make_metadata(train_df: pd.DataFrame):
    """Create SDV metadata for a single table."""
    try:
        from sdv.metadata import SingleTableMetadata
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ImportError("sdv is required for GaussianCopula/CTGAN/TVAE. Install via `pip install sdv`.") from e

    md = SingleTableMetadata()
    md.detect_from_dataframe(train_df)
    md.update_column(LABEL_COL, sdtype="categorical")
    return md


def gen_gaussian_copula(train_df: pd.DataFrame, int_cols, ranges) -> pd.DataFrame:
    from sdv.single_table import GaussianCopulaSynthesizer

    md = _make_metadata(train_df)
    model = GaussianCopulaSynthesizer(md, enforce_rounding=False)
    model.fit(train_df)
    syn = model.sample(num_rows=len(train_df))
    return postprocess(syn, int_cols, ranges)


def gen_ctgan(
    train_df: pd.DataFrame,
    int_cols,
    ranges,
    epochs: int,
    batch_size: int,
    pac: int,
    seed: int,
    use_cuda: bool,
) -> pd.DataFrame:
    from sdv.single_table import CTGANSynthesizer

    md = _make_metadata(train_df)
    set_seed(seed)
    model = CTGANSynthesizer(md, epochs=epochs, batch_size=batch_size, pac=pac, verbose=False, cuda=use_cuda)
    model.fit(train_df)
    syn = model.sample(num_rows=len(train_df))
    return postprocess(syn, int_cols, ranges)


def gen_tvae(
    train_df: pd.DataFrame,
    int_cols,
    ranges,
    epochs: int,
    batch_size: int,
    seed: int,
    use_cuda: bool,
) -> pd.DataFrame:
    from sdv.single_table import TVAESynthesizer

    md = _make_metadata(train_df)
    set_seed(seed)
    model = TVAESynthesizer(md, epochs=epochs, batch_size=batch_size, verbose=False, cuda=use_cuda)
    model.fit(train_df)
    syn = model.sample(num_rows=len(train_df))
    return postprocess(syn, int_cols, ranges)


def patch_rmsnorm() -> None:
    try:
        import torch
        import torch.nn as nn
        if hasattr(nn, "RMSNorm"):
            return

        class RMSNorm(nn.Module):  # type: ignore
            def __init__(self, normalized_shape, eps: float = 1e-8):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(normalized_shape))
                self.eps = eps

            def forward(self, x):
                var = x.pow(2).mean(dim=-1, keepdim=True)
                x = x * torch.rsqrt(var + self.eps)
                return self.weight * x

        nn.RMSNorm = RMSNorm  # type: ignore[attr-defined]
    except Exception:
        pass


def gen_ddpm(train_df: pd.DataFrame, int_cols, ranges, n_iter: int, batch_size: int, seed: int) -> pd.DataFrame:
    import torch

    # synthcity -> opacus expects `torch.nn.RMSNorm` (torch<2.4 may not have it).
    # Patch must happen before importing synthcity/opacus.
    patch_rmsnorm()
    from synthcity.plugins import Plugins

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plugin = Plugins().get(
        "ddpm",
        n_iter=n_iter,
        batch_size=batch_size,
        random_state=seed,
        device=device,
    )
    plugin.fit(train_df)
    syn = plugin.generate(count=len(train_df)).data
    return postprocess(syn, int_cols, ranges)


def gen_tabpfn_cond(train_df: pd.DataFrame, int_cols, ranges, cfg: Dict, seed: int) -> pd.DataFrame:
    if TabPFNConditionalGenerator is None:
        raise RuntimeError("TabPFNConditionalGenerator not available (expected in src/generator.py)")
    config = cfg.copy()
    config["seed"] = seed
    config["target_column"] = LABEL_COL
    config.setdefault("preserve_label_distribution", True)
    config.setdefault("integer_columns", [LABEL_COL])
    generator = TabPFNConditionalGenerator(config)
    tmp_file = Path("/tmp") / f"tabpfn_tmp_{os.getpid()}.csv"
    train_df.to_csv(tmp_file, index=False)
    try:
        syn_df = generator.generate(str(tmp_file))
    finally:
        tmp_file.unlink(missing_ok=True)
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
    seed: int,
    ctgan_params: Dict,
    tvae_params: Dict,
    ddpm_params: Dict,
    tabpfn_cfg: Dict,
    generators: List[str],
    include_baseline: bool,
) -> Dict:
    raw_df = pd.read_csv(csv_path)
    df, int_cols, float_cols, ranges = clean_dataframe(raw_df)
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    folds = []
    fold_id = 0

    X_all = df[feature_cols].to_numpy()
    y_all = df[LABEL_COL].to_numpy()
    use_cuda = False
    try:
        import torch

        use_cuda = bool(torch.cuda.is_available())
    except Exception:
        use_cuda = False

    for train_idx, test_idx, split_name in iter_splits(X_all, y_all, seed=seed, n_splits=5):
        fold_id += 1
        real_train = df.iloc[train_idx].reset_index(drop=True)
        real_test = df.iloc[test_idx].reset_index(drop=True)

        baseline = eval_models(real_train, real_test, seed) if include_baseline else {}
        fold_entry = {
            "fold": fold_id,
            "split": split_name,
            "baseline": baseline,
            "gens": {},
            "counts": {"real_train": len(real_train), "real_test": len(real_test)},
        }

        syn_cache: Dict[str, pd.DataFrame] = {}
        if "gaussian_copula" in generators:
            print(f"  [fold {fold_id}/5] gaussian_copula...", flush=True)
            syn_cache["gaussian_copula"] = gen_gaussian_copula(real_train, int_cols, ranges)
        if "sdv_ctgan" in generators:
            print(f"  [fold {fold_id}/5] sdv_ctgan (epochs={ctgan_params['epochs']}, cuda={use_cuda})...", flush=True)
            syn_cache["sdv_ctgan"] = gen_ctgan(
                real_train,
                int_cols,
                ranges,
                epochs=ctgan_params["epochs"],
                batch_size=ctgan_params["batch_size"],
                pac=ctgan_params["pac"],
                seed=seed + fold_id,
                use_cuda=use_cuda,
            )
        if "sdv_tvae" in generators:
            print(f"  [fold {fold_id}/5] sdv_tvae (epochs={tvae_params['epochs']}, cuda={use_cuda})...", flush=True)
            syn_cache["sdv_tvae"] = gen_tvae(
                real_train,
                int_cols,
                ranges,
                epochs=tvae_params["epochs"],
                batch_size=tvae_params["batch_size"],
                seed=seed + fold_id,
                use_cuda=use_cuda,
            )
        if "synthcity_ddpm" in generators:
            print(f"  [fold {fold_id}/5] synthcity_ddpm (n_iter={ddpm_params['n_iter']}, cuda={use_cuda})...", flush=True)
            syn_cache["synthcity_ddpm"] = gen_ddpm(
                real_train,
                int_cols,
                ranges,
                n_iter=ddpm_params["n_iter"],
                batch_size=ddpm_params["batch_size"],
                seed=seed + fold_id,
            )
        if "tabpfn" in generators:
            print(f"  [fold {fold_id}/5] tabpfn_conditional...", flush=True)
            syn_cache["tabpfn"] = gen_tabpfn_cond(real_train, int_cols, ranges, tabpfn_cfg, seed + fold_id)

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
            "baseline": aggregate_metrics_mean_std([f["baseline"] for f in folds]) if include_baseline else {},
            "generators": {
                g: {
                    "synthetic": aggregate_metrics_mean_std([f["gens"][g]["synthetic"] for f in folds]),
                }
                for g in gens_in_result
            },
        },
        "params": {
            "ctgan": ctgan_params,
            "tvae": tvae_params,
            "ddpm": ddpm_params,
            "tabpfn": tabpfn_cfg,
            "seed": seed,
        },
    }


def write_markdown(results: List[Dict], out_path: Path) -> None:
    title = "# SOTA Generators Pipeline Results"

    def render_dataset_section(res: Dict) -> List[str]:
        lines: List[str] = []
        lines.append(f"## {res['dataset']}")
        lines.append(f"- path: `{res['path']}`")
        lines.append(f"- data: `{res['data_file']}`")
        lines.append(f"- samples: {res['samples']}")
        params = res["params"]
        lines.append(f"- CTGAN: {params['ctgan']}")
        lines.append(f"- TVAE: {params['tvae']}")
        lines.append(f"- SynthCity DDPM: {params['ddpm']}")
        lines.append(f"- TabPFN conditional: {params['tabpfn']}")
        lines.append(f"- seed: {params['seed']}")
        lines.append("")
        lines.append("| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        if res["summary"].get("baseline"):
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

    for res in results:
        name = str(res["dataset"])
        sections_map[name] = render_dataset_section(res)
        if name not in sections_in_order:
            sections_in_order.append(name)

    out_lines: List[str] = list(header_lines)
    if out_lines and out_lines[-1].strip() != "":
        out_lines.append("")
    for name in sections_in_order:
        out_lines.extend(sections_map[name])
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")

    out_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOTA generators 5-fold TSTR pipeline.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--results-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/sota.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-baseline", action="store_true", help="Also report real-train->real-test baseline.")
    parser.add_argument("--ctgan-epochs", type=int, default=50)
    parser.add_argument("--ctgan-batch-size", type=int, default=64)
    parser.add_argument("--ctgan-pac", type=int, default=1)
    parser.add_argument("--tvae-epochs", type=int, default=50)
    parser.add_argument("--tvae-batch-size", type=int, default=64)
    parser.add_argument("--ddpm-iter", type=int, default=120)
    parser.add_argument("--ddpm-batch-size", type=int, default=128)
    parser.add_argument("--tabpfn-gibbs", type=int, default=3)
    parser.add_argument("--tabpfn-batch-size", type=int, default=256)
    parser.add_argument("--tabpfn-clip-low", type=float, default=0.01)
    parser.add_argument("--tabpfn-clip-high", type=float, default=0.99)
    parser.add_argument("--tabpfn-use-gpu", action="store_true")
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["gaussian_copula", "sdv_ctgan", "sdv_tvae", "synthcity_ddpm", "tabpfn"],
        help="Subset of generators to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    results = []
    for ds_path in expand_dataset_dirs(args.datasets):
        csv_path = find_dataset_csv(ds_path)
        if csv_path is None:
            # Should not happen because expand_dataset_dirs filters this case.
            print(f"[skip] {ds_path} has no data.csv/dataset.csv", file=sys.stderr)
            continue
        print(f"\n[dataset] {ds_path.name} ({csv_path.name})", flush=True)
        res = run_dataset(
            ds_path,
            csv_path=csv_path,
            seed=args.seed,
            ctgan_params={"epochs": args.ctgan_epochs, "batch_size": args.ctgan_batch_size, "pac": args.ctgan_pac},
            tvae_params={"epochs": args.tvae_epochs, "batch_size": args.tvae_batch_size},
            ddpm_params={"n_iter": args.ddpm_iter, "batch_size": args.ddpm_batch_size},
            tabpfn_cfg={
                "sample_count": None,
                "num_gibbs_rounds": args.tabpfn_gibbs,
                "batch_size": args.tabpfn_batch_size,
                "clip_quantile_low": args.tabpfn_clip_low,
                "clip_quantile_high": args.tabpfn_clip_high,
                "use_gpu": args.tabpfn_use_gpu,
            },
            generators=args.generators,
            include_baseline=args.include_baseline,
        )
        results.append(res)
        write_markdown(results, args.results_md)
    if not results:
        print("No datasets processed.", file=sys.stderr)
        sys.exit(1)
    print(f"Done. Results written to {args.results_md}")


if __name__ == "__main__":
    main()
