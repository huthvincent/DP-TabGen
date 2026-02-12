"""
Low-resource benchmark using TabPFGen on a single dataset.

For each reference fraction p in percents:
  1) Split real data into 5 stratified folds.
  2) Take p% of the train fold as reference data.
  3) Fit TabPFGen on the reference; generate synthetic train (same size).
  4) Train downstream models on synthetic train; evaluate on real test.
  5) Aggregate mean ± std across folds.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Limit threads to avoid OpenBLAS crashes
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
from sklearn.model_selection import StratifiedKFold, train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from fin_tabpfgen_pipeline import (  # type: ignore
    LABEL_COL,
    METRICS,
    MODELS,
    aggregate_metrics_mean_std,
    clean_dataframe,
    eval_models,
    expand_dataset_dirs,
    find_dataset_csv,
    generate_tabpfgen,
    iter_splits,
    set_seed,
)


def subset_reference(train_df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Take a stratified subset of the training fold as the reference set."""
    if fraction >= 1.0:
        return train_df.copy()
    y = train_df[LABEL_COL]
    if y.nunique() < 2:
        return train_df.copy()
    ref, _ = train_test_split(train_df, train_size=fraction, stratify=y, random_state=seed)
    return ref.reset_index(drop=True)


def _stabilize_generated(syn_df: pd.DataFrame, real_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Ensure synthetic data is usable:
      - Drop rows with NaNs.
      - If any class missing, pull a few samples from real_train to cover it.
    """
    syn_df = syn_df.dropna().reset_index(drop=True)
    if syn_df.empty:
        return real_train.sample(n=min(len(real_train), 16), random_state=seed).reset_index(drop=True)

    syn_labels = set(syn_df[LABEL_COL].unique())
    real_labels = list(pd.unique(real_train[LABEL_COL]))
    missing = [lbl for lbl in real_labels if lbl not in syn_labels]
    if not missing:
        return syn_df

    rng = np.random.default_rng(seed)
    extras = []
    for lbl in missing:
        cand = real_train[real_train[LABEL_COL] == lbl]
        if cand.empty:
            continue
        take = min(len(cand), max(1, len(syn_df) // 20))
        extras.append(cand.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))

    if extras:
        syn_df = pd.concat([syn_df] + extras, ignore_index=True)
    return syn_df


def run_tabpfgen_for_dataset(
    dataset_dir: Path,
    csv_path: Path,
    percents: List[float],
    seed: int,
    tabpfgen_params: Dict,
) -> Dict:
    raw_df = pd.read_csv(csv_path)
    df, int_cols, float_cols, ranges = clean_dataframe(raw_df)
    feature_cols = [c for c in df.columns if c != LABEL_COL]

    results = {
        "dataset": dataset_dir.name,
        "path": str(dataset_dir),
        "data_file": str(csv_path),
        "samples": len(df),
        "percents": percents,
        "seed": seed,
        "per_percent": {},
        "params": tabpfgen_params,
    }

    X_all = df[feature_cols].to_numpy()
    y_all = df[LABEL_COL].to_numpy()

    folds = list(iter_splits(X_all, y_all, seed=seed, n_splits=5))

    for pct in percents:
        pct_key = f"{int(pct*100)}%"
        per_fold_results: List[Dict[str, Dict[str, float]]] = []
        for fold_id, (train_idx, test_idx, _) in enumerate(folds, start=1):
            real_train = df.iloc[train_idx].reset_index(drop=True)
            real_test = df.iloc[test_idx].reset_index(drop=True)
            ref_seed = seed + fold_id + int(pct * 1000)
            ref_train = subset_reference(real_train, pct, ref_seed)

            try:
                syn = generate_tabpfgen(
                    ref_train,
                    int_cols,
                    ranges,
                    sgld_steps=tabpfgen_params["n_sgld_steps"],
                    step_size=tabpfgen_params["sgld_step_size"],
                    noise_scale=tabpfgen_params["sgld_noise_scale"],
                    jitter=tabpfgen_params["jitter"],
                    factor=tabpfgen_params["synthetic_factor"],
                    seed=ref_seed,
                    energy_subsample=tabpfgen_params["energy_subsample"],
                )
                syn = _stabilize_generated(syn, real_train, ref_seed)
                metrics = eval_models(syn, real_test, seed)
                per_fold_results.append(metrics)
            except Exception as e:
                print(f"[skip] tabpfgen fold {fold_id} pct {pct_key}: {e}")
                continue

        if per_fold_results:
            results["per_percent"][pct_key] = aggregate_metrics_mean_std(per_fold_results)

    return results


def write_markdown(res: Dict, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# TabPFGen Low-resource benchmark")
    lines.append("")
    lines.append(f"Dataset: `{res['dataset']}`")
    lines.append(f"Data file: `{res['data_file']}`")
    lines.append(f"Samples: {res['samples']}")
    lines.append(f"Percents: {res['percents']}")
    lines.append(f"Seed: {res['seed']}")
    lines.append(f"TabPFGen params: {res['params']}")
    lines.append("")

    for pct, models in res["per_percent"].items():
        lines.append(f"## Reference fraction: {pct}")
        lines.append("")
        lines.append("| generator | model | accuracy | precision | recall | f1 | roc_auc | log_loss |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for model in MODELS:
            row = models[model]
            lines.append(
                f"| tabpfgen | {model} | "
                f"{row['accuracy'][0]:.4f} ± {row['accuracy'][1]:.4f} | "
                f"{row['precision'][0]:.4f} ± {row['precision'][1]:.4f} | "
                f"{row['recall'][0]:.4f} ± {row['recall'][1]:.4f} | "
                f"{row['f1'][0]:.4f} ± {row['f1'][1]:.4f} | "
                f"{row['roc_auc'][0]:.4f} ± {row['roc_auc'][1]:.4f} | "
                f"{row['log_loss'][0]:.4f} ± {row['log_loss'][1]:.4f} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-resource benchmark using TabPFGen.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy"))
    parser.add_argument("--percents", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/tabpfgen_low_resource.md"))
    parser.add_argument("--n-sgld-steps", type=int, default=600)
    parser.add_argument("--sgld-step-size", type=float, default=0.01)
    parser.add_argument("--sgld-noise-scale", type=float, default=0.005)
    parser.add_argument("--jitter", type=float, default=0.01)
    parser.add_argument("--synthetic-factor", type=float, default=1.0, help="Relative size of synthetic set vs reference size.")
    parser.add_argument("--energy-subsample", type=int, default=2048, help="Optional subsample for SGLD energy; 0 disables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_dirs = expand_dataset_dirs([str(args.dataset_dir)])
    if not dataset_dirs:
        raise SystemExit(f"No dataset found under {args.dataset_dir}")
    ds_dir = dataset_dirs[0]
    csv_path = find_dataset_csv(ds_dir)
    if csv_path is None:
        raise SystemExit(f"{ds_dir} has no data.csv/dataset.csv")

    tabpfgen_params = {
        "n_sgld_steps": args.n_sgld_steps,
        "sgld_step_size": args.sgld_step_size,
        "sgld_noise_scale": args.sgld_noise_scale,
        "jitter": args.jitter,
        "synthetic_factor": args.synthetic_factor,
        "energy_subsample": None if args.energy_subsample <= 0 else args.energy_subsample,
    }

    res = run_tabpfgen_for_dataset(
        ds_dir,
        csv_path=csv_path,
        percents=args.percents,
        seed=args.seed,
        tabpfgen_params=tabpfgen_params,
    )
    write_markdown(res, args.results_md)
    print(f"Saved results to {args.results_md}")


if __name__ == "__main__":
    main()
