"""
Low-resource synthetic data benchmark for a single dataset.

For each percentage p in {10,20,30,40,50}:
  1) Split the real data into 5 stratified folds (train/test).
  2) From each train fold, take p% as the reference set to fit generators.
  3) Generate a synthetic train set (same size as the reference set).
  4) Train downstream models on the synthetic train set; evaluate on the real test fold.
  5) Aggregate mean ± std across folds.

Generators supported: Gaussian Copula, SDV CTGAN, SDV TVAE, SynthCity DDPM, TabPFN conditional.
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

from fin_sota_pipeline import (  # type: ignore
    LABEL_COL,
    METRICS,
    MODELS,
    aggregate_metrics_mean_std,
    clean_dataframe,
    eval_models,
    expand_dataset_dirs,
    find_dataset_csv,
    gen_ctgan,
    gen_ddpm,
    gen_gaussian_copula,
    gen_tabpfn_cond,
    gen_tvae,
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
    # use train_test_split to preserve class distribution
    ref, _ = train_test_split(train_df, train_size=fraction, stratify=y, random_state=seed)
    return ref.reset_index(drop=True)


def _stabilize_generated(syn_df: pd.DataFrame, real_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Make sure synthetic data is usable by downstream models:
      - Drop rows with NaNs.
      - Ensure at least one sample for every class; if missing, borrow a few from real_train.
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


def aggregate_across_folds(per_fold: List[Dict[str, Dict[str, Dict[str, float]]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Aggregate per-fold metrics -> mean/std per model for a generator."""
    agg: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for model in MODELS:
        rows = []
        for fold_res in per_fold:
            rows.append([fold_res[model][m] for m in METRICS])
        arr = np.array(rows, dtype=float)
        agg[model] = {m: _safe_mean_std(arr[:, i]) for i, m in enumerate(METRICS)}
    return agg


def _safe_mean_std(values: np.ndarray) -> Tuple[float, float]:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return math.nan, math.nan
    if valid.size == 1:
        return float(valid.mean()), 0.0
    return float(valid.mean()), float(valid.std(ddof=1))


def run_low_resource_for_dataset(
    dataset_dir: Path,
    csv_path: Path,
    percents: List[float],
    seed: int,
    ctgan_params: Dict,
    tvae_params: Dict,
    ddpm_params: Dict,
    tabpfn_cfg: Dict,
    generators: List[str],
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
        "params": {
            "ctgan": ctgan_params,
            "tvae": tvae_params,
            "ddpm": ddpm_params,
            "tabpfn": tabpfn_cfg,
            "generators": generators,
        },
    }

    X_all = df[feature_cols].to_numpy()
    y_all = df[LABEL_COL].to_numpy()

    use_cuda = False
    try:
        import torch

        use_cuda = bool(torch.cuda.is_available())
    except Exception:
        use_cuda = False
    if ("tabpfn" in generators) and (not use_cuda) and not tabpfn_cfg.get("use_gpu", False):
        print("[info] CUDA unavailable and tabpfn_use_gpu not set; skipping TabPFN generator.")
        generators = [g for g in generators if g != "tabpfn"]

    # Precompute folds once
    folds = list(iter_splits(X_all, y_all, seed=seed, n_splits=5))

    for pct in percents:
        pct_key = f"{int(pct*100)}%"
        per_fold_results: Dict[str, List[Dict[str, Dict[str, float]]]] = {g: [] for g in generators}
        for fold_id, (train_idx, test_idx, split_name) in enumerate(folds, start=1):
            real_train = df.iloc[train_idx].reset_index(drop=True)
            real_test = df.iloc[test_idx].reset_index(drop=True)
            ref_seed = seed + fold_id + int(pct * 1000)
            ref_train = subset_reference(real_train, pct, ref_seed)

            syn_cache: Dict[str, pd.DataFrame] = {}
            if "gaussian_copula" in generators:
                try:
                    syn = gen_gaussian_copula(ref_train, int_cols, ranges)
                    syn_cache["gaussian_copula"] = _stabilize_generated(syn, real_train, ref_seed)
                except Exception as e:
                    print(f"[skip] gaussian_copula fold {fold_id} pct {pct_key}: {e}")
            if "sdv_ctgan" in generators:
                try:
                    syn = gen_ctgan(
                        ref_train,
                        int_cols,
                        ranges,
                        epochs=ctgan_params["epochs"],
                        batch_size=ctgan_params["batch_size"],
                        pac=ctgan_params["pac"],
                        seed=ref_seed,
                        use_cuda=use_cuda,
                    )
                    syn_cache["sdv_ctgan"] = _stabilize_generated(syn, real_train, ref_seed)
                except Exception as e:
                    print(f"[skip] sdv_ctgan fold {fold_id} pct {pct_key}: {e}")
            if "sdv_tvae" in generators:
                try:
                    syn = gen_tvae(
                        ref_train,
                        int_cols,
                        ranges,
                        epochs=tvae_params["epochs"],
                        batch_size=tvae_params["batch_size"],
                        seed=ref_seed,
                        use_cuda=use_cuda,
                    )
                    syn_cache["sdv_tvae"] = _stabilize_generated(syn, real_train, ref_seed)
                except Exception as e:
                    print(f"[skip] sdv_tvae fold {fold_id} pct {pct_key}: {e}")
            if "synthcity_ddpm" in generators:
                try:
                    syn = gen_ddpm(
                        ref_train,
                        int_cols,
                        ranges,
                        n_iter=ddpm_params["n_iter"],
                        batch_size=ddpm_params["batch_size"],
                        seed=ref_seed,
                    )
                    syn_cache["synthcity_ddpm"] = _stabilize_generated(syn, real_train, ref_seed)
                except Exception as e:
                    print(f"[skip] synthcity_ddpm fold {fold_id} pct {pct_key}: {e}")
            if "tabpfn" in generators:
                try:
                    syn = gen_tabpfn_cond(ref_train, int_cols, ranges, tabpfn_cfg, ref_seed)
                    syn_cache["tabpfn"] = _stabilize_generated(syn, real_train, ref_seed)
                except Exception as e:
                    print(f"[skip] tabpfn fold {fold_id} pct {pct_key}: {e}")

            for gname, gtrain in syn_cache.items():
                metrics = eval_models(gtrain, real_test, seed)
                per_fold_results[gname].append(metrics)

        # aggregate per generator
        agg_for_pct = {}
        for gname, fold_list in per_fold_results.items():
            if not fold_list:
                continue
            agg_for_pct[gname] = aggregate_metrics_mean_std(fold_list)
        results["per_percent"][pct_key] = agg_for_pct

    return results


def write_markdown(res: Dict, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Low-resource synthetic benchmark")
    lines.append("")
    lines.append(f"Dataset: `{res['dataset']}`")
    lines.append(f"Data file: `{res['data_file']}`")
    lines.append(f"Samples: {res['samples']}")
    lines.append(f"Percents: {res['percents']}")
    lines.append(f"Seed: {res['seed']}")
    lines.append(f"Generators: {res['params']['generators']}")
    lines.append("")

    for pct, gen_res in res["per_percent"].items():
        lines.append(f"## Reference fraction: {pct}")
        lines.append("")
        lines.append("| generator | model | accuracy | precision | recall | f1 | roc_auc | log_loss |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for gname, models in gen_res.items():
            for model in MODELS:
                row = models[model]
                lines.append(
                    f"| {gname} | {model} | "
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
    parser = argparse.ArgumentParser(description="Low-resource synthetic benchmark for a single dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy"))
    parser.add_argument("--percents", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/low_resource.md"))
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

    dataset_dirs = expand_dataset_dirs([str(args.dataset_dir)])
    if not dataset_dirs:
        raise SystemExit(f"No dataset found under {args.dataset_dir}")
    ds_dir = dataset_dirs[0]
    csv_path = find_dataset_csv(ds_dir)
    if csv_path is None:
        raise SystemExit(f"{ds_dir} has no data.csv/dataset.csv")

    tabpfn_cfg = {
        "sample_count": None,
        "num_gibbs_rounds": args.tabpfn_gibbs,
        "batch_size": args.tabpfn_batch_size,
        "clip_quantile_low": args.tabpfn_clip_low,
        "clip_quantile_high": args.tabpfn_clip_high,
        "use_gpu": args.tabpfn_use_gpu,
    }

    res = run_low_resource_for_dataset(
        ds_dir,
        csv_path=csv_path,
        percents=args.percents,
        seed=args.seed,
        ctgan_params={"epochs": args.ctgan_epochs, "batch_size": args.ctgan_batch_size, "pac": args.ctgan_pac},
        tvae_params={"epochs": args.tvae_epochs, "batch_size": args.tvae_batch_size},
        ddpm_params={"n_iter": args.ddpm_iter, "batch_size": args.ddpm_batch_size},
        tabpfn_cfg=tabpfn_cfg,
        generators=args.generators,
    )
    write_markdown(res, args.results_md)
    print(f"Saved results to {args.results_md}")


if __name__ == "__main__":
    main()
