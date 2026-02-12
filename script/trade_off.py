"""
TabPFGen noise-scale trade-off on polish_companies_bankruptcy.

For each sgld_noise_scale in a predefined list:
  - Train TabPFGen on each fold's real-train, generate synthetic of same size.
  - Evaluate downstream classifiers on real test fold (TSTR).
  - Save a full synthetic dataset (generated from the entire real set) for inspection.
  - Aggregate mean/std metrics across 5 folds.

Outputs:
  - Synthetic CSVs under datasets/Results/trade_off/*.csv
  - Markdown summary at datasets/Results/trade_off/result.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT / "script") not in sys.path:
    sys.path.append(str(ROOT / "script"))

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


def run_one_noise(
    df: pd.DataFrame,
    int_cols: List[str],
    ranges: Dict,
    noise_scale: float,
    seed: int,
) -> Dict:
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    X_all = df[feature_cols].to_numpy()
    y_all = df[LABEL_COL].to_numpy()
    folds = list(iter_splits(X_all, y_all, seed=seed, n_splits=5))

    per_fold = []
    for fold_id, (train_idx, test_idx, _) in enumerate(folds, start=1):
        real_train = df.iloc[train_idx].reset_index(drop=True)
        real_test = df.iloc[test_idx].reset_index(drop=True)
        ref_seed = seed + fold_id
        syn = generate_tabpfgen(
            real_train,
            int_cols,
            ranges,
            sgld_steps=600,
            step_size=0.01,
            noise_scale=noise_scale,
            jitter=0.01,
            factor=1.0,
            seed=ref_seed,
            energy_subsample=2048,
        )
        metrics = eval_models(syn, real_test, seed)
        per_fold.append(metrics)

    agg = aggregate_metrics_mean_std(per_fold)
    return agg


def write_markdown(out_path: Path, results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    lines: List[str] = []
    lines.append("# TabPFGen noise-scale trade-off (polish_companies_bankruptcy)")
    lines.append("")
    lines.append("Noise scales tested: 0.005, 0.01, 0.015, 0.02, 0.025, 0.03 (sgld_noise_scale).")
    lines.append("")
    for noise, models in results.items():
        lines.append(f"## sgld_noise_scale = {noise}")
        lines.append("")
        lines.append("| model | accuracy | precision | recall | f1 | roc_auc | log_loss |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for model in MODELS:
            m = models[model]
            lines.append(
                f"| {model} | "
                f"{m['accuracy'][0]:.4f} ± {m['accuracy'][1]:.4f} | "
                f"{m['precision'][0]:.4f} ± {m['precision'][1]:.4f} | "
                f"{m['recall'][0]:.4f} ± {m['recall'][1]:.4f} | "
                f"{m['f1'][0]:.4f} ± {m['f1'][1]:.4f} | "
                f"{m['roc_auc'][0]:.4f} ± {m['roc_auc'][1]:.4f} | "
                f"{m['log_loss'][0]:.4f} ± {m['log_loss'][1]:.4f} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TabPFGen noise-scale trade-off on polish_companies_bankruptcy.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/trade_off"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_dirs = expand_dataset_dirs([str(args.dataset_dir)])
    if not dataset_dirs:
        raise SystemExit(f"No dataset found under {args.dataset_dir}")
    ds_dir = dataset_dirs[0]
    csv_path = find_dataset_csv(ds_dir)
    if csv_path is None:
        raise SystemExit(f"{ds_dir} has no data.csv/dataset.csv")

    raw_df = pd.read_csv(csv_path)
    df, int_cols, float_cols, ranges = clean_dataframe(raw_df)

    noise_list = [
        ("5e-3", 0.005),
        ("1e-2", 0.01),
        ("15e-3", 0.015),
        ("2e-2", 0.02),
        ("25e-3", 0.025),
        ("3e-2", 0.03),
    ]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for tag, noise in noise_list:
        # Save a full synthetic dataset for reference (fit on all real)
        syn_all = generate_tabpfgen(
            df,
            int_cols,
            ranges,
            sgld_steps=600,
            step_size=0.01,
            noise_scale=noise,
            jitter=0.01,
            factor=1.0,
            seed=args.seed,
            energy_subsample=2048,
        )
        syn_path = args.out_dir / f"{tag}.csv"
        syn_all.to_csv(syn_path, index=False)

        agg = run_one_noise(df, int_cols, ranges, noise_scale=noise, seed=args.seed)
        results[f"{noise:.3f}"] = agg

    write_markdown(args.out_dir / "result.md", results)
    print(f"Saved results to {args.out_dir/'result.md'}")


if __name__ == "__main__":
    main()
