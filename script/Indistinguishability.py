"""
TabPFGen indistinguishability analysis for a single dataset.

Steps:
1) Load and clean the real dataset (polish_companies_bankruptcy by default).
2) Fit TabPFGen on 100% of the real data; generate a synthetic dataset of equal size.
3) Save the synthetic dataset to CSV.
4) Compute similarity / privacy proxies between real and synthetic:
   - MMD (RBF kernel) on feature vectors.
   - Marginal distances: Wasserstein (numeric) and TVD (int/categorical).
   - Correlation matrix difference (Frobenius norm).
   - C2ST (real-vs-synth classifier) accuracy and ROC-AUC.
5) Write a markdown report with metric definitions and values.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Limit threads for stability
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
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from fin_tabpfgen_pipeline import (  # type: ignore
    LABEL_COL,
    clean_dataframe,
    expand_dataset_dirs,
    find_dataset_csv,
    generate_tabpfgen,
    set_seed,
)


def _subsample_pair(X: np.ndarray, Y: np.ndarray, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_samples and len(Y) <= max_samples:
        return X, Y
    rng = np.random.default_rng(seed)
    n = min(len(X), max_samples)
    m = min(len(Y), max_samples)
    idx_x = rng.choice(len(X), size=n, replace=False)
    idx_y = rng.choice(len(Y), size=m, replace=False)
    return X[idx_x], Y[idx_y]


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None, max_samples: int = 5000, seed: int = 42) -> float:
    """Squared MMD with RBF kernel. Subsamples to max_samples per set to keep memory bounded."""
    X, Y = _subsample_pair(X, Y, max_samples=max_samples, seed=seed)
    XY = np.vstack([X, Y])
    if sigma is None:
        # median heuristic
        dists = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=2)
        median_sq = np.median(dists)
        if median_sq <= 0:
            median_sq = 1.0
        sigma = math.sqrt(0.5 * median_sq)
    gamma = 1.0 / (2 * sigma * sigma)

    def kmat(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * d2)

    Kxx = kmat(X, X)
    Kyy = kmat(Y, Y)
    Kxy = kmat(X, Y)
    n = len(X)
    m = len(Y)
    # unbiased estimators
    mmd = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean()
    return float(mmd)


def marginal_distances(real: pd.DataFrame, synth: pd.DataFrame, int_cols: List[str], float_cols: List[str]) -> Dict:
    marg = {"numeric_wasserstein": {}, "int_tvd": {}}
    for col in float_cols:
        marg["numeric_wasserstein"][col] = float(wasserstein_distance(real[col], synth[col]))
    for col in int_cols:
        r_counts = real[col].value_counts(normalize=True)
        s_counts = synth[col].value_counts(normalize=True)
        all_vals = r_counts.index.union(s_counts.index)
        tvd = 0.5 * float((r_counts.reindex(all_vals, fill_value=0) - s_counts.reindex(all_vals, fill_value=0)).abs().sum())
        marg["int_tvd"][col] = tvd
    return marg


def corr_diff(real: pd.DataFrame, synth: pd.DataFrame, feature_cols: List[str]) -> float:
    real_corr = real[feature_cols].corr().to_numpy()
    synth_corr = synth[feature_cols].corr().to_numpy()
    diff = real_corr - synth_corr
    return float(np.linalg.norm(diff, ord="fro"))


def c2st(real: pd.DataFrame, synth: pd.DataFrame, feature_cols: List[str], seed: int) -> Dict[str, float]:
    X = pd.concat([real[feature_cols], synth[feature_cols]], axis=0).to_numpy()
    y = np.array([0] * len(real) + [1] * len(synth))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X_train, y[train_idx])
    prob = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)
    acc = accuracy_score(y[test_idx], pred)
    try:
        auc = roc_auc_score(y[test_idx], prob)
    except Exception:
        auc = float("nan")
    return {"accuracy": float(acc), "roc_auc": float(auc)}


def write_markdown(out_path: Path, metrics: Dict, params: Dict) -> None:
    lines: List[str] = []
    lines.append("# Indistinguishability analysis (TabPFGen)")
    lines.append("")
    lines.append("## What the metrics mean")
    lines.append("- MMD: kernel-based distance between real and synthetic feature distributions (lower is better; 0 means identical).")
    lines.append("- Marginals: Wasserstein distance for numeric features; TVD (total variation distance) for integer/categorical features (lower is better).")
    lines.append("- Correlation gap: Frobenius norm of correlation matrix difference (lower means similar dependency structure).")
    lines.append("- C2ST: accuracy/AUC of a classifier distinguishing real vs synthetic (closer to 0.5 for both indicates harder to distinguish).")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Real data: `{metrics['real_path']}` (samples={metrics['samples']})")
    lines.append(f"- Synthetic data: `{metrics['synth_path']}` (generated with TabPFGen)")
    lines.append(f"- TabPFGen params: {params}")
    lines.append("")
    lines.append("## Results")
    lines.append(f"- MMD (RBF): `{metrics['mmd']:.6f}`")
    lines.append(f"- Correlation gap (Frobenius): `{metrics['corr_gap']:.6f}`")
    lines.append(f"- C2ST accuracy: `{metrics['c2st']['accuracy']:.4f}`, ROC-AUC: `{metrics['c2st']['roc_auc']:.4f}`")
    lines.append("")
    lines.append("### Marginal distances")
    lines.append("#### Numeric (Wasserstein)")
    if metrics["marginals"]["numeric_wasserstein"]:
        for col, val in metrics["marginals"]["numeric_wasserstein"].items():
            lines.append(f"- `{col}`: {val:.6f}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("#### Integer/Categorical (TVD)")
    if metrics["marginals"]["int_tvd"]:
        for col, val in metrics["marginals"]["int_tvd"].items():
            lines.append(f"- `{col}`: {val:.6f}")
    else:
        lines.append("- (none)")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TabPFGen synthetic data and run indistinguishability analysis.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/Indistinguishability.md"))
    parser.add_argument("--n-sgld-steps", type=int, default=600)
    parser.add_argument("--sgld-step-size", type=float, default=0.01)
    parser.add_argument("--sgld-noise-scale", type=float, default=0.005)
    parser.add_argument("--jitter", type=float, default=0.01)
    parser.add_argument("--synthetic-factor", type=float, default=1.0)
    parser.add_argument("--energy-subsample", type=int, default=2048)
    args = parser.parse_args()

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

    raw_df = pd.read_csv(csv_path)
    real_df, int_cols, float_cols, ranges = clean_dataframe(raw_df)
    feature_cols = [c for c in real_df.columns if c != LABEL_COL]

    syn_df = generate_tabpfgen(
        real_df,
        int_cols,
        ranges,
        sgld_steps=tabpfgen_params["n_sgld_steps"],
        step_size=tabpfgen_params["sgld_step_size"],
        noise_scale=tabpfgen_params["sgld_noise_scale"],
        jitter=tabpfgen_params["jitter"],
        factor=tabpfgen_params["synthetic_factor"],
        seed=args.seed,
        energy_subsample=tabpfgen_params["energy_subsample"],
    )
    syn_df = syn_df.reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    syn_df.to_csv(args.out_csv, index=False)

    # Metrics
    X_real = real_df[feature_cols].to_numpy()
    X_syn = syn_df[feature_cols].to_numpy()
    mmd_val = compute_mmd_rbf(X_real, X_syn, max_samples=5000, seed=args.seed)
    marg = marginal_distances(real_df, syn_df, int_cols, float_cols)
    corr_gap = corr_diff(real_df, syn_df, feature_cols)
    c2st_res = c2st(real_df, syn_df, feature_cols, seed=args.seed)

    metrics = {
        "real_path": str(csv_path),
        "synth_path": str(args.out_csv),
        "samples": len(real_df),
        "mmd": mmd_val,
        "marginals": marg,
        "corr_gap": corr_gap,
        "c2st": c2st_res,
    }
    write_markdown(args.out_md, metrics, tabpfgen_params)
    print(f"Saved synthetic CSV to {args.out_csv}")
    print(f"Saved report to {args.out_md}")


if __name__ == "__main__":
    main()
