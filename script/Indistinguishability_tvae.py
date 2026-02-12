"""
TVAE indistinguishability analysis for polish_companies_bankruptcy.

Generates a full-size synthetic dataset with TVAE, then compares to real data:
- MMD (RBF, subsampled) for global distribution similarity.
- Marginal distances (Wasserstein for numeric; TVD for integer/categorical).
- Correlation matrix gap (Frobenius norm).
- C2ST (real vs synth classifier) accuracy/AUC as a disclosure-risk proxy.
Outputs synthetic CSV and a markdown report including side-by-side comparison
with existing TabPFGen results if available.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Limit threads
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

from fin_sota_pipeline import (  # type: ignore
    LABEL_COL,
    clean_dataframe,
    expand_dataset_dirs,
    find_dataset_csv,
    gen_tvae,
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
    X, Y = _subsample_pair(X, Y, max_samples=max_samples, seed=seed)
    XY = np.vstack([X, Y])
    if sigma is None:
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


def stabilize_syn(syn_df: pd.DataFrame, real_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    syn_df = syn_df.dropna().reset_index(drop=True)
    if syn_df.empty:
        return real_df.sample(n=min(len(real_df), 16), random_state=seed).reset_index(drop=True)
    syn_labels = set(syn_df[LABEL_COL].unique())
    real_labels = list(pd.unique(real_df[LABEL_COL]))
    missing = [lbl for lbl in real_labels if lbl not in syn_labels]
    rng = np.random.default_rng(seed)
    extras = []
    for lbl in missing:
        cand = real_df[real_df[LABEL_COL] == lbl]
        if cand.empty:
            continue
        take = min(len(cand), max(1, len(syn_df) // 20))
        extras.append(cand.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    if extras:
        syn_df = pd.concat([syn_df] + extras, ignore_index=True)
    return syn_df


def compute_all_metrics(real_df: pd.DataFrame, syn_df: pd.DataFrame, int_cols: List[str], float_cols: List[str], seed: int, real_path: str, synth_path: str) -> Dict:
    feature_cols = [c for c in real_df.columns if c != LABEL_COL]
    X_real = real_df[feature_cols].to_numpy()
    X_syn = syn_df[feature_cols].to_numpy()
    return {
        "real_path": real_path,
        "synth_path": synth_path,
        "samples": len(real_df),
        "mmd": compute_mmd_rbf(X_real, X_syn, max_samples=5000, seed=seed),
        "marginals": marginal_distances(real_df, syn_df, int_cols, float_cols),
        "corr_gap": corr_diff(real_df, syn_df, feature_cols),
        "c2st": c2st(real_df, syn_df, feature_cols, seed=seed),
    }


def write_markdown(out_path: Path, metrics: Dict[str, Dict], params: Dict, tabpfgen_metrics: Dict | None) -> None:
    lines: List[str] = []
    lines.append("# Indistinguishability analysis (TVAE vs TabPFGen)")
    lines.append("")
    lines.append("## What the metrics mean")
    lines.append("- MMD: kernel-based distance between real and synthetic feature distributions (lower is better; 0 means identical).")
    lines.append("- Marginals: Wasserstein distance for numeric features; TVD (total variation distance) for integer/categorical features (lower is better).")
    lines.append("- Correlation gap: Frobenius norm of correlation matrix difference (lower means similar dependency structure).")
    lines.append("- C2ST: accuracy/AUC of a classifier distinguishing real vs synthetic (closer to 0.5 for both indicates harder to distinguish).")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Real data: `{metrics['tvae']['real_path']}` (samples={metrics['tvae']['samples']})")
    lines.append(f"- TVAE synthetic: `{metrics['tvae']['synth_path']}`")
    lines.append(f"- TVAE params: {params}")
    if tabpfgen_metrics:
        lines.append(f"- TabPFGen synthetic: `{tabpfgen_metrics['synth_path']}` (re-used for comparison)")
    lines.append("")

    lines.append("## Summary")
    lines.append("| generator | MMD | corr_gap | C2ST_acc | C2ST_auc |")
    lines.append("| --- | --- | --- | --- | --- |")
    def row(name, m):
        lines.append(
            f"| {name} | {m['mmd']:.6f} | {m['corr_gap']:.6f} | {m['c2st']['accuracy']:.4f} | {m['c2st']['roc_auc']:.4f} |"
        )
    row("TVAE", metrics["tvae"])
    if tabpfgen_metrics:
        row("TabPFGen", tabpfgen_metrics)
    lines.append("")

    def dump_details(name: str, m: Dict):
        lines.append(f"## Details: {name}")
        lines.append(f"- MMD (RBF): `{m['mmd']:.6f}`")
        lines.append(f"- Correlation gap (Frobenius): `{m['corr_gap']:.6f}`")
        lines.append(f"- C2ST accuracy: `{m['c2st']['accuracy']:.4f}`, ROC-AUC: `{m['c2st']['roc_auc']:.4f}`")
        lines.append("")
        lines.append("### Marginal distances")
        lines.append("#### Numeric (Wasserstein)")
        if m["marginals"]["numeric_wasserstein"]:
            for col, val in m["marginals"]["numeric_wasserstein"].items():
                lines.append(f"- `{col}`: {val:.6f}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("#### Integer/Categorical (TVD)")
        if m["marginals"]["int_tvd"]:
            for col, val in m["marginals"]["int_tvd"].items():
                lines.append(f"- `{col}`: {val:.6f}")
        else:
            lines.append("- (none)")
        lines.append("")

    dump_details("TVAE", metrics["tvae"])
    if tabpfgen_metrics:
        dump_details("TabPFGen", tabpfgen_metrics)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TVAE indistinguishability analysis vs TabPFGen comparison.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability_tvae.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/Indistinguishability.md"))
    parser.add_argument("--tabpfgen-csv", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability.csv"))
    parser.add_argument("--tvae-epochs", type=int, default=50)
    parser.add_argument("--tvae-batch-size", type=int, default=256)
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
    real_df, int_cols, float_cols, ranges = clean_dataframe(raw_df)
    feature_cols = [c for c in real_df.columns if c != LABEL_COL]

    use_cuda = False
    try:
        import torch
        use_cuda = bool(torch.cuda.is_available())
    except Exception:
        use_cuda = False

    syn_df = gen_tvae(
        real_df,
        int_cols,
        ranges,
        epochs=args.tvae_epochs,
        batch_size=args.tvae_batch_size,
        seed=args.seed,
        use_cuda=use_cuda,
    )
    syn_df = stabilize_syn(syn_df, real_df, args.seed).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    syn_df.to_csv(args.out_csv, index=False)

    tvae_metrics = compute_all_metrics(real_df, syn_df, int_cols, float_cols, args.seed, str(csv_path), str(args.out_csv))

    tabpfgen_metrics = None
    if args.tabpfgen_csv.exists():
        tab_df_raw = pd.read_csv(args.tabpfgen_csv)
        tab_df, tab_int, tab_float, _ = clean_dataframe(tab_df_raw)
        tabpfgen_metrics = compute_all_metrics(real_df, tab_df, int_cols, float_cols, args.seed, str(csv_path), str(args.tabpfgen_csv))

    metrics = {"tvae": tvae_metrics}
    write_markdown(args.out_md, metrics, {"tvae_epochs": args.tvae_epochs, "tvae_batch_size": args.tvae_batch_size, "use_cuda": use_cuda}, tabpfgen_metrics)
    print(f"Saved TVAE synthetic CSV to {args.out_csv}")
    print(f"Wrote report to {args.out_md}")


if __name__ == "__main__":
    main()
