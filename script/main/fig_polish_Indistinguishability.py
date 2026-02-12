"""
Plots for polish_companies_bankruptcy indistinguishability:
1) Marginal distributions (our method real vs synth) for a subset of features.
2) Correlation heatmaps: real, our method synth, |diff|.
3) C2ST bar plot (real-vs-synth AUC/Acc) for our method and TVAE.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent

import sys

if str(ROOT / "script") not in sys.path:
    sys.path.append(str(ROOT / "script"))

from fin_tabpfgen_pipeline import LABEL_COL, clean_dataframe  # type: ignore


def c2st_auc_acc(real_df: pd.DataFrame, synth_df: pd.DataFrame, feature_cols: List[str], seed: int) -> Tuple[float, float]:
    X = pd.concat([real_df[feature_cols], synth_df[feature_cols]], axis=0).to_numpy()
    y = np.array([0] * len(real_df) + [1] * len(synth_df))
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
    return float(auc), float(acc)


def pick_features(feature_cols: List[str], k: int = 8) -> List[str]:
    return feature_cols[:k] if len(feature_cols) <= k else feature_cols[:k]


def plot_marginals(real_df: pd.DataFrame, synth_df: pd.DataFrame, feature_cols: List[str], out_path: Path) -> None:
    sns.set_style("whitegrid")
    n = len(feature_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows), constrained_layout=True)
    axes = axes.flatten()
    for ax, col in zip(axes, feature_cols):
        r = real_df[col]
        s = synth_df[col]
        combined = pd.concat([r, s], axis=0).dropna()
        lo = combined.quantile(0.001)
        hi = combined.quantile(0.999)
        span = hi - lo
        if span <= 0:
            span = 1.0
        pad = 0.05 * span
        x_min, x_max = lo - pad, hi + pad

        if pd.api.types.is_integer_dtype(r) or pd.api.types.is_bool_dtype(r):
            bins = np.arange(np.floor(x_min), np.ceil(x_max) + 1)
            ax.hist(r, bins=bins, alpha=0.5, label="real", density=True, color="#1f77b4")
            ax.hist(s, bins=bins, alpha=0.5, label="our method", density=True, color="#ff7f0e")
        else:
            sns.kdeplot(r, ax=ax, label="real", color="#1f77b4", fill=True, alpha=0.3, clip=(x_min, x_max))
            sns.kdeplot(s, ax=ax, label="our method", color="#ff7f0e", fill=True, alpha=0.3, clip=(x_min, x_max))
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("")
        ax.set_title(col)
        ax.legend(fontsize=8)
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Marginal distributions (real vs our method)", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_corr(real_df: pd.DataFrame, synth_df: pd.DataFrame, feature_cols: List[str], out_path: Path) -> None:
    real_corr = real_df[feature_cols].corr()
    synth_corr = synth_df[feature_cols].corr()
    diff = (real_corr - synth_corr).abs()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
    sns.heatmap(synth_corr, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
    hm = sns.heatmap(diff, ax=axes[2], cmap="magma", cbar=True, cbar_kws={"ticks": [0, 0.15, 0.30]})

    # Remove titles; simplify axes labels
    for ax in axes:
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Attr", fontsize=24, fontweight="bold")
        ax.set_ylabel("Attr", fontsize=24, fontweight="bold")

    # Style colorbar ticks
    cb = hm.collections[0].colorbar
    cb.ax.tick_params(labelsize=24)
    cb.set_ticklabels(["0", "0.15", "0.30"])
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontweight("bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_c2st_bars(results: List[Tuple[str, float, float]], out_path: Path) -> None:
    labels = [r[0] for r in results]
    aucs = [r[1] for r in results]
    accs = [r[2] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, aucs, width, label="ROC-AUC")
    ax.bar(x + width / 2, accs, width, label="Accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("C2ST (real vs synth)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot indistinguishability figures for polish_companies_bankruptcy.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy"))
    parser.add_argument("--tabpfgen-csv", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability.csv"))
    parser.add_argument("--tvae-csv", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability_tvae.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/plots/indistinguishability"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-count", type=int, default=8)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.dataset_dir / "data.csv")
    real_df, int_cols, float_cols, _ = clean_dataframe(raw_df)
    feature_cols = [c for c in real_df.columns if c != LABEL_COL]
    sel_feats = pick_features(feature_cols, k=args.feature_count)

    tab_df_raw = pd.read_csv(args.tabpfgen_csv)
    tab_df, tab_int, tab_float, _ = clean_dataframe(tab_df_raw)

    plot_marginals(real_df, tab_df, sel_feats, args.out_dir / "marginals_tabpfgen.png")
    plot_corr(real_df, tab_df, sel_feats, args.out_dir / "corr_tabpfgen.png")

    results = []
    auc_tab, acc_tab = c2st_auc_acc(real_df, tab_df, sel_feats, seed=args.seed)
    results.append(("our method", auc_tab, acc_tab))

    if args.tvae_csv.exists():
        tvae_raw = pd.read_csv(args.tvae_csv)
        tvae_df, _, _, _ = clean_dataframe(tvae_raw)
        auc_tvae, acc_tvae = c2st_auc_acc(real_df, tvae_df, sel_feats, seed=args.seed)
        results.append(("TVAE", auc_tvae, acc_tvae))

    plot_c2st_bars(results, args.out_dir / "c2st_bar.png")


if __name__ == "__main__":
    main()
