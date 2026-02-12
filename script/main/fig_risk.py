"""
Plots for Decision-Critical Tail Fidelity (Risk Scoring) on polish_companies_bankruptcy.

Outputs (default to plots/risk/):
1) score_dist.png: Real vs synthetic score distribution (hist/kde) with PSI annotation.
2) decile_event_rate.png: Decile-wise event rate curves (real vs synthetic).
3) tail_topk.png: Top 1/5/10% event rate & capture rate (grouped bars).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from risk import train_logistic, psi, decile_calibration, tail_metrics  # type: ignore
from fin_tabpfgen_pipeline import LABEL_COL, clean_dataframe, set_seed  # type: ignore


def score_distributions(real_scores: np.ndarray, synth_scores: np.ndarray, psi_val: float, out_path: Path) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 40
    ax.hist(real_scores, bins=bins, alpha=0.5, density=True, label="Real", color="#1f77b4")
    ax.hist(synth_scores, bins=bins, alpha=0.5, density=True, label="Synthetic", color="#ff7f0e")
    sns.kdeplot(real_scores, ax=ax, color="#1f77b4", linewidth=2)
    sns.kdeplot(synth_scores, ax=ax, color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Risk score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title("Score distribution: Real vs Synthetic", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 0.3)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(
        0.02,
        0.95,
        f"PSI={psi_val:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=11,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def decile_plot(deciles: List[dict], out_path: Path) -> None:
    bins = [d["bin"] for d in deciles]
    real = [d["real_event_rate"] for d in deciles]
    synth = [d["synth_event_rate"] for d in deciles]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, real, marker="o", label="Real", color="#1f77b4", linewidth=2)
    ax.plot(x, synth, marker="o", label="Synthetic", color="#ff7f0e", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45)
    ax.set_ylabel("Event rate", fontsize=12, fontweight="bold")
    ax.set_title("Decile calibration (event rate)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def tail_bar(tails: List[dict], out_path: Path) -> None:
    segments = [t["segment"] for t in tails]
    real_event = [t["real_event_rate"] for t in tails]
    synth_event = [t["synth_event_rate"] for t in tails]
    real_cap = [t["real_capture_rate"] for t in tails]
    synth_cap = [t["synth_capture_rate"] for t in tails]

    x = np.arange(len(segments))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].bar(x - width / 2, real_event, width, label="Real", color="#1f77b4")
    axes[0].bar(x + width / 2, synth_event, width, label="Synthetic", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(segments)
    axes[0].set_ylabel("Event rate", fontsize=12, fontweight="bold")
    axes[0].set_title("Tail event rate", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x - width / 2, real_cap, width, label="Real", color="#1f77b4")
    axes[1].bar(x + width / 2, synth_cap, width, label="Synthetic", color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(segments)
    axes[1].set_ylabel("Capture rate", fontsize=12, fontweight="bold")
    axes[1].set_title("Tail capture rate", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot risk scoring fidelity figures.")
    parser.add_argument("--real-path", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy/data.csv"))
    parser.add_argument("--synth-path", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/plots/risk"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    real_raw = pd.read_csv(args.real_path)
    real_df, _, _, _ = clean_dataframe(real_raw)
    synth_raw = pd.read_csv(args.synth_path)
    synth_df, _, _, _ = clean_dataframe(synth_raw)

    clf = train_logistic(real_df, args.seed)
    real_scores = clf.predict_proba(real_df.drop(columns=[LABEL_COL]).to_numpy())[:, 1]
    synth_scores = clf.predict_proba(synth_df.drop(columns=[LABEL_COL]).to_numpy())[:, 1]
    real_labels = real_df[LABEL_COL].to_numpy()
    synth_labels = synth_df[LABEL_COL].to_numpy()

    psi_val = psi(real_scores, synth_scores, bins=10)
    deciles = decile_calibration(real_scores, real_labels, synth_scores, synth_labels)
    tails = tail_metrics(real_scores, real_labels, synth_scores, synth_labels)

    out_dir = args.out_dir
    score_distributions(real_scores, synth_scores, psi_val, out_dir / "score_dist.png")
    decile_plot(deciles, out_dir / "decile_event_rate.png")
    tail_bar(tails, out_dir / "tail_topk.png")


if __name__ == "__main__":
    main()
