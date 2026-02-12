"""
Combined trade-off plot with MIA TPR@5% overlay.

- Primary y-axis: ROC-AUC vs DP noise scale for all classifiers (from trade_off/result.md).
- Secondary y-axis: TPR@5% vs noise (from MIA.md), dashed line.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ["logistic", "xgboost", "lightgbm", "catboost", "mlp"]


def parse_tradeoff(path: Path) -> Dict[str, Dict[str, float]]:
    """noise -> model -> auc_mean"""
    res: Dict[str, Dict[str, float]] = {}
    current_noise = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("## sgld_noise_scale"):
                current_noise = line.split("=")[1].strip()
                res.setdefault(current_noise, {})
                continue
            if not line.startswith("|") or line.startswith("| ---"):
                continue
            if current_noise is None:
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 7:
                continue
            model = parts[0]
            auc_field = parts[5]
            try:
                auc_mean = float(auc_field.split("±")[0].strip())
            except Exception:
                auc_mean = float("nan")
            res[current_noise][model] = auc_mean
    return res


def parse_mia(path: Path) -> Dict[float, float]:
    """noise (float) -> TPR@5%"""
    res: Dict[float, float] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|") or line.startswith("| ---"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 8 or parts[0] == "noise tag":
                continue
            try:
                tag_val = float(parts[0])
            except Exception:
                continue
            try:
                tpr5 = float(parts[7])
            except Exception:
                tpr5 = float("nan")
            res[tag_val] = tpr5
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trade-off with MIA TPR overlay.")
    parser.add_argument("--tradeoff-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/trade_off/result.md"))
    parser.add_argument("--mia-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/MIA.md"))
    parser.add_argument("--out", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/plots/mia.png"))
    parser.add_argument("--noise-list", nargs="+", type=float, default=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
    args = parser.parse_args()

    trade = parse_tradeoff(args.tradeoff_md)
    mia = parse_mia(args.mia_md)
    noise_vals = sorted(args.noise_list)

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(MODEL_ORDER)))
    fig, ax1 = plt.subplots(figsize=(7, 4))

    for color, model in zip(colors, MODEL_ORDER):
        ys = []
        for nv in noise_vals:
            key = f"{nv:.3f}"
            ys.append(trade.get(key, {}).get(model, np.nan))
        ax1.plot(noise_vals, ys, marker="o", label=model, linewidth=2, color=color)
    ax1.set_xlabel("DP noise scale", fontsize=12, fontweight="bold")
    ax1.set_ylabel("ROC-AUC", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    tpr_vals = [mia.get(nv, np.nan) for nv in noise_vals]
    ax2.plot(noise_vals, tpr_vals, marker="s", linestyle="--", color="#d62728", linewidth=2, label="TPR@5% (MIA)")
    ax2.set_ylabel("TPR@5% (MIA)", fontsize=12, fontweight="bold", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 0.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
