"""
Trade-off curves for TabPFGen noise scale on polish_companies_bankruptcy.

Reads datasets/Results/trade_off/result.md and plots, for each classifier:
  x = sgld_noise_scale
  y = ROC-AUC (mean)
Outputs to plots/trade-off/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Models order to keep consistent styling
MODEL_ORDER = ["logistic", "xgboost", "lightgbm", "catboost", "mlp"]
COLOR = "#1f77b4"


def parse_md(path: Path) -> Dict[str, Dict[str, float]]:
    """
    Returns: noise -> model -> roc_auc_mean
    """
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


def plot_model(model: str, noise_vals: List[float], data: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    ys = []
    for nv in noise_vals:
        key = f"{nv:.3f}"
        ys.append(data.get(key, {}).get(model, np.nan))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(noise_vals, ys, marker="o", color=COLOR, linewidth=2)
    ax.set_xlabel("DP noise scale", fontsize=12, fontweight="bold")
    ax.set_ylabel("ROC-AUC", fontsize=12, fontweight="bold")
    ax.set_title(f"Noise vs AUC — {model}", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{model}_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_models(noise_vals: List[float], data: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(MODEL_ORDER)))
    fig, ax = plt.subplots(figsize=(7, 4))
    for color, model in zip(colors, MODEL_ORDER):
        ys = []
        for nv in noise_vals:
            key = f"{nv:.3f}"
            ys.append(data.get(key, {}).get(model, np.nan))
        ax.plot(noise_vals, ys, marker="o", label=model, linewidth=2, color=color)
    ax.set_xlabel("DP noise scale", fontsize=12, fontweight="bold")
    ax.set_ylabel("ROC-AUC", fontsize=12, fontweight="bold")
    ax.set_title("Noise vs AUC — all models", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "all_models_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TabPFGen noise trade-off curves.")
    parser.add_argument("--result-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/trade_off/result.md"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/plots/trade-off"))
    parser.add_argument("--noise-list", nargs="+", type=float, default=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
    args = parser.parse_args()

    data = parse_md(args.result_md)
    noise_vals = sorted(args.noise_list)

    for model in MODEL_ORDER:
        plot_model(model, noise_vals, data, args.out_dir)
    plot_all_models(noise_vals, data, args.out_dir)


if __name__ == "__main__":
    main()
