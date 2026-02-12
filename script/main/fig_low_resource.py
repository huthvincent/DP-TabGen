"""
Plot learning curves (AUC vs. reference fraction) for low-resource benchmarks.

Sources:
- SOTA generators: datasets/Results/low_resource.md
- TabPFGen: datasets/Results/tabpfgen_low_resource.md

For each classifier, plot AUC across percents (0.1–0.5) with different colors per generator.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_md(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns:
      percents -> model -> generator -> auc_mean
    """
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    current_pct = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("## Reference fraction:"):
                pct_label = line.split(":")[1].strip()
                current_pct = pct_label
                data.setdefault(current_pct, {})
                continue
            if not line.startswith("|") or line.startswith("| ---"):
                continue
            if current_pct is None:
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 8:
                continue
            gen, model, *_metrics = parts
            # metrics order: accuracy, precision, recall, f1, roc_auc, log_loss
            auc_field = parts[6]
            try:
                auc_mean = float(auc_field.split("±")[0].strip())
            except Exception:
                auc_mean = float("nan")
            data[current_pct].setdefault(model, {})
            data[current_pct][model][gen] = auc_mean
    return data


def merge_sources(sota: Dict, tabpfgen: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    percents = set(sota.keys()) | set(tabpfgen.keys())
    merged: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pct in percents:
        merged[pct] = {}
        for src in (sota.get(pct, {}), tabpfgen.get(pct, {})):
            for model, gens in src.items():
                merged[pct].setdefault(model, {})
                merged[pct][model].update(gens)
    return merged


def plot_model(model: str, percents: List[float], merged: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path) -> None:
    gens = sorted({g for pct in merged.values() if model in pct for g in pct[model].keys()})
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(gens)))
    label_map = {"tabpfgen": "our method"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for color, gen in zip(colors, gens):
        ys = []
        xs = []
        for p in percents:
            pct_label = f"{int(p*100)}%"
            val = merged.get(pct_label, {}).get(model, {}).get(gen, np.nan)
            ys.append(val)
            xs.append(p)
        display = label_map.get(gen, gen)
        ax.plot(xs, ys, marker="o", label=display, color=color, linewidth=2)

    ax.set_xlabel("Reference fraction", fontsize=12, fontweight="bold")
    ax.set_ylabel("ROC-AUC", fontsize=12, fontweight="bold")
    ax.set_title(f"AUC vs. fraction — {model}", fontsize=13, fontweight="bold")
    ax.set_xticks(percents)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(title="Generator", fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{model}_auc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot low-resource AUC learning curves.")
    parser.add_argument("--sota-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/low_resource.md"))
    parser.add_argument("--tabpfgen-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/tabpfgen_low_resource.md"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/plots/low_resource"))
    parser.add_argument("--percents", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    sota = parse_md(args.sota_md)
    tabpfgen = parse_md(args.tabpfgen_md)
    merged = merge_sources(sota, tabpfgen)

    # Models present in merged data
    models = sorted({m for pct in merged.values() for m in pct.keys()})
    for model in models:
        plot_model(model, args.percents, merged, args.out_dir)


if __name__ == "__main__":
    main()
