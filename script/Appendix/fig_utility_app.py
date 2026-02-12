"""
Appendix plotting helper.

Appendix plotting helper.

Draws per-dataset, per-metric, per-model bar plots:
  /home/zhu11/TabPFN/sync_data_proj/plots/appendix/utility/{dataset}_{model}_{metric}.png
Each plot: x-axis = generators, y = metric mean, error bar = std (if available).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/zhu11/TabPFN/sync_data_proj")
MAIN_DIR = ROOT / "script/main"
sys.path.insert(0, str(MAIN_DIR))

from fig_utility import (  # type: ignore
    DEFAULT_MODELS,
    DEFAULT_SOTA_MD,
    DEFAULT_TABPFGEN_MD,
    DEFAULT_PALETTE,
    GEN_NAME_MAP,
    DATASET_NAME_MAP,
    parse_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix plots (grouped by model).")
    parser.add_argument("--sota-md", type=Path, default=DEFAULT_SOTA_MD)
    parser.add_argument("--tabpfgen-md", type=Path, default=DEFAULT_TABPFGEN_MD)
    parser.add_argument("--datasets", nargs="*", default=None, help="If omitted, use all datasets found in markdowns.")
    parser.add_argument("--metrics", nargs="+", default=["roc_auc"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots/appendix/utility")
    parser.add_argument("--palette", nargs="+", default=DEFAULT_PALETTE)
    parser.add_argument("--ymax", type=float, default=None)
    args = parser.parse_args()

    # Ensure a consistent, high-contrast style
    if not args.palette:
        args.palette = DEFAULT_PALETTE
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    df = pd.concat(
        [
            parse_markdown(args.sota_md, source="sota"),
            parse_markdown(args.tabpfgen_md, source="tabpfgen"),
        ],
        ignore_index=True,
    )
    df = df[(df["scheme"] == "synthetic") & (df["generator"] != "baseline")]
    df = df[df["metric"].isin(args.metrics)]
    df = df[df["model"].isin(args.models)]

    # Auto-discover datasets if none provided
    datasets = args.datasets or sorted(df["dataset"].unique().tolist())

    def plot_one(dataset: str, metric: str, model: str, palette: List[str]) -> Path:
        sub = df[(df["dataset"] == dataset) & (df["metric"] == metric) & (df["model"] == model)]
        if sub.empty:
            raise ValueError("no data")
        gens = sorted(sub["generator"].unique().tolist())
        pretty_gens = [GEN_NAME_MAP.get(g, g) for g in gens]
        means = [sub[sub["generator"] == g]["mean"].mean() for g in gens]
        stds = [sub[sub["generator"] == g]["std"].mean() for g in gens]

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("white")
        x = np.arange(len(gens))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            width=0.65,
            color=[palette[i % len(palette)] for i in range(len(gens))],
            edgecolor="black",
            linewidth=1.0,
            alpha=1.0,
        )
        for xi, h, e in zip(x, means, stds):
            if np.isnan(h):
                continue
            ax.text(
                xi,
                h + (e if not np.isnan(e) else 0) + 0.01,
                f"{h:.3f}±{e:.3f}" if not np.isnan(e) else f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(pretty_gens, rotation=0)
        ax.set_ylabel("ROAUC" if metric == "roc_auc" else metric.upper())
        title_ds = DATASET_NAME_MAP.get(dataset, dataset)
        ax.set_title(f"{title_ds} – {model} – {metric}", loc="center")
        if args.ymax is not None:
            ax.set_ylim(top=args.ymax)
        ax.set_ylim(bottom=0, top=(args.ymax if args.ymax is not None else 1.05))
        ax.grid(True, axis="y", alpha=0.25)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{dataset}_{model}_{metric}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_path

    for ds in datasets:
        for metric in args.metrics:
            for model in args.models:
                try:
                    out = plot_one(ds, metric, model, args.palette)
                    print(f"[ok] {ds} {model} {metric} -> {out}")
                except Exception as exc:
                    print(f"[skip] {ds} {model} {metric}: {exc}")


if __name__ == "__main__":
    main()
