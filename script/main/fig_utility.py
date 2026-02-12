"""
Plot utility bar charts for synthetic data performance.

Reads SOTA and TabPFGen markdown result files, parses metrics, and plots
ROC-AUC / PR-AUC (if present) per dataset. Supports grouping by model or
averaging across models with error bars.

Example:
    python fig_utility.py \
        --datasets australian_credit_approval bank_marketing \
        --metrics roc_auc \
        --group-mode average \
        --output-dir /home/zhu11/TabPFN/sync_data_proj/plots/utility
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Force default matplotlib style to avoid inherited grayscale themes
plt.style.use("default")
# Default paths
ROOT = Path("/home/zhu11/TabPFN/sync_data_proj")
DEFAULT_SOTA_MD = ROOT / "datasets/Results/sota_part.md"
DEFAULT_TABPFGEN_MD = ROOT / "datasets/Results/tabpfgen_part.md"
DEFAULT_OUTPUT_DIR = ROOT / "plots/utility"

DEFAULT_DATASETS = [
    "australian_credit_approval",
    "bank_marketing",
    "german_credit",
    "polish_companies_bankruptcy",
]

DEFAULT_MODELS = ["logistic", "xgboost", "lightgbm", "catboost", "mlp"]
DEFAULT_METRICS = ["roc_auc"]

# Pretty names for generators / datasets
GEN_NAME_MAP = {
    "gaussian_copula": "Gaussian Copula",
    "sdv_ctgan": "CTGAN",
    "sdv_tvae": "TVAE",
    "synthcity_ddpm": "DDPM",
    "tabpfgen": "Ours",
    "tabpfn_conditional": "TabPFN",
}

DATASET_NAME_MAP = {
    "australian_credit_approval": "Australian Credit Approval",
    "bank_marketing": "Bank Marketing",
    "german_credit": "German Credit",
    "polish_companies_bankruptcy": "Polish Companies Bankruptcy",
}

# Color-blind friendly palette (high contrast)
DEFAULT_PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d", "#e6ab02"]


def parse_metric(token: str) -> Tuple[float, float]:
    """Parse a token like '0.8130 ± 0.0558' or '0.8130'."""
    token = token.strip()
    if "±" in token:
        mean_str, std_str = token.split("±")
        return float(mean_str), float(std_str)
    try:
        return float(token), 0.0
    except Exception:
        return math.nan, math.nan


def parse_markdown(md_path: Path, source: str) -> pd.DataFrame:
    """Parse a markdown result file into a long-form DataFrame."""
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    lines = md_path.read_text(encoding="utf-8").splitlines()
    records: List[Dict] = []
    dataset: str | None = None
    header: List[str] = []

    for line in lines:
        if line.startswith("## "):
            dataset = line[3:].strip()
            continue
        if line.startswith("| generator"):
            header = [h.strip() for h in line.split("|")[1:-1]]
            continue
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        if dataset is None or not header:
            continue

        cols = [c.strip() for c in line.split("|")[1:-1]]
        row = dict(zip(header, cols))
        generator = row.get("generator", "")
        scheme = row.get("scheme", "")
        model = row.get("model", "")

        metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "pr_auc"]
        for m in metric_names:
            if m in row:
                mean, std = parse_metric(row[m])
                records.append(
                    {
                        "dataset": dataset,
                        "source": source,
                        "generator": generator,
                        "scheme": scheme,
                        "model": model,
                        "metric": m,
                        "mean": mean,
                        "std": std,
                    }
                )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"No records parsed from {md_path}")
    return df


def aggregate(df: pd.DataFrame, group_mode: str, models: List[str]) -> pd.DataFrame:
    """Aggregate metrics.

    - group_mode == "average": average across models, std aggregated via rms of stds.
    - group_mode == "by_model": keep per-model rows.
    """
    if "std" not in df.columns:
        df = df.copy()
        df["std"] = 0.0

    if group_mode == "by_model":
        return df[df["model"].isin(models)].copy()

    def agg_std(sub) -> float:
        # `sub` can be a Series (named agg on column "std") or a DataFrame.
        if isinstance(sub, pd.Series):
            vals = sub.dropna().to_numpy()
        else:
            vals = sub["std"].dropna().to_numpy()
        if len(vals) == 0:
            return math.nan
        return float(math.sqrt((vals**2).mean()))

    grouped = (
        df[df["model"].isin(models)]
        .groupby(["dataset", "source", "generator", "scheme", "metric"], as_index=False)
        .agg(mean=("mean", "mean"), std=("std", agg_std))
    )
    grouped["model"] = "avg"
    return grouped


def plot_dataset(
    df: pd.DataFrame,
    dataset: str,
    metrics: List[str],
    group_mode: str,
    output_dir: Path,
    palette: List[str],
    ymax: float | None = None,
) -> Path:
    if not palette:
        palette = DEFAULT_PALETTE
    # Larger fonts for readability
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )

    df_ds = df[(df["dataset"] == dataset) & (df["scheme"] == "synthetic") & (df["generator"] != "baseline")]
    if df_ds.empty:
        raise ValueError(f"No data for dataset={dataset}")

    gens = sorted(df_ds["generator"].unique().tolist())
    pretty_gens = [GEN_NAME_MAP.get(g, g) for g in gens]
    models = sorted(df_ds["model"].unique().tolist())

    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")

    for idx, metric in enumerate(metrics):
        ax = axes[idx, 0]
        sub = df_ds[df_ds["metric"] == metric]
        if sub.empty:
            ax.set_visible(False)
            continue

        ax.set_facecolor("#f2f2f2")

        if group_mode == "by_model":
            x_labels = models
            n_gen = len(gens)
            x = np.arange(len(x_labels))
            width = 0.8 / max(n_gen, 1)
            for g_idx, gen in enumerate(gens):
                data = sub[sub["generator"] == gen]
                heights = [data[(data["model"] == m)]["mean"].mean() for m in x_labels]
                errors = [data[(data["model"] == m)]["std"].mean() for m in x_labels]
                ax.bar(
                    x + g_idx * width,
                    heights,
                    width=width,
                    color=palette[g_idx % len(palette)],
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=1.0,
                    yerr=errors,
                    capsize=4,
                )
                for xi, h, e in zip(x + g_idx * width, heights, errors):
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
                        rotation=0,
                    )
            ax.set_xticks(x + width * (n_gen - 1) / 2)
            ax.set_xticklabels(x_labels, rotation=0)
        else:
            x_labels = pretty_gens
            x = np.arange(len(x_labels))
            heights = [sub[sub["generator"] == g]["mean"].mean() for g in gens]
            errors = [sub[sub["generator"] == g]["std"].mean() for g in gens]
            ax.bar(
                x,
                heights,
                width=0.65,
                color=[palette[i % len(palette)] for i in range(len(x_labels))],
                edgecolor="black",
                linewidth=1.0,
                alpha=1.0,
                yerr=errors,
                capsize=4,
            )
            for xi, h, e in zip(x, heights, errors):
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
                    rotation=0,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=0)

        title_ds = DATASET_NAME_MAP.get(dataset, dataset)
        ax.set_title(title_ds, fontsize=16, loc="center")
        ax.set_ylabel("ROAUC" if metric == "roc_auc" else metric.upper())
        # Fix y-range to make bars visible and consistent
        ax.set_ylim(bottom=0, top=(ymax if ymax is not None else 1.05))
        ax.grid(True, axis="y", alpha=0.25)
        # Legend intentionally removed

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}.png"
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot utility bar charts from markdown results.")
    parser.add_argument("--sota-md", type=Path, default=DEFAULT_SOTA_MD, help="Path to SOTA results markdown.")
    parser.add_argument("--tabpfgen-md", type=Path, default=DEFAULT_TABPFGEN_MD, help="Path to TabPFGen results markdown.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, help="Datasets to plot (names matching markdown sections).")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to plot (e.g., roc_auc, pr_auc).")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to include.")
    parser.add_argument("--group-mode", choices=["average", "by_model"], default="average", help="Average across models or group by model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save plots.")
    parser.add_argument("--palette", nargs="+", default=DEFAULT_PALETTE)
    parser.add_argument("--ymax", type=float, default=None, help="Optional y-axis upper limit.")
    args = parser.parse_args()

    dfs = []
    dfs.append(parse_markdown(args.sota_md, source="sota"))
    dfs.append(parse_markdown(args.tabpfgen_md, source="tabpfgen"))
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all[df_all["metric"].isin(args.metrics)]
    df_all = aggregate(df_all, group_mode=args.group_mode, models=args.models)

    for ds in args.datasets:
        try:
            out = plot_dataset(
                df_all,
                dataset=ds,
                metrics=args.metrics,
                group_mode=args.group_mode,
                output_dir=args.output_dir,
                palette=args.palette,
                ymax=args.ymax,
            )
            print(f"[ok] {ds} -> {out}")
        except Exception as exc:
            print(f"[skip] {ds}: {exc}")


if __name__ == "__main__":
    main()
