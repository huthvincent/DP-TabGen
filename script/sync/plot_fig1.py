import re
from io import StringIO
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_cv_baseline(cv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cv_path)
    df = df[df["model"] != "tabpfn"]
    df = df[df["metric_name"] == "accuracy"]
    df = df[["dataset", "model", "mean"]]
    df["dataset_label"] = df["dataset"].str.title()
    return df


def load_synth_accuracy(md_path: Path) -> pd.DataFrame:
    text = Path(md_path).read_text()
    blocks = re.split(r"\n# ", text)
    rows = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        header = lines[0].strip()
        if "Synthetic" not in header:
            continue
        dataset_label = header.split("Synthetic")[0].strip(" #")
        dataset = dataset_label.lower()
        sections = re.split(r"\n## ", "\n".join(lines[1:]))
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            m = re.search(r"synthetic_(\d+)", sec.splitlines()[0])
            if not m:
                continue
            synth = int(m.group(1))
            table_lines = [ln for ln in sec.splitlines() if ln.strip().startswith("|")]
            if len(table_lines) < 3:
                continue
            table_lines = [table_lines[0]] + table_lines[2:]
            df = pd.read_csv(StringIO("\n".join(table_lines)), sep="|")
            df = df.dropna(axis=1, how="all")
            df = df.rename(columns=lambda c: str(c).strip())
            df = df.drop(columns=[c for c in df.columns if c.strip() == ""], errors="ignore")
            df = df[["model", "accuracy"]]
            df["dataset"] = dataset
            df["dataset_label"] = dataset_label
            df["synth"] = synth
            rows.append(df)
    result = pd.concat(rows, ignore_index=True)
    result["accuracy"] = pd.to_numeric(result["accuracy"], errors="coerce")
    result["dataset_label"] = result["dataset_label"].str.title()
    return result


def plot_heatmap(pivot: pd.DataFrame, title: str, cbar_label: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.set_theme(context="talk", style="white")
    cmap = sns.color_palette("viridis", as_cmap=True)
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        linewidths=0.6,
        cmap=cmap,
        vmin=0.7,
        vmax=1.0,
        cbar_kws={"label": cbar_label},
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Dataset")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_baseline_heatmap(cv_path: Path, out_path: Path) -> None:
    df = load_cv_baseline(cv_path)
    pivot = df.pivot(index="dataset_label", columns="model", values="mean")
    plot_heatmap(pivot, "Fig1a: Real-Data CV Accuracy", "CV accuracy (real data)", out_path)


def plot_synth_heatmap(md_path: Path, synth_level: int, out_path: Path) -> None:
    df = load_synth_accuracy(md_path)
    df_level = df[df["synth"] == synth_level]
    pivot = df_level.pivot(index="dataset_label", columns="model", values="accuracy")
    title = f"Fig1{'b' if synth_level==100 else 'c' if synth_level==200 else 'd'}: Train on Synthetic {synth_level}% → Test on Real"
    plot_heatmap(pivot, title, "Accuracy (synthetic train → real test)", out_path)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[4]
    cv_file = base_dir / "finetune_tabpfn_v2/dataset/cv_results_summary.csv"
    md_file = base_dir / "TabPFN/sync_data_proj/datasets/document/metric.md"
    plots_dir = base_dir / "TabPFN/sync_data_proj/plots"

    plot_baseline_heatmap(cv_file, plots_dir / "fig1a_baseline_heatmap.png")
    plot_synth_heatmap(md_file, 100, plots_dir / "fig1b_synth100_heatmap.png")
    plot_synth_heatmap(md_file, 200, plots_dir / "fig1c_synth200_heatmap.png")
    plot_synth_heatmap(md_file, 1000, plots_dir / "fig1d_synth1000_heatmap.png")
