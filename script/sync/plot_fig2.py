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
    df = df[["dataset", "mean"]]
    best = df.loc[df.groupby("dataset")["mean"].idxmax()].copy()
    return best


def load_synth_metrics(md_path: Path) -> pd.DataFrame:
    text = Path(md_path).read_text()
    blocks = re.split(r"\n# ", text)
    records = []
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
            df["dataset"] = dataset
            df["dataset_label"] = dataset_label
            df["synth"] = synth
            records.append(df)
    result = pd.concat(records, ignore_index=True)
    num_cols = [c for c in result.columns if c not in ["model", "dataset", "dataset_label"]]
    for col in num_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def plot_scaling(md_path: Path, cv_path: Path, out_path: Path) -> None:
    synth_df = load_synth_metrics(md_path)
    base_df = load_cv_baseline(cv_path)
    best_synth = synth_df.loc[synth_df.groupby(["dataset", "synth"])["accuracy"].idxmax()].copy()
    best_synth["dataset_label"] = best_synth["dataset_label"].str.title()
    base_df["dataset_label"] = base_df["dataset"].str.title()
    order = ["Arrhythmia", "Heart", "Pima", "Wdbc", "Ckd", "Uti"]
    n_cols = 3
    n_rows = 2
    sns.set_theme(context="talk", style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8), sharey=True)
    palette = sns.color_palette("Set2")
    for idx, dataset in enumerate(order):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = best_synth[best_synth["dataset_label"] == dataset].sort_values("synth")
        ax.plot(
            sub["synth"],
            sub["accuracy"],
            marker="o",
            color=palette[1],
            linewidth=2.3,
            label="Best model on synthetic",
        )
        base_val = float(
            base_df.loc[base_df["dataset_label"] == dataset, "mean"].values[0]
        )
        ax.axhline(base_val, color=palette[0], linestyle="--", linewidth=2, label="Real-data CV best")
        ax.set_title(dataset)
        ax.set_xlabel("Synthetic data size (% of real)")
        ax.set_ylabel("Test accuracy on real set" if idx % n_cols == 0 else "")
        ax.set_ylim(0.6, 1.05)
        ax.set_xticks([100, 200, 1000])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=12)
    fig.suptitle("Scaling Synthetic Data → Test Accuracy on Real Set", y=1.02)
    plt.tight_layout(rect=(0, 0.06, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    root = Path(__file__).resolve()
    home = root.parents[4]
    md_file = home / "TabPFN/sync_data_proj/datasets/document/metric.md"
    cv_file = home / "finetune_tabpfn_v2/dataset/cv_results_summary.csv"
    out_file = home / "TabPFN/sync_data_proj/plots/fig2_synth_scaling.png"
    plot_scaling(md_file, cv_file, out_file)
