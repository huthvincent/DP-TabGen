import re
from io import StringIO
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_synth(md_path: Path) -> pd.DataFrame:
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
    num_cols = [c for c in result.columns if c not in ["model", "dataset", "dataset_label", "synth"]]
    for col in num_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def plot_model_robustness(md_path: Path, out_path: Path) -> None:
    synth = load_synth(md_path)
    synth_1000 = synth[synth["synth"] == 1000]
    sns.set_theme(context="talk", style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        data=synth_1000,
        x="model",
        y="accuracy",
        whis=1.5,
        palette="Set2",
        hue="model",
        dodge=False,
        legend=False,
    )
    sns.stripplot(
        data=synth_1000,
        x="model",
        y="accuracy",
        color="black",
        size=5,
        alpha=0.65,
        jitter=0.15,
    )
    ax.set_ylabel("Accuracy on real test set (synthetic 1000%)")
    ax.set_xlabel("Model")
    ax.set_ylim(0.75, 1.02)
    ax.set_title("Model Robustness Across Datasets (Synthetic 1000%)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    root = Path(__file__).resolve()
    home = root.parents[4]
    md_file = home / "TabPFN/sync_data_proj/datasets/document/metric.md"
    out_file = home / "TabPFN/sync_data_proj/plots/fig5_model_robustness.png"
    plot_model_robustness(md_file, out_file)
