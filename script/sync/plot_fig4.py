import re
from io import StringIO
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_cv_baseline(cv_path: Path) -> dict:
    df = pd.read_csv(cv_path)
    df = df[df["model"] != "tabpfn"]
    df_acc = df[df["metric_name"] == "accuracy"]
    df_f1 = df[df["metric_name"] == "macro_f1"]
    acc_val = (
        df_acc[df_acc["dataset"] == "arrhythmia"]
        .loc[df_acc[df_acc["dataset"] == "arrhythmia"]["mean"].idxmax(), "mean"]
    )
    f1_val = (
        df_f1[df_f1["dataset"] == "arrhythmia"]
        .loc[df_f1[df_f1["dataset"] == "arrhythmia"]["mean"].idxmax(), "mean"]
    )
    return {"acc": float(acc_val), "macro_f1": float(f1_val)}


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
            df["synth"] = synth
            records.append(df)
    result = pd.concat(records, ignore_index=True)
    num_cols = [c for c in result.columns if c not in ["model", "dataset", "synth"]]
    for col in num_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def plot_arrhythmia(md_path: Path, cv_path: Path, out_path: Path) -> None:
    synth = load_synth(md_path)
    arr = synth[synth["dataset"] == "arrhythmia"].copy()
    arr_best = arr.loc[arr.groupby("synth")["accuracy"].idxmax()].sort_values("synth")
    baseline = load_cv_baseline(cv_path)

    sns.set_theme(context="talk", style="whitegrid", font_scale=1.05)
    plt.figure(figsize=(7, 5))
    palette = sns.color_palette("colorblind")
    plt.plot(
        arr_best["synth"],
        arr_best["accuracy"],
        marker="o",
        linewidth=2.4,
        color=palette[0],
        label="Accuracy (best model per size)",
    )
    plt.plot(
        arr_best["synth"],
        arr_best["f1_macro"],
        marker="s",
        linewidth=2.2,
        color=palette[1],
        label="Macro F1 (best model per size)",
    )
    plt.axhline(baseline["acc"], color=palette[0], linestyle="--", linewidth=1.8, label="Baseline accuracy (real CV)")
    plt.axhline(baseline["macro_f1"], color=palette[1], linestyle="--", linewidth=1.8, label="Baseline macro F1 (real CV)")
    plt.ylim(0.6, 1.02)
    plt.xticks([100, 200, 1000])
    plt.xlabel("Synthetic data size (% of real)")
    plt.ylabel("Score on real test set")
    plt.title("Arrhythmia: Class-Balance Gains from Synthetic Data")
    plt.legend(frameon=False, ncol=1)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    root = Path(__file__).resolve()
    home = root.parents[4]
    md_file = home / "TabPFN/sync_data_proj/datasets/document/metric.md"
    cv_file = home / "finetune_tabpfn_v2/dataset/cv_results_summary.csv"
    out_file = home / "TabPFN/sync_data_proj/plots/fig4_arrhythmia_macro.png"
    plot_arrhythmia(md_file, cv_file, out_file)
