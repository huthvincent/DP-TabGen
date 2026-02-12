import re
from io import StringIO
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


GEN_NAME_MAP = {
    "sync_gaussian_copula.csv": "GC",
    "sync_sdv_ctgan.csv": "CTGAN",
    "sync_sdv_tvae.csv": "TVAE",
    "sync_synthcity_ddpm.csv": "DDPM",
    "sync_tabpfn.csv": "TabPFN",
    "ours_synth100": "Ours",
}


def load_sota(md_path: Path) -> pd.DataFrame:
    """Parse SOTA_arr_metric.md for arrhythmia metrics per generator/model."""
    text = Path(md_path).read_text()
    blocks = re.split(r"\n## ", text)
    rows = []
    for block in blocks:
        block = block.strip()
        if not block or block.startswith("# "):
            continue
        header = block.splitlines()[0].strip()
        gen_name = header.replace("`", "").strip()
        gen_readable = GEN_NAME_MAP.get(gen_name, gen_name)
        table_lines = [ln for ln in block.splitlines() if ln.strip().startswith("|")]
        if len(table_lines) < 3:
            continue
        table_lines = [table_lines[0]] + table_lines[2:]
        df = pd.read_csv(StringIO("\n".join(table_lines)), sep="|")
        df = df.dropna(axis=1, how="all")
        df = df.rename(columns=lambda c: str(c).strip())
        df = df.drop(columns=[c for c in df.columns if c.strip() == ""], errors="ignore")
        if "model" in df.columns:
            df["model"] = df["model"].astype(str).str.strip()
        df["generator"] = gen_readable
        rows.append(df)
    result = pd.concat(rows, ignore_index=True)
    numeric_cols = [c for c in result.columns if c not in ["model", "generator"]]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def load_ours(md_path: Path) -> pd.DataFrame:
    """Parse metric.md for arrhythmia synthetic_100 metrics."""
    text = Path(md_path).read_text()
    # find Arrhythmia block
    blocks = re.split(r"\n# ", text)
    arr_block = next(
        (blk for blk in blocks if blk.strip().startswith("Arrhythmia Synthetic")), None
    )
    if arr_block is None:
        raise ValueError("Arrhythmia block not found in metric.md")
    sections = re.split(r"\n## ", "\n".join(arr_block.splitlines()[1:]))
    target = None
    for sec in sections:
        sec = sec.strip()
        if sec.startswith("synthetic_100"):
            target = sec
            break
    if target is None:
        raise ValueError("synthetic_100 section not found for Arrhythmia")
    table_lines = [ln for ln in target.splitlines() if ln.strip().startswith("|")]
    table_lines = [table_lines[0]] + table_lines[2:]
    df = pd.read_csv(StringIO("\n".join(table_lines)), sep="|")
    df = df.dropna(axis=1, how="all")
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.drop(columns=[c for c in df.columns if c.strip() == ""], errors="ignore")
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.strip()
    df["generator"] = GEN_NAME_MAP["ours_synth100"]
    numeric_cols = [c for c in df.columns if c not in ["model", "generator"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_data(ours_md: Path, sota_md: Path) -> pd.DataFrame:
    ours = load_ours(ours_md)
    sota = load_sota(sota_md)
    df = pd.concat([ours, sota], ignore_index=True)
    df["generator"] = pd.Categorical(
        df["generator"],
        categories=[
            GEN_NAME_MAP["ours_synth100"],
            GEN_NAME_MAP["sync_gaussian_copula.csv"],
            GEN_NAME_MAP["sync_sdv_ctgan.csv"],
            GEN_NAME_MAP["sync_sdv_tvae.csv"],
            GEN_NAME_MAP["sync_synthcity_ddpm.csv"],
            GEN_NAME_MAP["sync_tabpfn.csv"],
        ],
        ordered=True,
    )
    return df


def plot_single_model(df: pd.DataFrame, metric: str, model: str, out_path: Path, log_scale: bool = False) -> None:
    sub = df[df["model"] == model].copy()

    if log_scale:
        # ensure positivity for log scale
        sub[metric] = sub[metric].clip(lower=1e-6)

    order = [
        GEN_NAME_MAP["ours_synth100"],
        GEN_NAME_MAP["sync_gaussian_copula.csv"],
        GEN_NAME_MAP["sync_sdv_ctgan.csv"],
        GEN_NAME_MAP["sync_sdv_tvae.csv"],
        GEN_NAME_MAP["sync_synthcity_ddpm.csv"],
        GEN_NAME_MAP["sync_tabpfn.csv"],
    ]
    sub["generator"] = pd.Categorical(sub["generator"], categories=order, ordered=True)
    sub = sub.sort_values("generator").reset_index(drop=True)
    sub["is_ours"] = sub["generator"] == GEN_NAME_MAP["ours_synth100"]
    sub["generator"] = sub["generator"].astype(str)
    sns.set_theme(context="talk", style="whitegrid")
    plt.figure(figsize=(7.5, 4.5))
    highlight_color = "#1b9e77"  # ours
    other_color = "#b2b2b2"
    colors = [highlight_color if flag else other_color for flag in sub["is_ours"]]

    ax = plt.gca()
    positions = range(len(sub))
    ax.bar(
        positions,
        sub[metric],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(sub["generator"], rotation=0, fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title())
    if log_scale:
        ax.set_yscale("log")

    # Simple legend to clarify highlight
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=highlight_color, edgecolor="black", linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor=other_color, edgecolor="black", linewidth=0.8),
    ]
    labels = ["Ours", "Others"]
    ax.legend(handles, labels, frameon=False, loc="upper right", title="")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    root = Path(__file__).resolve()
    home = root.parents[4]
    ours_md = home / "TabPFN/sync_data_proj/datasets/document/metric.md"
    sota_md = home / "TabPFN/sync_data_proj/datasets/document/SOTA_arr_metric.md"
    df_all = prepare_data(ours_md, sota_md)

    plots_dir = home / "TabPFN/sync_data_proj/plots"
    # logistic
    plot_single_model(
        df_all,
        metric="accuracy",
        model="logistic",
        out_path=plots_dir / "fig2a_sota.png",
    )
    plot_single_model(
        df_all,
        metric="log_loss",
        model="logistic",
        out_path=plots_dir / "fig2b_sota.png",
        log_scale=True,
    )
    # catboost
    plot_single_model(
        df_all,
        metric="accuracy",
        model="catboost",
        out_path=plots_dir / "fig2c_sota.png",
    )
    plot_single_model(
        df_all,
        metric="log_loss",
        model="catboost",
        out_path=plots_dir / "fig2d_sota.png",
        log_scale=True,
    )
