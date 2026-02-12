# path: sync_data_proj/test_pipeline.py
"""端到端冒烟测试：生成混合类型数据并对比分布。"""
from __future__ import annotations

import subprocess
import sys
from itertools import zip_longest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (放在 Agg 设置之后)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "sync_data_proj/input_test.csv"
OUTPUT_PATH = PROJECT_ROOT / "sync_data_proj/output_test.csv"
PLOT_PATH = PROJECT_ROOT / "sync_data_proj/distribution_compare.png"
CONFIG_PATH = PROJECT_ROOT / "config/config.yaml"


def build_synthetic_input(rows: int = 800) -> pd.DataFrame:
    """构造包含连续 + 类别 + 目标列的混合数据。"""

    rng = np.random.default_rng(2024)
    target = rng.choice(["高", "中", "低"], size=rows, p=[0.4, 0.35, 0.25])
    mean_map = {"高": 2.5, "中": 0.5, "低": -1.5}
    stdev_map = {"高": 0.7, "中": 1.0, "低": 0.9}
    gaussian = np.array([rng.normal(mean_map[label], stdev_map[label]) for label in target])
    uniform = rng.uniform(-2, 2, size=rows)
    categorical = rng.choice(["红", "绿", "蓝", "黄"], size=rows, p=[0.3, 0.3, 0.2, 0.2])
    bool_flag = rng.random(rows) < 0.5
    correlated = gaussian * 0.8 + rng.normal(0, 0.3, size=rows)
    df = pd.DataFrame(
        {
            "gaussian_feature": gaussian,
            "uniform_feature": uniform,
            "color_code": categorical,
            "is_active": bool_flag,
            "correlated_feature": correlated,
            "label": target,
        }
    )
    return df


def run_cli(input_path: Path, output_path: Path) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src/cli_generate.py"),
        "--config",
        str(CONFIG_PATH),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--n",
        "500",
        "--seed",
        "123",
        "--cpu-only",
        "--target-column",
        "label",
        "--preserve-label-distribution",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def plot_comparison(original: pd.DataFrame, synthetic: pd.DataFrame, output_path: Path) -> None:
    cols = original.columns.tolist()
    ncols = 2
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    for idx, col in enumerate(cols):
        ax = axes[idx]
        orig_series = original[col]
        synth_series = synthetic[col]
        is_categorical = (
            orig_series.dtype == "object"
            or orig_series.dtype.name == "category"
            or orig_series.nunique() <= 10
            or orig_series.dtype == bool
        )
        if is_categorical:
            categories = sorted(set(orig_series.astype(str)) | set(synth_series.astype(str)))
            orig_counts = orig_series.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
            synth_counts = (
                synth_series.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
            )
            xs = np.arange(len(categories))
            ax.bar(xs - 0.2, orig_counts.values, width=0.4, label="原始", alpha=0.8)
            ax.bar(xs + 0.2, synth_counts.values, width=0.4, label="合成", alpha=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels(categories, rotation=30, ha="right")
        else:
            ax.hist(orig_series, bins=20, alpha=0.5, label="原始", density=True)
            ax.hist(synth_series, bins=20, alpha=0.5, label="合成", density=True)
        ax.set_title(col)
        ax.legend()

    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(original: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    print(f"原始样本数: {len(original)}, 合成样本数: {len(synthetic)}")
    numeric_cols = original.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:3]:  # 逐列打印前 3 个数值特征对比
        orig_mean = original[col].mean()
        synth_mean = synthetic[col].mean()
        orig_std = original[col].std()
        synth_std = synthetic[col].std()
        print(f"列[{col}] 原始(mean={orig_mean:.3f}, std={orig_std:.3f}) | 合成(mean={synth_mean:.3f}, std={synth_std:.3f})")

    cat_cols = [c for c in original.columns if c not in numeric_cols]
    for col in cat_cols[:2]:
        orig_top = original[col].value_counts().head(3)
        synth_top = synthetic[col].value_counts().head(3)
        pairs = zip_longest(orig_top.items(), synth_top.items(), fillvalue=("无", 0))
        print(f"列[{col}] 前 3 类别分布：")
        for (o_val, o_cnt), (s_val, s_cnt) in pairs:
            print(f"  原始 {o_val}:{o_cnt} | 合成 {s_val}:{s_cnt}")


def main() -> None:
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_input = build_synthetic_input()
    df_input.to_csv(INPUT_PATH, index=False)
    run_cli(INPUT_PATH, OUTPUT_PATH)
    df_output = pd.read_csv(OUTPUT_PATH)
    plot_comparison(df_input, df_output, PLOT_PATH)
    print_summary(df_input, df_output)
    print(f"分布对比图已保存：{PLOT_PATH}")


if __name__ == "__main__":
    main()
