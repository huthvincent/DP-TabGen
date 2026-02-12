"""
Decision-Critical Tail Fidelity for Risk Scoring (polish_companies_bankruptcy).

Steps:
1) 读取真实数据，清洗，训练 LogisticRegression 风险模型（标准化 + LR）。
2) 用同一模型对真实数据与合成数据（TabPFGen）出分 s=f(x)。
3) 计算并输出：
   - 分数分布偏移：PSI、KS（真实 vs 合成分数）。
   - 分段校准：按照真实分数的十分位分箱，比较每箱事件率（标签=1 比例）。
   - 尾部指标：以真实分数 99%/95%/90% 分位为阈值，报告真实/合成在 Top1/5/10% 段的事件率及抓取率（capture rate）。
4) 写入中文说明与数值到 datasets/Results/Risk.md。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT / "script") not in sys.path:
    sys.path.append(str(ROOT / "script"))

from fin_tabpfgen_pipeline import LABEL_COL, clean_dataframe, set_seed  # type: ignore


def train_logistic(real_df: pd.DataFrame, seed: int) -> Pipeline:
    X = real_df.drop(columns=[LABEL_COL]).to_numpy()
    y = real_df[LABEL_COL].to_numpy()
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=800, solver="lbfgs")),
        ]
    )
    clf.fit(X, y)
    return clf


def psi(real_scores: np.ndarray, synth_scores: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    quantiles = np.linspace(0, 100, bins + 1)
    cuts = np.percentile(real_scores, quantiles)
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    real_hist, _ = np.histogram(real_scores, bins=cuts)
    synth_hist, _ = np.histogram(synth_scores, bins=cuts)
    real_pct = real_hist / max(real_hist.sum(), eps)
    synth_pct = synth_hist / max(synth_hist.sum(), eps)
    psi_val = np.sum((real_pct - synth_pct) * np.log((real_pct + eps) / (synth_pct + eps)))
    return float(psi_val)


def decile_calibration(scores: np.ndarray, labels: np.ndarray, synth_scores: np.ndarray, synth_labels: np.ndarray) -> List[Dict]:
    """Bin by real-score deciles; report event rate per bin for real & synth."""
    cuts = np.percentile(scores, np.linspace(0, 100, 11))
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    res = []
    for i in range(10):
        lo, hi = cuts[i], cuts[i + 1]
        real_mask = (scores > lo) & (scores <= hi)
        synth_mask = (synth_scores > lo) & (synth_scores <= hi)
        real_rate = labels[real_mask].mean() if real_mask.any() else math.nan
        synth_rate = synth_labels[synth_mask].mean() if synth_mask.any() else math.nan
        res.append(
            {
                "bin": f"Decile {i+1}",
                "range": (float(lo), float(hi)),
                "real_event_rate": float(real_rate),
                "synth_event_rate": float(synth_rate),
                "real_count": int(real_mask.sum()),
                "synth_count": int(synth_mask.sum()),
            }
        )
    return res


def tail_metrics(scores: np.ndarray, labels: np.ndarray, synth_scores: np.ndarray, synth_labels: np.ndarray) -> List[Dict]:
    thresholds = {
        "Top1%": np.percentile(scores, 99),
        "Top5%": np.percentile(scores, 95),
        "Top10%": np.percentile(scores, 90),
    }
    total_pos_real = labels.sum()
    total_pos_synth = synth_labels.sum()
    res = []
    for name, thr in thresholds.items():
        real_mask = scores >= thr
        synth_mask = synth_scores >= thr
        real_rate = labels[real_mask].mean() if real_mask.any() else math.nan
        synth_rate = synth_labels[synth_mask].mean() if synth_mask.any() else math.nan
        real_capture = labels[real_mask].sum() / total_pos_real if total_pos_real > 0 else math.nan
        synth_capture = synth_labels[synth_mask].sum() / total_pos_synth if total_pos_synth > 0 else math.nan
        res.append(
            {
                "segment": name,
                "threshold": float(thr),
                "real_event_rate": float(real_rate),
                "synth_event_rate": float(synth_rate),
                "real_capture_rate": float(real_capture),
                "synth_capture_rate": float(synth_capture),
                "real_count": int(real_mask.sum()),
                "synth_count": int(synth_mask.sum()),
            }
        )
    return res


def write_markdown(path: Path, metrics: Dict) -> None:
    m = metrics
    lines: List[str] = []
    lines.append("# 决策关键尾部一致性（风险评分）")
    lines.append("")
    lines.append("## 指标说明")
    lines.append("- PSI：分数分布偏移，0 表示无偏移，>0 越大偏移越多。")
    lines.append("- KS：两分布最大差异，0 表示重合，越大差异越大。")
    lines.append("- 分段校准：按真实分数的十分位分箱，对比每箱事件率（标签=1 比例）。")
    lines.append("- 尾部指标：按真实分数阈值（Top1/5/10%）计算事件率与抓取率（捕获坏样本比例）。")
    lines.append("")
    lines.append("## 全局分布对比")
    lines.append(f"- PSI（score）：{m['psi']:.6f}")
    lines.append(f"- KS（score）：{m['ks']:.6f}")
    lines.append("")
    lines.append("## 分段校准（deciles，基于真实分数分箱）")
    lines.append("| Decile | Score range | Real event rate | Synth event rate | Real count | Synth count |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in m["deciles"]:
        rng = f"[{row['range'][0]:.4f}, {row['range'][1]:.4f}]"
        lines.append(
            f"| {row['bin']} | {rng} | {row['real_event_rate']:.6f} | {row['synth_event_rate']:.6f} | {row['real_count']} | {row['synth_count']} |"
        )
    lines.append("")
    lines.append("## 尾部（高风险段）")
    lines.append("| Segment | Threshold | Real event rate | Synth event rate | Real capture | Synth capture | Real n | Synth n |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in m["tails"]:
        lines.append(
            f"| {row['segment']} | {row['threshold']:.6f} | {row['real_event_rate']:.6f} | {row['synth_event_rate']:.6f} | "
            f"{row['real_capture_rate']:.6f} | {row['synth_capture_rate']:.6f} | {row['real_count']} | {row['synth_count']} |"
        )
    lines.append("")
    lines.append("## 备注")
    lines.append("- 分数来源：真实数据训练的 Logistic 回归模型，对真实与合成数据评分。")
    lines.append("- 阈值与分箱均基于真实分数，保证比较基准一致。")
    lines.append("- capture rate=该段捕获的坏样本数量 / 全体坏样本数量。")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk scoring tail fidelity (polish_companies_bankruptcy).")
    parser.add_argument("--real-path", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy/data.csv"))
    parser.add_argument("--synth-path", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/Risk.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    real_raw = pd.read_csv(args.real_path)
    real_df, int_cols, float_cols, _ = clean_dataframe(real_raw)
    synth_raw = pd.read_csv(args.synth_path)
    synth_df, _, _, _ = clean_dataframe(synth_raw)

    clf = train_logistic(real_df, args.seed)
    real_scores = clf.predict_proba(real_df.drop(columns=[LABEL_COL]).to_numpy())[:, 1]
    synth_scores = clf.predict_proba(synth_df.drop(columns=[LABEL_COL]).to_numpy())[:, 1]
    real_labels = real_df[LABEL_COL].to_numpy()
    synth_labels = synth_df[LABEL_COL].to_numpy()

    psi_val = psi(real_scores, synth_scores, bins=10)
    ks_val = ks_2samp(real_scores, synth_scores).statistic
    deciles = decile_calibration(real_scores, real_labels, synth_scores, synth_labels)
    tails = tail_metrics(real_scores, real_labels, synth_scores, synth_labels)

    metrics = {
        "psi": psi_val,
        "ks": ks_val,
        "deciles": deciles,
        "tails": tails,
    }
    write_markdown(args.out_md, metrics)
    print(f"Wrote report to {args.out_md}")


if __name__ == "__main__":
    main()
