"""
Membership inference analysis (KDE-based) on polish_companies_bankruptcy synthetic datasets.

For each synthetic dataset (different sgld_noise_scale):
 1) Clean synthetic + real data.
 2) Split real data into member/non-member (train/test).
 3) Fit scaler + PCA on synthetic features only, then Gaussian KDE in PCA space (bandwidth via Scott rule).
 4) Score member/non-member records (log-density) and compute:
    - Attack AUC
    - TPR@FPR=1% and 5% (threshold on non-member scores)
Results written to datasets/Results/MIA.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT / "script") not in sys.path:
    sys.path.append(str(ROOT / "script"))

from fin_tabpfgen_pipeline import LABEL_COL, clean_dataframe, set_seed  # type: ignore


def fit_kde(synth_feat: np.ndarray, bandwidth: float) -> KernelDensity:
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(synth_feat)
    return kde


def scott_bandwidth(data: np.ndarray) -> float:
    n, d = data.shape
    if n <= 1:
        return 1.0
    h = n ** (-1.0 / (d + 4))
    return float(h)


def scores_for(kde: KernelDensity, scaler: StandardScaler, pca: PCA, X: np.ndarray) -> np.ndarray:
    X_std = scaler.transform(X)
    X_pca = pca.transform(X_std)
    return kde.score_samples(X_pca)


def tpr_at_fpr(member_scores: np.ndarray, non_scores: np.ndarray, target_fpr: float) -> float:
    thr = np.percentile(non_scores, 100 * (1 - target_fpr))
    tpr = (member_scores >= thr).mean()
    return float(tpr)


def write_markdown(out_path: Path, rows: List[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Membership inference (KDE in PCA space)")
    lines.append("")
    lines.append("攻击流程：用合成数据拟合 StandardScaler+PCA+Gaussian KDE，对候选记录打分（log-density）。若生成器过拟合，成员得分更高。")
    lines.append("")
    lines.append("| noise tag | bandwidth | PCA dim | members | non-members | AUC | TPR@1% | TPR@5% |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['bandwidth']:.4f} | {r['pca_dim']} | {r['n_members']} | {r['n_nonmembers']} | "
            f"{r['auc']:.4f} | {r['tpr1']:.4f} | {r['tpr5']:.4f} |"
        )
    lines.append("")
    lines.append("说明：AUC 越接近 0.5 越好（越难区分）；TPR@FPR 越低越好。bandwidth 使用 Scott rule 在合成数据上估计。")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="KDE-based MIA on synthetic datasets.")
    parser.add_argument("--real-path", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy/data.csv"))
    parser.add_argument("--synth-dir", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/trade_off"))
    parser.add_argument("--out-md", type=Path, default=Path("/home/zhu11/TabPFN/sync_data_proj/datasets/Results/MIA.md"))
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--max-kde-samples", type=int, default=20000, help="Subsample synthetic rows for KDE fit.")
    parser.add_argument("--max-eval-samples", type=int, default=8000, help="Subsample member/non-member for scoring.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    real_raw = pd.read_csv(args.real_path)
    real_df, _, _, _ = clean_dataframe(real_raw)
    feature_cols = [c for c in real_df.columns if c != LABEL_COL]
    X_real = real_df[feature_cols].to_numpy()
    y_real = real_df[LABEL_COL].to_numpy()

    # Split member/non-member from real data
    X_mem, X_non, y_mem, y_non = train_test_split(X_real, y_real, test_size=0.2, random_state=args.seed, stratify=y_real)

    synth_files = [
        ("5e-3", args.synth_dir / "5e-3.csv"),
        ("1e-2", args.synth_dir / "1e-2.csv"),
        ("15e-3", args.synth_dir / "15e-3.csv"),
        ("2e-2", args.synth_dir / "2e-2.csv"),
        ("25e-3", args.synth_dir / "25e-3.csv"),
        ("3e-2", args.synth_dir / "3e-2.csv"),
    ]

    results = []
    for tag, path in synth_files:
        if not path.exists():
            continue
        synth_raw = pd.read_csv(path)
        synth_df, _, _, _ = clean_dataframe(synth_raw)
        synth_X = synth_df[feature_cols].to_numpy()

        # Subsample synthetic for KDE fit
        if len(synth_X) > args.max_kde_samples:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(synth_X), size=args.max_kde_samples, replace=False)
            synth_X_fit = synth_X[idx]
        else:
            synth_X_fit = synth_X

        scaler = StandardScaler()
        synth_std = scaler.fit_transform(synth_X_fit)
        pca_dim = min(args.pca_dim, synth_std.shape[1])
        pca = PCA(n_components=pca_dim, random_state=args.seed)
        synth_pca = pca.fit_transform(synth_std)
        bw = scott_bandwidth(synth_pca)
        kde = fit_kde(synth_pca, bandwidth=bw)

        # Subsample for evaluation
        rng = np.random.default_rng(args.seed)
        mem_eval = X_mem if len(X_mem) <= args.max_eval_samples else X_mem[rng.choice(len(X_mem), size=args.max_eval_samples, replace=False)]
        non_eval = X_non if len(X_non) <= args.max_eval_samples else X_non[rng.choice(len(X_non), size=args.max_eval_samples, replace=False)]

        mem_scores = scores_for(kde, scaler, pca, mem_eval)
        non_scores = scores_for(kde, scaler, pca, non_eval)
        labels = np.concatenate([np.ones_like(mem_scores), np.zeros_like(non_scores)])
        scores = np.concatenate([mem_scores, non_scores])
        try:
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = float("nan")
        tpr1 = tpr_at_fpr(mem_scores, non_scores, target_fpr=0.01)
        tpr5 = tpr_at_fpr(mem_scores, non_scores, target_fpr=0.05)

        results.append(
            {
                "tag": tag,
                "bandwidth": bw,
                "pca_dim": pca_dim,
                "n_members": len(mem_scores),
                "n_nonmembers": len(non_scores),
                "auc": auc,
                "tpr1": tpr1,
                "tpr5": tpr5,
            }
        )

    write_markdown(args.out_md, results)
    print(f"Wrote MIA results to {args.out_md}")


if __name__ == "__main__":
    main()
