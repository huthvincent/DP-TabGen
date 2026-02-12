#!/usr/bin/env python
# path: sync_data_proj/batch_generate.py
"""批处理脚本：遍历 datasets/*/data.csv 并生成对应的同规模合成集。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generator import TabPFNConditionalGenerator  # noqa: E402

DATASETS_DIR = PROJECT_ROOT / "sync_data_proj" / "datasets"


def load_base_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def discover_datasets(names: Iterable[str]) -> List[Path]:
    if names:
        candidates = [DATASETS_DIR / name for name in names]
    else:
        candidates = sorted(p for p in DATASETS_DIR.iterdir() if p.is_dir())
    valid = []
    for path in candidates:
        if not path.exists():
            print(f"[WARN] 跳过不存在的目录: {path}")
            continue
        if not (path / "data.csv").exists():
            print(f"[WARN] {path} 缺少 data.csv，跳过。")
            continue
        valid.append(path)
    return valid


def build_generator_config(base: Dict[str, object], args: argparse.Namespace, sample_count: int) -> Dict[str, object]:
    cfg = base.copy()
    cfg["sample_count"] = sample_count
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.use_gpu:
        cfg["use_gpu"] = True
    if args.cpu_only:
        cfg["use_gpu"] = False
    if args.num_gibbs_rounds is not None:
        cfg["num_gibbs_rounds"] = args.num_gibbs_rounds
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    return cfg


def generate_for_dataset(dataset_dir: Path, base_config: Dict[str, object], args: argparse.Namespace) -> None:
    input_csv = dataset_dir / "data.csv"
    output_csv = dataset_dir / "sync_data.csv"
    df = pd.read_csv(input_csv)
    sample_count = len(df)
    cfg = build_generator_config(base_config, args, sample_count)
    generator = TabPFNConditionalGenerator(cfg)
    print(f"[INFO] 处理 {dataset_dir.name}: 输入 {sample_count} 行 -> 输出 {output_csv}")
    synthetic = generator.generate(str(input_csv), n_samples=sample_count)
    synthetic.to_csv(output_csv, index=False)
    print(f"[INFO] {dataset_dir.name} 完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量生成 sync dataset")
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "config/config.yaml",
        help="基础配置文件路径，将被每个数据集加载并按需覆盖",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="指定要处理的数据集目录名称（默认遍历 datasets 下所有含 data.csv 的目录）",
    )
    parser.add_argument("--seed", type=int, help="覆盖全局随机种子")
    parser.add_argument("--use-gpu", action="store_true", help="强制使用 GPU")
    parser.add_argument("--cpu-only", action="store_true", help="强制使用 CPU")
    parser.add_argument("--num-gibbs-rounds", type=int, help="覆盖 Gibbs 轮数")
    parser.add_argument("--batch-size", type=int, help="覆盖批大小")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_base_config(Path(args.config))
    dataset_dirs = discover_datasets(args.datasets or [])
    if not dataset_dirs:
        print("[WARN] 未找到任何可处理的数据集目录。")
        return
    for dataset_dir in dataset_dirs:
        generate_for_dataset(dataset_dir, base_config, args)


if __name__ == "__main__":
    main()
