#!/usr/bin/env python3
import csv
import logging
import sys
import time
from pathlib import Path


def ensure_packages():
    import importlib
    import subprocess

    packages = {
        "pandas": "pandas",
        "requests": "requests",
        "liac-arff": "arff",
    }
    for pkg, module in packages.items():
        try:
            importlib.import_module(module)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])


ensure_packages()

import pandas as pd
import requests
import zipfile
import arff

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path("/home/zhu11/TabPFN/sync_data_proj/datasets")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def get_csv_shape(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        row_count = sum(1 for _ in reader)
    return row_count, len(header)


def ensure_label_last(df: pd.DataFrame) -> pd.DataFrame:
    if "label" in df.columns and df.columns[-1] != "label":
        ordered_cols = [col for col in df.columns if col != "label"] + ["label"]
        df = df[ordered_cols]
    return df


def download_file(url: str, dest_path: Path, retries: int = 3, chunk_size: int = 1 << 20, timeout: int = 60) -> Path:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(dest_path.name + ".part")
    session = requests.Session()
    try:
        for attempt in range(1, retries + 1):
            try:
                existing_size = temp_path.stat().st_size if temp_path.exists() else 0
                headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
                with session.get(url, stream=True, timeout=timeout, headers=headers) as response:
                    response.raise_for_status()
                    mode = "ab" if existing_size and response.status_code == 206 else "wb"
                    if mode == "wb" and temp_path.exists():
                        temp_path.unlink()
                    with open(temp_path, mode) as out_file:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                out_file.write(chunk)
                if dest_path.exists():
                    dest_path.unlink()
                temp_path.rename(dest_path)
                return dest_path
            except Exception as exc:
                logging.warning("Attempt %d to download %s failed: %s", attempt, url, exc)
                if attempt == retries:
                    temp_path.unlink(missing_ok=True)
                    raise
                time.sleep(2)
    finally:
        session.close()
    raise RuntimeError(f"Unable to download {url}")


def extract_zip(zip_path: Path, target_dir: Path):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def map_german_labels(series: pd.Series) -> pd.Series:
    non_na = series.dropna()
    if not non_na.empty:
        numeric_series = pd.to_numeric(non_na, errors="coerce")
        if not numeric_series.isna().any():
            unique_vals = set(numeric_series.astype(int).unique())
            if unique_vals.issubset({1, 2}) and len(unique_vals) == 2:
                coerced_full = pd.to_numeric(series, errors="coerce").astype("Int64")
                mapped = coerced_full.map({1: 0, 2: 1})
                return mapped.astype("Int64")
    counts = non_na.value_counts()
    if counts.empty:
        raise ValueError("Unable to infer German Credit labels.")
    mapping = {counts.index[0]: 0}
    for cls in counts.index[1:]:
        mapping[cls] = 1
    logging.info("German credit fallback label mapping: %s", mapping)
    mapped_series = series.map(mapping)
    return mapped_series.astype("Int64")


def load_arff_dataframe(arff_path: Path) -> pd.DataFrame:
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as handle:
        dataset = arff.load(handle)
    columns = [attr[0] for attr in dataset["attributes"]]
    df = pd.DataFrame(dataset["data"], columns=columns)
    df = df.replace("?", pd.NA)
    label_columns = [col for col in df.columns if col.lower() == "class"]
    if label_columns:
        df = df.rename(columns={label_columns[0]: "label"})
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    return df


def prepare_german_credit(base_dir: Path):
    dataset_dir = base_dir / "german_credit"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target_path = dataset_dir / "data.csv"
    if target_path.exists():
        return get_csv_shape(target_path)
    sources = [
        ("numeric", "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"),
        ("symbolic", "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"),
    ]
    raw_path = None
    for label, url in sources:
        candidate = dataset_dir / url.split("/")[-1]
        try:
            download_file(url, candidate)
            raw_path = candidate
            logging.info("Downloaded German Credit (%s) from %s", label, url)
            break
        except Exception as exc:
            logging.warning("Failed to download German Credit source %s: %s", url, exc)
    if raw_path is None:
        raise RuntimeError("Unable to download German Credit dataset.")
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, na_values=["?"], engine="python")
    total_cols = df.shape[1]
    if total_cols < 2:
        raise ValueError("German Credit dataset is missing columns.")
    feature_cols = [f"f{i}" for i in range(1, total_cols)]
    df.columns = feature_cols + ["label"]
    df["label"] = map_german_labels(df["label"])
    df.to_csv(target_path, index=False, encoding="utf-8")
    return df.shape


def prepare_australian_credit(base_dir: Path):
    dataset_dir = base_dir / "australian_credit_approval"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target_path = dataset_dir / "data.csv"
    if target_path.exists():
        return get_csv_shape(target_path)
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
    raw_path = dataset_dir / "australian.dat"
    download_file(url, raw_path)
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, na_values=["?"], engine="python")
    if df.shape[1] != 15:
        raise ValueError("Australian Credit dataset should have 15 columns.")
    feature_cols = [f"f{i}" for i in range(1, 15)]
    df.columns = feature_cols + ["label"]
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df.to_csv(target_path, index=False, encoding="utf-8")
    return df.shape


def prepare_bank_marketing(base_dir: Path):
    dataset_dir = base_dir / "bank_marketing"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target_path = dataset_dir / "data.csv"
    if target_path.exists():
        return get_csv_shape(target_path)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    zip_path = dataset_dir / "bank-additional.zip"
    download_file(url, zip_path)
    extract_zip(zip_path, dataset_dir)
    csv_path = dataset_dir / "bank-additional" / "bank-additional-full.csv"
    if not csv_path.exists():
        raise FileNotFoundError("bank-additional-full.csv not found after extraction.")
    df = pd.read_csv(csv_path, sep=";", na_values=["?"])
    if "y" not in df.columns:
        raise ValueError("Target column 'y' missing from Bank Marketing dataset.")
    df = df.rename(columns={"y": "label"})
    label_series = df["label"].astype(str).str.strip().str.lower()
    df["label"] = label_series.map({"yes": 1, "no": 0})
    if df["label"].isna().any():
        raise ValueError("Bank Marketing dataset contains unexpected label values.")
    df["label"] = df["label"].astype("Int64")
    df = ensure_label_last(df)
    df.to_csv(target_path, index=False, encoding="utf-8")
    return df.shape


def prepare_polish_companies_bankruptcy(base_dir: Path):
    dataset_dir = base_dir / "polish_companies_bankruptcy"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target_path = dataset_dir / "data.csv"
    if target_path.exists():
        return get_csv_shape(target_path)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip"
    zip_path = dataset_dir / "data.zip"
    download_file(url, zip_path)
    extract_zip(zip_path, dataset_dir)
    arff_files = sorted(dataset_dir.rglob("*year.arff"))
    if not arff_files:
        raise FileNotFoundError("No ARFF files found for Polish Companies Bankruptcy dataset.")
    frames = [load_arff_dataframe(path) for path in arff_files]
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "label" not in df.columns:
        raise ValueError("Label column missing after loading ARFF files.")
    df = ensure_label_last(df)
    df.to_csv(target_path, index=False, encoding="utf-8")
    return df.shape


def main():
    tasks = [
        ("german_credit", prepare_german_credit),
        ("australian_credit_approval", prepare_australian_credit),
        ("bank_marketing", prepare_bank_marketing),
        ("polish_companies_bankruptcy", prepare_polish_companies_bankruptcy),
    ]
    results = []
    for name, handler in tasks:
        rows, cols = handler(BASE_DIR)
        results.append((name, rows, cols))
    for name, rows, cols in results:
        print(f"{name}/data.csv -> {rows} rows, {cols} cols")


if __name__ == "__main__":
    main()
