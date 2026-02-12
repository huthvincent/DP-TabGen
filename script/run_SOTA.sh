#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_SOTA.sh <dataset> <train_csv_name>
# Example: bash run_SOTA.sh heart synthetic_100.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets"

DATASET="${1:-heart}"
VARIANT="${2:-synthetic_100.csv}"

# Map dataset -> target column (override via TARGET env if needed)
TARGET="${TARGET:-}"
if [[ -z "${TARGET}" ]]; then
  case "${DATASET}" in
    arrhythmia) TARGET="label_arrhythmia_class" ;;
    ckd) TARGET="label_class" ;;
    heart) TARGET="label_heart_disease" ;;
    pima) TARGET="label_diabetes" ;;
    uti) TARGET="label_nephritis_of_renal_pelvis_origin" ;;
    wdbc) TARGET="label_diagnosis" ;;
    *) echo "Unknown dataset '${DATASET}', please set TARGET env"; exit 1 ;;
  esac
fi

TRAIN_CSV="${BASE}/${DATASET}/${VARIANT}"
TEST_CSV="${BASE}/${DATASET}/dataset.csv"

if [[ ! -f "${TRAIN_CSV}" ]]; then
  echo "Training CSV not found: ${TRAIN_CSV}" >&2
  exit 1
fi
if [[ ! -f "${TEST_CSV}" ]]; then
  echo "Test CSV not found: ${TEST_CSV}" >&2
  exit 1
fi

SEED="${SEED:-3407}"
N_JOBS="${N_JOBS:--1}"
VAL_SPLIT="${VAL_SPLIT:-0.0}"
MODELS="${MODELS:-logistic xgboost lightgbm catboost mlp}"
REPORT_PATH="${REPORT_PATH:-${BASE}/${DATASET}/metrics_${VARIANT%.csv}.csv}"
PARAMS_JSON="${PARAMS_JSON:-}"

cmd=(python "${SCRIPT_DIR}/run_SOTA.py"
  --train-csv "${TRAIN_CSV}"
  --test-csv "${TEST_CSV}"
  --target "${TARGET}"
  --seed "${SEED}"
  --n-jobs "${N_JOBS}"
  --val-split "${VAL_SPLIT}"
  --report-path "${REPORT_PATH}"
  --models)

for m in ${MODELS}; do
  cmd+=("${m}")
done

if [[ -n "${PARAMS_JSON}" ]]; then
  cmd+=(--params "${PARAMS_JSON}")
fi

echo "Running SOTA: dataset=${DATASET}, train=${TRAIN_CSV}, test=${TEST_CSV}, models=${MODELS}"
"${cmd[@]}"
