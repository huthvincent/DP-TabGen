#!/usr/bin/env bash
# Run TVAE indistinguishability analysis and compare with TabPFGen results.

set -euo pipefail

CONDA_ENV=sync_data
DATASET_DIR=/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy
OUT_CSV=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability_tvae.csv
OUT_MD=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/Indistinguishability.md
TABPFGEN_CSV=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability.csv
SEED=42
TVAE_EPOCHS=50
TVAE_BATCH=256

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/Indistinguishability_tvae.py"

python_cmd=(conda run -n "${CONDA_ENV}" python)

cmd=(
  "${python_cmd[@]}" "$PY"
  --dataset-dir "$DATASET_DIR"
  --out-csv "$OUT_CSV"
  --out-md "$OUT_MD"
  --tabpfgen-csv "$TABPFGEN_CSV"
  --seed "$SEED"
  --tvae-epochs "$TVAE_EPOCHS"
  --tvae-batch-size "$TVAE_BATCH"
)

printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
