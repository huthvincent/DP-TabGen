#!/usr/bin/env bash
# Run TabPFGen indistinguishability analysis on polish_companies_bankruptcy.

set -euo pipefail

CONDA_ENV=tabpfgen
DATASET_DIR=/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy
OUT_CSV=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability.csv
OUT_MD=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/Indistinguishability.md
SEED=42
# TabPFGen params (aligned with tabpfgen_part.md)
N_SGLD_STEPS=600
SGLD_STEP_SIZE=0.01
SGLD_NOISE_SCALE=0.005
JITTER=0.01
SYNTHETIC_FACTOR=1.0
ENERGY_SUBSAMPLE=2048   # set 0 to disable subsampling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/Indistinguishability.py"

python_cmd=(conda run -n "${CONDA_ENV}" python)

cmd=(
  "${python_cmd[@]}" "$PY"
  --dataset-dir "$DATASET_DIR"
  --out-csv "$OUT_CSV"
  --out-md "$OUT_MD"
  --seed "$SEED"
  --n-sgld-steps "$N_SGLD_STEPS"
  --sgld-step-size "$SGLD_STEP_SIZE"
  --sgld-noise-scale "$SGLD_NOISE_SCALE"
  --jitter "$JITTER"
  --synthetic-factor "$SYNTHETIC_FACTOR"
  --energy-subsample "$ENERGY_SUBSAMPLE"
)

printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
