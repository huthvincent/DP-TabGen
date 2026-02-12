#!/usr/bin/env bash
# Low-resource benchmark runner for TabPFGen.

set -euo pipefail

# ---------------- user parameters ----------------
CONDA_ENV=tabpfgen
DATASET_DIR=/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy
PERCENTS=(0.1 0.2 0.3 0.4 0.5)
RESULTS_MD=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/tabpfgen_low_resource.md
SEED=42
# TabPFGen hyper-parameters (from tabpfgen_part.md defaults)
N_SGLD_STEPS=600
SGLD_STEP_SIZE=0.01
SGLD_NOISE_SCALE=0.005
JITTER=0.01
SYNTHETIC_FACTOR=1.0
ENERGY_SUBSAMPLE=2048   # set 0 to disable subsampling
# -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/tabpfgen_low_resource.py"

python_cmd=(conda run -n "${CONDA_ENV}" python)

cmd=(
  "${python_cmd[@]}" "$PY"
  --dataset-dir "$DATASET_DIR"
  --results-md "$RESULTS_MD"
  --seed "$SEED"
  --percents "${PERCENTS[@]}"
  --n-sgld-steps "$N_SGLD_STEPS"
  --sgld-step-size "$SGLD_STEP_SIZE"
  --sgld-noise-scale "$SGLD_NOISE_SCALE"
  --jitter "$JITTER"
  --synthetic-factor "$SYNTHETIC_FACTOR"
  --energy-subsample "$ENERGY_SUBSAMPLE"
)

printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
