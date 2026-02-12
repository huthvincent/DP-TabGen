#!/usr/bin/env bash
# Low-resource synthetic benchmark runner.

set -euo pipefail

# ---------------- user parameters ----------------
CONDA_ENV=sync_data
DATASET_DIR=/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy
PERCENTS=(0.1 0.2 0.3 0.4 0.5)
RESULTS_MD=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/low_resource.md
SEED=42
GENERATORS=(gaussian_copula sdv_ctgan sdv_tvae synthcity_ddpm)
# Generator hyper-parameters
CTGAN_EPOCHS=10
CTGAN_BATCH=64
CTGAN_PAC=1
TVAE_EPOCHS=10
TVAE_BATCH=64
DDPM_ITER=20
DDPM_BATCH=128
TABPFN_GIBBS=3
TABPFN_BATCH=256
TABPFN_CLIP_LOW=0.01
TABPFN_CLIP_HIGH=0.99
TABPFN_USE_GPU=0   # set 1 to request GPU; 0 to avoid CUDA checks

# Allow TabPFN CPU on large datasets if GPU missing
export TABPFN_ALLOW_CPU_LARGE_DATASET=1
# -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/low_resource.py"

python_cmd=(conda run -n "${CONDA_ENV}" python)

cmd=(
  "${python_cmd[@]}" "$PY"
  --dataset-dir "$DATASET_DIR"
  --results-md "$RESULTS_MD"
  --seed "$SEED"
  --percents "${PERCENTS[@]}"
  --ctgan-epochs "$CTGAN_EPOCHS"
  --ctgan-batch-size "$CTGAN_BATCH"
  --ctgan-pac "$CTGAN_PAC"
  --tvae-epochs "$TVAE_EPOCHS"
  --tvae-batch-size "$TVAE_BATCH"
  --ddpm-iter "$DDPM_ITER"
  --ddpm-batch-size "$DDPM_BATCH"
  --tabpfn-gibbs "$TABPFN_GIBBS"
  --tabpfn-batch-size "$TABPFN_BATCH"
  --tabpfn-clip-low "$TABPFN_CLIP_LOW"
  --tabpfn-clip-high "$TABPFN_CLIP_HIGH"
  --generators "${GENERATORS[@]}"
)

[ "$TABPFN_USE_GPU" -eq 1 ] && cmd+=(--tabpfn-use-gpu)

printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
