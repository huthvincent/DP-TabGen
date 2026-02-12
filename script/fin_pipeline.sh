#!/usr/bin/env bash
# Unified runner: TabPFGen (tabpfgen env) + other SOTA generators (sync_data env).
#
# - Reads dataset folders (Fin_data/* and EHR_datasets/*).
# - Runs TabPFGen in env `tabpfgen` and SOTA generators in env `sync_data`.
# - Writes intermediate markdown and a merged final markdown:
#     /home/zhu11/TabPFN/sync_data_proj/datasets/Results/tabpfgen.md
#
# User-friendly knobs live in the "User config" block below.

set -euo pipefail

# -------- User config --------
# Default behavior: run both TabPFGen + SOTA and merge results.
RUN_TABPFGEN=1
RUN_SOTA=1

# CPU thread controls
# Some servers have >128 CPU threads; OpenBLAS can crash with:
#   "Program is Terminated... maximum of 128 threads"
# Setting these values prevents thread oversubscription / OpenBLAS segfaults.
CPU_THREADS=32   # recommended <= 64

DATASET_DIRS=(
  /home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/*
  /home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/*
)

RESULTS_DIR=/home/zhu11/TabPFN/sync_data_proj/datasets/Results
TABPFGEN_PART_MD=$RESULTS_DIR/tabpfgen_part.md
SOTA_PART_MD=$RESULTS_DIR/sota_part.md
FINAL_MD=$RESULTS_DIR/tabpfgen.md

# TabPFGen params
SGLD_STEPS=600
SGLD_STEP_SIZE=0.01
SGLD_NOISE_SCALE=0.005
INIT_JITTER=0.01
SYNTH_FACTOR=1.0
TABPFGEN_ENERGY_SUBSAMPLE=2048  # set empty to disable, e.g. TABPFGEN_ENERGY_SUBSAMPLE=""
SEED=42

# SOTA params
CTGAN_EPOCHS=50
CTGAN_BATCH=64
CTGAN_PAC=1
TVAE_EPOCHS=50
TVAE_BATCH=64
DDPM_ITER=120
DDPM_BATCH=128
TABPFN_GIBBS=3
TABPFN_BATCH=256
TABPFN_CLIP_LOW=0.01
TABPFN_CLIP_HIGH=0.99
TABPFN_USE_GPU=1
# Space-separated list of SOTA generators to run (pass empty to use Python defaults)
SOTA_GENERATORS="gaussian_copula sdv_ctgan sdv_tvae synthcity_ddpm tabpfn"

# -----------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABPFGEN_PY="${SCRIPT_DIR}/fin_tabpfgen_pipeline.py"
SOTA_PY="${SCRIPT_DIR}/fin_sota_pipeline.py"

# -------- CLI options --------
# Examples:
#   bash fin_pipeline.sh --sota-only
#   bash fin_pipeline.sh --tabpfgen-only
#   bash fin_pipeline.sh --sota-only --sota-generators "gaussian_copula sdv_tvae"
#   bash fin_pipeline.sh --sota-only /path/to/dataset1 /path/to/dataset2
#
# Note: any remaining positional arguments are treated as dataset folders (overriding DATASET_DIRS).
while [ "$#" -gt 0 ]; do
  case "$1" in
    --sota-only)
      RUN_TABPFGEN=0
      shift
      ;;
    --tabpfgen-only)
      RUN_SOTA=0
      shift
      ;;
    --sota-generators)
      shift
      SOTA_GENERATORS="$1"
      shift
      ;;
    --help|-h)
      echo "Usage: bash fin_pipeline.sh [--sota-only|--tabpfgen-only] [--sota-generators \"g1 g2 ...\"] [dataset_dir1 dataset_dir2 ...]"
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done
if [ "$#" -gt 0 ]; then
  DATASET_DIRS=("$@")
fi

echo "Datasets:"
for d in "${DATASET_DIRS[@]}"; do
  [ -d "$d" ] && echo "  - $d" || echo "  - (skip, not a dir) $d"
done

# Threading limits (must be set *before* launching Python so BLAS/OpenMP pick them up)
export OMP_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export BLIS_NUM_THREADS="$CPU_THREADS"
export SYNC_PIPELINE_CPU_THREADS="$CPU_THREADS"

# Prepare results dir
mkdir -p "$RESULTS_DIR"

# Run TabPFGen part
if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
if [ "$RUN_TABPFGEN" -eq 1 ]; then
  echo "Running TabPFGen in tabpfgen env..."
  conda activate tabpfgen 2>/dev/null
  PYTHONUNBUFFERED=1 python "$TABPFGEN_PY" \
    --datasets "${DATASET_DIRS[@]}" \
    --results-md "$TABPFGEN_PART_MD" \
    --sgld-steps "$SGLD_STEPS" \
    --sgld-step-size "$SGLD_STEP_SIZE" \
    --sgld-noise-scale "$SGLD_NOISE_SCALE" \
    --jitter "$INIT_JITTER" \
    --synthetic-factor "$SYNTH_FACTOR" \
    $( [ -n "${TABPFGEN_ENERGY_SUBSAMPLE}" ] && echo "--energy-subsample" "$TABPFGEN_ENERGY_SUBSAMPLE" ) \
    --seed "$SEED" \
    --generators tabpfgen
else
  echo "Skipping TabPFGen (RUN_TABPFGEN=0)."
fi

# Run SOTA part
if [ "$RUN_SOTA" -eq 1 ]; then
  echo "Running SOTA generators in sync_data env..."
  conda activate sync_data 2>/dev/null
  PYTHONUNBUFFERED=1 python "$SOTA_PY" \
    --datasets "${DATASET_DIRS[@]}" \
    --results-md "$SOTA_PART_MD" \
    --seed "$SEED" \
    --ctgan-epochs "$CTGAN_EPOCHS" \
    --ctgan-batch-size "$CTGAN_BATCH" \
    --ctgan-pac "$CTGAN_PAC" \
    --tvae-epochs "$TVAE_EPOCHS" \
    --tvae-batch-size "$TVAE_BATCH" \
    --ddpm-iter "$DDPM_ITER" \
    --ddpm-batch-size "$DDPM_BATCH" \
    --tabpfn-gibbs "$TABPFN_GIBBS" \
    --tabpfn-batch-size "$TABPFN_BATCH" \
    --tabpfn-clip-low "$TABPFN_CLIP_LOW" \
    --tabpfn-clip-high "$TABPFN_CLIP_HIGH" \
    $( [ "$TABPFN_USE_GPU" -eq 1 ] && echo "--tabpfn-use-gpu" ) \
    $( [ -n "${SOTA_GENERATORS:-}" ] && echo --generators $SOTA_GENERATORS )
else
  echo "Skipping SOTA generators (RUN_SOTA=0)."
fi

# Merge
{
  echo "# Synthetic TSTR Results"
  echo ""
  if [ -f "$TABPFGEN_PART_MD" ]; then
    echo "## TabPFGen"
    cat "$TABPFGEN_PART_MD"
    echo ""
  fi
  echo ""
  if [ -f "$SOTA_PART_MD" ]; then
    echo "## Other SOTA Generators"
    cat "$SOTA_PART_MD"
  fi
} > "$FINAL_MD"

echo "Done. See $FINAL_MD"
