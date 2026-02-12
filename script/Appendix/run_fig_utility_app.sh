#!/usr/bin/env bash
# Run appendix plots (grouped by model).

set -euo pipefail

# ---------------- user parameters ----------------
CONDA_ENV=plot            # change if plot env unavailable
SOTA_MD=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/sota_part.md
TABPFGEN_MD=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/tabpfgen_part.md
# Limit to financial datasets; adjust as needed
DATASETS=(australian_credit_approval bank_marketing german_credit polish_companies_bankruptcy)
METRICS=(roc_auc)
MODELS=(logistic xgboost lightgbm catboost mlp)
OUTPUT_DIR=/home/zhu11/TabPFN/sync_data_proj/plots/utility/appendix
YMAX=                     # optional y-limit
PALETTE=()  # leave empty to use default palette from fig_utility_app.py
# -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/fig_utility_app.py"

# Try to use the selected conda env; fall back to system python on failure.
python_cmd="python"
if command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda/etc/profile.d/conda.sh"
  fi
  if conda env list >/dev/null 2>&1 && conda env list | grep -q "^${CONDA_ENV} "; then
    python_cmd="conda run -n ${CONDA_ENV} python"
  fi
fi

cmd=(
  "$python_cmd" "$PY"
  --sota-md "$SOTA_MD"
  --tabpfgen-md "$TABPFGEN_MD"
  --output-dir "$OUTPUT_DIR"
  --metrics "${METRICS[@]}"
  --models "${MODELS[@]}"
)
[ ${#DATASETS[@]} -gt 0 ] && cmd+=(--datasets "${DATASETS[@]}")
[ -n "${YMAX}" ] && cmd+=(--ymax "$YMAX")
[ ${#PALETTE[@]} -gt 0 ] && cmd+=(--palette "${PALETTE[@]}")

printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
