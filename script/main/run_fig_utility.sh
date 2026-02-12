#!/usr/bin/env bash
# Run utility plots (main paper).
# Adjust parameters below; all options are passed to fig_utility.py.

set -euo pipefail

# ---------------- user parameters ----------------
CONDA_ENV=plot            # change to an existing env if plot is not available
SOTA_MD=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/sota_part.md
TABPFGEN_MD=/home/zhu11/TabPFN/sync_data_proj/datasets/Results/tabpfgen_part.md
DATASETS=(australian_credit_approval bank_marketing german_credit polish_companies_bankruptcy)
METRICS=(roc_auc)         # add pr_auc if available in markdown
GROUP_MODE=average        # average | by_model
MODELS=(logistic xgboost lightgbm catboost mlp)
OUTPUT_DIR=/home/zhu11/TabPFN/sync_data_proj/plots/utility
YMAX=                     # optional y-axis upper limit, leave empty for auto
# Palette is optional; leave empty to use defaults in fig_utility.py
# Palette (color-blind friendly); leave empty to use fig_utility.py defaults
PALETTE=()  # leave empty to use fig_utility.py defaults (darker color-blind palette)
# -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/fig_utility.py"

# Try to use the chosen conda env, but fall back to bare python if conda is
# unavailable or `conda env list` fails (some systems block that call).
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
  --group-mode "$GROUP_MODE"
  --output-dir "$OUTPUT_DIR"
)
[ ${#DATASETS[@]} -gt 0 ] && cmd+=(--datasets "${DATASETS[@]}")
[ ${#METRICS[@]} -gt 0 ] && cmd+=(--metrics "${METRICS[@]}")
[ ${#MODELS[@]} -gt 0 ] && cmd+=(--models "${MODELS[@]}")
[ -n "${YMAX}" ] && cmd+=(--ymax "$YMAX")
[ ${#PALETTE[@]} -gt 0 ] && cmd+=(--palette "${PALETTE[@]}")

printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
