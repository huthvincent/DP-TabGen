#!/usr/bin/env bash
# Plot AUC learning curves for low-resource benchmarks (SOTA + TabPFGen).

set -euo pipefail

CONDA_ENV=sync_data
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/fig_low_resource.py"

python_cmd=(conda run -n "${CONDA_ENV}" python)

cmd=("${python_cmd[@]}" "$PY")
printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
