#!/usr/bin/env bash
# Plot trade-off with MIA overlay.

set -euo pipefail

CONDA_ENV=sync_data
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/plot_mia.py"

python_cmd=(conda run -n "${CONDA_ENV}" python)

cmd=("${python_cmd[@]}" "$PY")
printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
