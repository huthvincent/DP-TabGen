#!/usr/bin/env bash
# Run KDE-based membership inference analysis.

set -euo pipefail

CONDA_ENV=sync_data
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/mia.py"

python_cmd=(conda run -n "${CONDA_ENV}" python)

cmd=("${python_cmd[@]}" "$PY")
printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
