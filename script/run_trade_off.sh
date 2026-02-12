#!/usr/bin/env bash
# Run TabPFGen noise-scale trade-off experiment.

set -euo pipefail

CONDA_ENV=tabpfgen
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/trade_off.py"

python_cmd=(conda run -n "${CONDA_ENV}" python)

cmd=("${python_cmd[@]}" "$PY")
printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
