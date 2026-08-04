#!/usr/bin/env bash
# Fast, data-free CPU verification using only requirements-core.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d "$ROOT/cp_svm" && -d "$ROOT/../cp_svm" ]]; then
  ROOT="$(cd "$ROOT/.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python3}"

"$PY" -m pytest \
  tests/test_fast_solver.py \
  tests/test_clinical_deletion.py \
  tests/test_clinical_selection.py \
  -p no:cacheprovider \
  -q

"$PY" reproducibility/check_evidence.py

echo "Quick contract and evidence verification passed."
