#!/usr/bin/env bash
# Data-free headline checks and figure generation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d "$ROOT/cp_svm" && -d "$ROOT/../cp_svm" ]]; then
  ROOT="$(cd "$ROOT/.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export SVATTN_OUTPUTS="${SVATTN_OUTPUTS:-$ROOT/outputs}"
PY="${PYTHON:-python3}"
MODE="${1:---quick}"

case "$MODE" in
  --quick)
    DELETE_ARGS=(--sizes 120 256 --trials 3 --warmup 1)
    ;;
  --paper)
    DELETE_ARGS=(
      --sizes 120 256 384 512
      --trials 30
      --seed 0
      --report "$SVATTN_OUTPUTS/deletion_latency_v2_reference_env.json"
    )
    ;;
  *)
    echo "usage: $0 [--quick|--paper]" >&2
    exit 2
    ;;
esac

mkdir -p "$SVATTN_OUTPUTS"

echo "=== quick contract and evidence checks ==="
PYTHON="$PY" bash reproducibility/run_quick.sh

echo "=== maintained and batched gradient checks ==="
"$PY" -m pytest \
  tests/test_diff_svdd.py \
  tests/test_fast_diff_svdd.py \
  tests/test_train_toy.py \
  tests/test_causal_layer.py \
  -p no:cacheprovider \
  -q

echo "=== fixed-C deletion latency ==="
"$PY" -m experiments.deletion_latency "${DELETE_ARGS[@]}"

echo "=== fixed-context selection and negative control ==="
"$PY" -m svattn.eviction_benchmark
"$PY" -m svattn.run_recall

echo "=== submitted claim-boundary figures ==="
"$PY" paper/make_v2_figures.py

echo "Headline tier completed; outputs are in $SVATTN_OUTPUTS."
