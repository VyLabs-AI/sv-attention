#!/usr/bin/env bash
# Credentialed MIMIC-IV v3.1 tier. No raw data or derived cache is redistributed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d "$ROOT/cp_svm" && -d "$ROOT/../cp_svm" ]]; then
  ROOT="$(cd "$ROOT/.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python3}"
MODE="${1:-all}"
shift || true

ICU_PATH="${MIMICIV_ICU_PATH:-}"
if [[ "${1:-}" == "--icu-path" ]]; then
  [[ $# -ge 2 ]] || { echo "--icu-path requires a path" >&2; exit 2; }
  ICU_PATH="$2"
  shift 2
fi
[[ $# -eq 0 ]] || { echo "usage: $0 {prepare|analyze|audit|all} [--icu-path PATH]" >&2; exit 2; }

CACHE="clinical_seq/cache/icu_vitals_n1500.npz"
mkdir -p clinical_seq/cache outputs

prepare() {
  [[ -n "$ICU_PATH" ]] || {
    echo "Set MIMICIV_ICU_PATH or pass --icu-path for prepare/all." >&2
    exit 2
  }
  "$PY" -m clinical_seq.build_icu_sequences \
    --icu-path "$ICU_PATH" \
    --n-stays 1500 \
    --min-los-days 2.0 \
    --min-hours 48 \
    --seed 0 \
    --out "$CACHE"
}

analyze() {
  [[ -f "$CACHE" ]] || {
    echo "$CACHE is missing; run '$0 prepare --icu-path PATH' first." >&2
    exit 2
  }

  # Full eligible cohort and deterioration-hour analyses.
  "$PY" -m clinical_seq.e2_selection \
    --cache "$CACHE" \
    --max-stays 1465 \
    --report outputs/mimic_composite_selection.json
  "$PY" -m clinical_seq.e2_selection \
    --cache "$CACHE" \
    --max-stays 1465 \
    --held-out-vital spo2 \
    --report outputs/mimic_held_out_spo2.json
  "$PY" -m clinical_seq.e3_forget --cache "$CACHE" --trials 100
  "$PY" -m experiments.p2_mimic
}

audit() {
  [[ -f "$CACHE" ]] || {
    echo "$CACHE is missing; run '$0 prepare --icu-path PATH' first." >&2
    exit 2
  }
  [[ -f outputs/qa_nlp_model.pt ]] || {
    echo "outputs/qa_nlp_model.pt is missing; run reproducibility/run_models.sh qa first." >&2
    exit 2
  }
  "$PY" -m experiments.forgetting_rigor \
    --trials 300 \
    --fig outputs/forgetting_rigor_v2.png \
    --report outputs/forgetting_rigor_v2.json
}

case "$MODE" in
  prepare) prepare ;;
  analyze) analyze ;;
  audit) audit ;;
  all) prepare; analyze ;;
  *) echo "usage: $0 {prepare|analyze|audit|all} [--icu-path PATH]" >&2; exit 2 ;;
esac
