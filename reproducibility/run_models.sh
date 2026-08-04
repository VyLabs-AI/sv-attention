#!/usr/bin/env bash
# Optional model/download tier. Nothing here is required for the CPU quick check.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d "$ROOT/cp_svm" && -d "$ROOT/../cp_svm" ]]; then
  ROOT="$(cd "$ROOT/.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python3}"
MODE="${1:-}"
mkdir -p outputs

case "$MODE" in
  qa)
    # The current trained-key path uses the MLX gate.
    QA_STEPS="${QA_STEPS:-500}"
    "$PY" -m experiments.forget_qa_nlp \
      --steps "$QA_STEPS" \
      --cache outputs/qa_nlp_model.pt \
      --fig outputs/forget_qa_nlp.png
    "$PY" -m experiments.interpretable_demos \
      --fig outputs/interpretable_demos.png
    ;;
  lm)
    : "${CORPUS:?Set CORPUS to an enwik8 or plain-text byte corpus path.}"
    MODEL_SEEDS="${MODEL_SEEDS:-0 1 2 3 4 5 6}"
    MODEL_STEPS="${MODEL_STEPS:-6000}"
    SVATTN_SOLVER="${SVATTN_SOLVER:-mlx}"
    for seed in $MODEL_SEEDS; do
      if [[ "$seed" == "0" ]]; then
        EVAL_EVERY=500
        CKPT_EVERY=250
      else
        EVAL_EVERY=1000
        CKPT_EVERY=500
      fi
      "$PY" -m experiments.a2_quality \
        --data "$CORPUS" \
        --n_bytes 5000000 \
        --steps "$MODEL_STEPS" \
        --batch 16 \
        --block 256 \
        --d_model 256 \
        --n_layers 4 \
        --n_heads 4 \
        --chunk 64 \
        --readout softmax \
        --solver "$SVATTN_SOLVER" \
        --fista_iters 80 \
        --eval_every "$EVAL_EVERY" \
        --ckpt_every "$CKPT_EVERY" \
        --n_eval 20 \
        --seed "$seed" \
        --variants sv swa \
        --fresh \
        --ckpt "outputs/a2_seed${seed}_checkpoint.pt" \
        --save_sv "outputs/a2_seed${seed}_sv_model.pt"
    done
    ;;
  lm10m)
    : "${CORPUS:?Set CORPUS to the enwik8 byte corpus path.}"
    MODEL_SEEDS="${MODEL_SEEDS:-0 1 2 3 4}"
    MODEL_STEPS="${MODEL_STEPS:-6000}"
    SVATTN_SOLVER="${SVATTN_SOLVER:-mlx}"
    for seed in $MODEL_SEEDS; do
      if [[ "$seed" == "0" ]]; then
        EVAL_EVERY=500
        CKPT_EVERY=250
      else
        EVAL_EVERY=1000
        CKPT_EVERY=500
      fi
      "$PY" -m experiments.a2_quality \
        --data "$CORPUS" \
        --n_bytes 20000000 \
        --steps "$MODEL_STEPS" \
        --batch 16 \
        --block 512 \
        --d_model 384 \
        --n_layers 6 \
        --n_heads 6 \
        --chunk 64 \
        --readout softmax \
        --solver "$SVATTN_SOLVER" \
        --fista_iters 80 \
        --eval_every "$EVAL_EVERY" \
        --ckpt_every "$CKPT_EVERY" \
        --n_eval 20 \
        --seed "$seed" \
        --variants sv swa \
        --fresh \
        --ckpt "outputs/a2_10m_seed${seed}_checkpoint.pt" \
        --save_sv "outputs/a2_10m_seed${seed}_sv_model.pt"
    done
    ;;
  lm32m)
    : "${CORPUS:?Set CORPUS to the enwik8 byte corpus path.}"
    MODEL_SEEDS="${MODEL_SEEDS:-0}"
    MODEL_STEPS="${MODEL_STEPS:-6000}"
    SVATTN_SOLVER="${SVATTN_SOLVER:-mlx}"
    for seed in $MODEL_SEEDS; do
      "$PY" -m experiments.a2_quality \
        --data "$CORPUS" \
        --n_bytes 20000000 \
        --steps "$MODEL_STEPS" \
        --batch 16 \
        --block 512 \
        --d_model 512 \
        --n_layers 10 \
        --n_heads 8 \
        --chunk 64 \
        --readout softmax \
        --solver "$SVATTN_SOLVER" \
        --fista_iters 80 \
        --eval_every 500 \
        --ckpt_every 500 \
        --n_eval 20 \
        --seed "$seed" \
        --variants sv swa \
        --fresh \
        --ckpt "outputs/a2_32m_seed${seed}_checkpoint.pt" \
        --save_sv "outputs/a2_32m_seed${seed}_sv_model.pt"
    done
    ;;
  throughput)
    "$PY" -m experiments.g1_throughput --paper
    ;;
  *)
    echo "usage: $0 {qa|lm|lm10m|lm32m|throughput}" >&2
    echo "  qa   trained-key deletion/editing audit; requires MLX" >&2
    echo "  lm   seven-seed matched-state LM run; requires CORPUS" >&2
    echo "  lm10m  five-seed 10M fixed-budget diagnostic; requires CORPUS" >&2
    echo "  lm32m  one-seed 32M fixed-budget diagnostic; requires CORPUS" >&2
    echo "  throughput  reference full-model timing protocol" >&2
    exit 2
    ;;
esac
