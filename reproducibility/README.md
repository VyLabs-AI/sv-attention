# Reproducibility guide

These tiers reproduce the existing solver, trained-key and selection results. The completed sequential audit has an independent frozen runtime; see [RELEASE_20260911.md](RELEASE_20260911.md) and [the journal evidence instructions](../journal_evidence/README.md).

Run commands from the repository root. Generated files go to `outputs/`;
credentialed clinical preprocessing writes to `clinical_seq/cache/`. Both are
ignored and excluded from releases.

## Dependency tiers

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install -r requirements-core.txt   # CPU solver and evidence
python -m pip install -r requirements-full.txt   # + Torch, plots, clinical
python -m pip install -r requirements-apple.txt  # + MLX on Apple silicon
```

## Tier 1: quick CPU verification

```bash
bash reproducibility/run_quick.sh
```

This reconstructs the named future-admission counterexample, checks fixed-\(C\)
decrement/refit behavior, tests held-out-vital labeling and bootstrap metadata,
and recomputes the released aggregate statistics.

## Tier 2: data-free headline checks

```bash
bash reproducibility/run_headline.sh --quick
```

The quick path adds maintained/batched gradient checks, smoke-sized deletion
latency, skewed-redundancy selection, the distinct-key negative control, and the distinct-key negative control. MLX-specific checks skip unless
`requirements-apple.txt` is installed.

Use the reported deletion timing protocol with:

```bash
bash reproducibility/run_headline.sh --paper
```

Wall time is hardware-sensitive; the checked reference summary is
`reproducibility/deletion_latency_v2.json`.

## Tier 3: trained keys and language modeling

These paths require Apple silicon and `requirements-apple.txt`.

Train the small recall model and run frozen-key deletion/editing:

```bash
bash reproducibility/run_models.sh qa
```

The 3.22M enwik8 result uses the first 5M bytes, seed 0 evaluation every 500
steps, and seeds 1--6 every 1,000 steps:

```bash
export CORPUS=/path/to/enwik8
export MODEL_SEEDS="0 1 2 3 4 5 6"
export MODEL_STEPS=6000
bash reproducibility/run_models.sh lm
```

For TinyStories, point `CORPUS` at a plain-text corpus and set
`MODEL_SEEDS="0 1 2"`.

The fixed-step larger diagnostics use the first 20M enwik8 bytes:

```bash
CORPUS=/path/to/enwik8 bash reproducibility/run_models.sh lm10m
CORPUS=/path/to/enwik8 bash reproducibility/run_models.sh lm32m
```

These are multi-hour matched-step, not matched-convergence, comparisons.

The reference throughput entry point is:

```bash
bash reproducibility/run_models.sh throughput
```

## Tier 4: credentialed MIMIC-IV

No MIMIC source record, identifier, cache, or per-stay output is distributed.
Credentialed users must obtain MIMIC-IV v3.1 independently from PhysioNet.

```bash
bash reproducibility/run_mimic.sh prepare \
  --icu-path /path/to/mimiciv/3.1/icu
bash reproducibility/run_mimic.sh analyze
```

The reference flow is:

```text
1,500 sampled candidates
  -> 1,465 eligible hourly sequences
  -> 1,329 composite-event stays
  ->   718 held-out-SpO2-event stays
```

The held-out control defines events from unstandardized hourly SpO2 below 90
after within-stay filling and cohort-median imputation, then removes SpO2 from
all selector inputs.

The full four-regime deletion audit additionally needs the trained QA checkpoint
from Tier 3:

```bash
bash reproducibility/run_mimic.sh audit
```

## Stored numerical evidence

`aggregates/v2_evidence.json` preserves the historical aggregates used by `check_evidence.py`; `aggregates/v3_evidence.json` and `aggregates/tiebreak_evidence.json` preserve the current deletion quantiles and projection comparison. Historical `H2O` keys in the clinical aggregate refer to the attention-free score proxy, not a claim of a full H2O implementation.

`check_evidence.py` recomputes paired language-model statistics and cross-checks the latency and clinical aggregate files. See `PROVENANCE.md` for the claim-to-command map and `HARDWARE.md` for timing and determinism boundaries. Manuscripts are maintained on arXiv; this repository contains reproduction material only.
