# Forgetful Attention

Official code and arXiv source for:

**Forgetful Attention: An Auditable Support-Vector Memory for Selective
Retention and Verified Deletion**

[arXiv:2607.12204](https://arxiv.org/abs/2607.12204)

## What this repository verifies

- algebraic reserve removal preserves the currently solved readout, with
  floating-point residuals checked against explicit tolerances;
- current reserve status alone does not guarantee future-history equivalence;
- completed maintained deletion is compared with a retained-key refit under the
  same fixed box parameter `C`;
- verification uses a maintained fp64 path, while training uses a separate
  batched approximation;
- selection is evaluated under declared redundancy and atypicality regimes;
- attempted/completed coverage and numerical deviations are reported together.

## Quick verification

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
bash reproducibility/run_quick.sh
```

The quick tier reconstructs the future-admission counterexample, checks the
fixed-\(C\) solver path, validates the held-out-vital helper, and recomputes the
released aggregate statistics.

For data-free gradient checks, latency, selection, the negative control, and
the four submitted figures:

```bash
python -m pip install -r requirements-full.txt
bash reproducibility/run_headline.sh --quick
```

Use `--paper` for the reported deletion-latency trial counts. Timing remains
hardware-sensitive. Install `requirements-apple.txt` to include MLX-specific
tests; they skip when MLX is unavailable.

## Optional result tiers

- Language modeling and trained-key audits require Apple silicon plus
  `requirements-apple.txt`; see `reproducibility/run_models.sh`.
- MIMIC-IV analyses require an independently credentialed MIMIC-IV v3.1 ICU
  download; see `reproducibility/run_mimic.sh`.
- The 10M/32M language-model diagnostics are multi-hour fixed-step runs, not
  matched-convergence studies.

The complete claim-to-command map is
`reproducibility/PROVENANCE.md`.

## Layout

- `cp_svm/` — maintained one-class solver and batch-QP references;
- `svattn/` — differentiable, causal, and batched SV-Attention paths;
- `experiments/` — only entry points used by the final paper;
- `clinical_seq/` — credentialed MIMIC preprocessing and aggregate analyses;
- `tests/` — focused contract and gradient tests;
- `paper/` — self-contained manuscript, aggregate evidence, and figure code;
- `reproducibility/` — commands, provenance, hardware notes, and aggregates.

## Build the paper

Install the external `tectonic` executable, then run:

```bash
python paper/make_v2_figures.py
cd paper
tectonic main.tex
```

The four submitted figures are tracked, so TeX compilation does not require
rerunning experiments.

## Data and release boundaries

No MIMIC record, identifier, per-stay result, or derived cache is distributed.
The released MIMIC JSON files contain cohort-level aggregates only. Corpora,
model checkpoints, logs, generated outputs, and the legacy MATLAB reference
implementation are also excluded.

Code is Apache-2.0. Bundled LaTeX styles and external dependencies retain their
own licenses; see `THIRD_PARTY_NOTICES.md`.

`MANIFEST.sha256` records the public snapshot contents and is regenerated with
`python reproducibility/build_manifest.py`.
