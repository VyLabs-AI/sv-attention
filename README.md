# Support-vector memory: auditable deletion

Code and numerical evidence for the current version of **What a Deletion Certificate Covers, and Where It Expires: Auditable Removal from a Support-Vector Memory**. [Read the paper on arXiv](https://arxiv.org/abs/2607.12204).

The memory fits a support-vector boundary around stored keys and uses their nonnegative coefficients to weight values. An exactly zero coefficient permits removal from the current readout; future admissions can invalidate that guarantee. Active-key deletion targets a retained-key refit at the original fixed coefficient cap. Numerical agreement and retained-value readout differences are reported separately.

## Reproduce the results

Use Python 3.11 in an isolated environment. From the repository root:

```sh
python -m pip install -r requirements-core.txt
bash reproducibility/run_quick.sh
```

This checks the future-admission counterexample, fixed-cap decrement/refit behavior and clinical-selection helpers using synthetic fixtures, then recomputes saved aggregate statistics. It requires no private dataset or model weights.

For the sequential audit, follow the [evidence verification instructions](journal_evidence/README.md). The bundle includes both the initial audit and the guarded procedure, their exact execution code, all measured results and tests. The initial audit completed 66 of 80 trajectories; the guard completed all 5,120 scheduled operations at unchanged acceptance thresholds, with eight initialization rescues and 84 update rescues.

The [reproduction guide](reproducibility/README.md) covers numerical experiments, model runs, the minimum-norm projection comparison and credentialed clinical analysis. The [result-to-command map](reproducibility/PROVENANCE.md) identifies required inputs and remaining reproduction limits. The measured study bundles retain the execution code and configuration associated with their results.

## Contents

- `cp_svm/` and `svattn/`: one-class solvers, reference QP, differentiable and causal attention implementations.
- `experiments/` and `clinical_seq/`: numerical, trained-key, language-model and clinical experiment entry points.
- `tests/`: solver, gradient, causality and synthetic clinical-helper checks.
- `reproducibility/`: commands, dependency and hardware records, aggregate evidence and file verification.
- `journal_evidence/`: sequential-audit code, protocols, numerical results and verification, packaged as one immutable archive.

## Data and licenses

Model experiments require the specified hardware and independently obtained corpora and checkpoints. MIMIC-IV analyses require credentialed access to the original dataset. Clinical evidence here contains cohort-level aggregates; the sequential audit uses synthetic solver inputs and numerical state records. Access to datasets and models remains subject to their distributors' terms. See the [hardware and environment notes](reproducibility/HARDWARE.md).

Original code is Apache-2.0. Third-party dependencies and the evidence bundle retain their existing notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). `MANIFEST.sha256` records the repository's release files and is regenerated with `python reproducibility/build_manifest.py`.
