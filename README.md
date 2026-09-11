# Support-vector memory: auditable deletion

Reproduction code and numerical evidence for **What a Deletion Certificate Covers, and Where It Expires: Auditable Removal from a Support-Vector Memory**. [Paper and updates on arXiv](https://arxiv.org/abs/2607.12204).

The memory fits a support-vector boundary around stored keys and uses their nonnegative coefficients to weight values. An exactly zero coefficient permits removal from the current readout; future admissions can invalidate that guarantee. Active-key deletion targets a retained-key refit at the original fixed coefficient cap. Numerical agreement and retained-value readout differences are reported separately.

The completed sequential audit preserves the original failures: 66 of 80 trajectories completed. A separately checked guard completed all 5,120 scheduled operations at unchanged acceptance thresholds, with eight initialization rescues and 84 update rescues. These finite synthetic results support explicit acceptance and rescue rules, without establishing general future-admission safety or a production speed advantage.

## Quick numerical checks

Use Python 3.11 and install `requirements-core.txt` in an isolated environment, then run:

```sh
bash reproducibility/run_quick.sh
```

This tests the future-admission counterexample, fixed-cap decrement/refit behavior and clinical-selection helpers using synthetic fixtures, then recomputes saved aggregate statistics. It requires no private dataset or model weights.

The completed sequential study is distributed as an immutable source-free ZIP in [`journal_evidence/`](journal_evidence/). Follow its instructions to extract, verify all file hashes, recompute guarded acceptance summaries and run small failure-injection tests. The archive contains the exact original and guarded runtime snapshots; execute from that extraction so its code is not mixed with the earlier root implementation.

## Reproduction scope

- `cp_svm/` and `svattn/`: maintained one-class solver, reference QP, differentiable and causal attention implementations.
- `experiments/` and `clinical_seq/`: numerical, trained-key, language-model and credentialed clinical reproduction entry points.
- `tests/`: solver, gradient, causality and synthetic clinical-helper checks.
- `reproducibility/`: dependency tiers, original commands, hardware notes, aggregate evidence and release provenance.
- `journal_evidence/`: complete source-free sequential audit supplement and extraction instructions.

See [the reproduction guide](reproducibility/README.md) and [September 11 provenance](reproducibility/RELEASE_20260911.md) for the optional tiers and minimum-norm projection protocol. Model experiments require their specified hardware and independently obtained corpora/checkpoints. MIMIC-IV analyses require credentialed access to the original dataset. Historical `H2O` aggregate labels denote the attention-free proxy evaluated here.

This repository holds the material needed to inspect and reproduce results. The evolving manuscript is maintained on arXiv. The original six Git commits are preserved; their earlier manuscript assets are absent from the current tip.

## Data and licenses

No MIMIC record, identifier, per-stay result, clinical cache, model checkpoint, corpus, credentials or private account files are distributed. Clinical files contain cohort-level aggregates; the sequential audit uses synthetic solver inputs and numerical state records. Access to original datasets and models remains subject to their distributors' terms.

Original code is Apache-2.0. Third-party dependencies and the immutable supplement retain their existing notices; see `THIRD_PARTY_NOTICES.md`. `MANIFEST.sha256` records the explicitly allowed current release files and is regenerated with `python reproducibility/build_manifest.py`.
