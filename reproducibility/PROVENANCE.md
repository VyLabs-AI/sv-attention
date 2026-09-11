# Result provenance map

All paths and commands are relative to this repository root. Unless noted,
scripts set NumPy and Torch seeds internally to 0.

| Paper result or artifact | Source of record | Reproduction command | External input |
|---|---|---|---|
| Aggregate-evidence consistency and paired LM statistics | `reproducibility/check_evidence.py`, `reproducibility/aggregates/v2_evidence.json` | `python reproducibility/check_evidence.py` | None |
| Point-in-time reserve certificate, fixed-\(C\) regression, and future-admission counterexample | `tests/test_fast_solver.py`, `reproducibility/aggregates/v2_evidence.json` | `python -m pytest tests/test_fast_solver.py -q` | None |
| Maintained and batched gradient checks; 200-step teacher fit | `tests/test_diff_svdd.py`, `tests/test_fast_diff_svdd.py`, `tests/test_train_toy.py`, `tests/test_causal_layer.py` | `python -m pytest tests/test_diff_svdd.py tests/test_fast_diff_svdd.py tests/test_train_toy.py tests/test_causal_layer.py -q` | MLX-only checks skip when MLX is unavailable |
| Four-regime fixed-\(C\) decrement/refit coverage and numerical deviations | `experiments/forgetting_rigor.py`, `reproducibility/aggregates/v2_evidence.json` | `bash reproducibility/run_mimic.sh audit` | MIMIC cache and trained QA checkpoint |
| Maintained deletion latency versus retained-set refit (0.30--1.04 ms; 24--223x) | `experiments/deletion_latency.py`, `reproducibility/deletion_latency_v2.json` | `python -m experiments.deletion_latency --sizes 120 256 384 512 --trials 30 --seed 0 --report outputs/deletion_latency_v2_reference_env.json` | None; wall time is hardware-specific |
| Skewed-redundancy matched-budget selection | `svattn/eviction_benchmark.py`, `reproducibility/aggregates/v2_evidence.json` | `python -m svattn.eviction_benchmark` | None |
| Distinct-key gate-off negative control | `svattn/run_recall.py`, `svattn/recall.py`, `reproducibility/aggregates/v2_evidence.json` | `python -m svattn.run_recall` | None |
| Frozen learned-key retained-refit baseline and maintained remove-add editing | `experiments/forget_qa_nlp.py`, `experiments/interpretable_demos.py` | `bash reproducibility/run_models.sh qa` | Trains a local checkpoint; MLX required |
| 3.22M seven-seed enwik8 matched-state result | `experiments/a2_quality.py`, `reproducibility/aggregates/v2_evidence.json` | `CORPUS=/path/to/enwik8 MODEL_SEEDS="0 1 2 3 4 5 6" bash reproducibility/run_models.sh lm` | User-supplied corpus; MLX required |
| 3.22M three-seed TinyStories result | `experiments/a2_quality.py`, `reproducibility/aggregates/v2_evidence.json` | `CORPUS=/path/to/tinystories.txt MODEL_SEEDS="0 1 2" bash reproducibility/run_models.sh lm` | User-supplied corpus; MLX required |
| 10M five-seed fixed-step optimization boundary | `experiments/a2_quality.py`, `reproducibility/aggregates/v2_evidence.json` | `CORPUS=/path/to/enwik8 bash reproducibility/run_models.sh lm10m` | User-supplied corpus; MLX required |
| 32M one-seed fixed-step optimization boundary | `experiments/a2_quality.py`, `reproducibility/aggregates/v2_evidence.json` | `CORPUS=/path/to/enwik8 bash reproducibility/run_models.sh lm32m` | User-supplied corpus; MLX required |
| ICU hourly-sequence cache and cohort accounting | `clinical_seq/build_icu_sequences.py` | `bash reproducibility/run_mimic.sh prepare --icu-path /path/to/mimiciv/3.1/icu` | Credentialed MIMIC-IV v3.1 ICU module |
| ICU composite threshold-hour retention | `clinical_seq/e2_selection.py`, `reproducibility/mimic_composite_selection.json` | `python -m clinical_seq.e2_selection --max-stays 1465 --report outputs/mimic_composite_selection.json` | Local MIMIC cache |
| Held-out-SpO2 retention control and paired bootstrap | `clinical_seq/e2_selection.py`, `reproducibility/mimic_held_out_spo2.json` | `python -m clinical_seq.e2_selection --max-stays 1465 --held-out-vital spo2 --report outputs/mimic_held_out_spo2.json` | Local MIMIC cache |
| Patient-record gate deletion | `clinical_seq/e3_forget.py`, `tests/test_clinical_deletion.py` | `python -m clinical_seq.e3_forget --trials 100` | Local MIMIC cache |
| Next-six-hour ICU prediction control | `experiments/p2_mimic.py` | `python -m experiments.p2_mimic` | Local MIMIC cache |
| Full-model training-step throughput boundary | `experiments/g1_throughput.py`, `reproducibility/HARDWARE.md` | `bash reproducibility/run_models.sh throughput` | Apple M3 Ultra reference; hardware/framework-specific |

## Seed policy

- Solver, selection, clinical, and figure scripts use seed 0 unless a command
  exposes `--seed`.
- `experiments/deletion_latency.py` derives independent trial seeds from its
  `--seed`, context size, and trial index.
- The paper-scale enwik8 comparison uses seeds `0 1 2 3 4 5 6`; TinyStories
  uses seeds `0 1 2`.
- The counterexample test checks the submitted point-in-time attention residual,
  future coefficient, and finite-probe decision deviation.
- Output files are products, not inputs of authority. Scripts, explicit command
  parameters, and aggregate evidence define provenance.

## Clinical cohort accounting

The reference flow is **1,465 eligible -> 1,329 composite-event stays -> 400
prediction-subset stays**. The first count is the valid hourly-sequence cohort
after fixed-seed preprocessing. The second excludes stays without a
threshold-defined deterioration hour for the matched-budget retention analysis.
The third is the fixed subset used by the prediction script.

The held-out-SpO2 control analyzes the 718 eligible stays containing at least
one unstandardized hourly SpO2-below-90 event after within-stay filling and
cohort-median imputation. SpO2 is excluded from all selector inputs. Released
clinical evidence is aggregate-only: no MIMIC record, identifier, cache, or
per-stay result is distributed.

The completed sequential audit and guarded follow-up are documented in `RELEASE_20260911.md` and distributed in `../journal_evidence/`. The minimum-norm projection runner is `experiments/forgetting_tiebreak.py`; its saved aggregate and exact six-tolerance protocol are in `aggregates/tiebreak_evidence.json`.
