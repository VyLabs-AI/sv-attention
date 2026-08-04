# Hardware, runtime, and determinism notes

## Reference environment

The reported deletion-latency measurements were collected on an Apple M3 Ultra
CPU. The batched training solver and reported MLX language-model runs used an
Apple-silicon GPU. Exact CPU solver checks are portable across macOS and Linux;
small floating-point differences between BLAS and convex-solver versions are
expected.

The refreshed deletion reference used macOS 26.5, Python 3.11.14, and NumPy
2.4.6. At 30 measured uniform-random removals per context size, the per-size
median deletion latency was 0.30--1.04 ms and the median per-trial
refit/deletion ratio was 24--223x. Reproduce the raw timing report with:

```bash
python -m experiments.deletion_latency \
  --sizes 120 256 384 512 \
  --trials 30 \
  --seed 0 \
  --report outputs/deletion_latency_v2_reference_env.json
```

The hardened full-model throughput reference used an Apple M3 Ultra with 32
physical CPU cores and 512 GiB unified memory, macOS 26.5, Python 3.11.14,
PyTorch 2.12.1, NumPy 2.4.6, and MLX 0.31.2. Run:

```bash
python -m experiments.g1_throughput --paper
```

For the checked run (three warm-ups, seven synchronized repetitions), the
unfused PyTorch softmax reference reached 55,789 tokens/s on CPU and 326,648
tokens/s on MPS. The direct mixed PyTorch-CPU/MLX-GPU hybrid reached 9,125
tokens/s. The script prints the interquartile ranges and full device labels;
these are hardware measurements rather than portable golden values.

Every timing command prints or should be recorded with:

- operating system and architecture;
- CPU/GPU model;
- Python, NumPy, Torch, CVXPY, and MLX versions;
- thread settings and power mode;
- command, seed, warmup count, and measured repetitions.

Do not compare smoke-run timings with paper tables. `deletion_latency.py`
alternates operation order and reports medians, but absolute latency remains a
property of the host.

## Runtime envelope

- `run_quick.sh`: usually 1–5 minutes on a current laptop CPU.
- `run_headline.sh --quick`: usually several minutes to tens of minutes.
- `run_headline.sh --paper`: CPU-heavy; allow one or more hours.
- `run_models.sh qa`: minutes to tens of minutes on Apple silicon with MLX.
- `run_models.sh lm`: hours per paper-scale seed. The full seven-seed sweep is
  an accelerator workload and should be scheduled accordingly.
- MIMIC preparation: one streaming pass over multi-gigabyte compressed
  `chartevents`; storage throughput dominates and the pass can take tens of
  minutes to hours.

These are planning ranges, not performance claims.

## Deterministic and non-deterministic components

Fixed seeds make generated samples, cohort ordering, and model initialization
repeatable. CPU double-precision solver tests use explicit tolerances and are
the strongest deterministic checks.

The following can vary while remaining valid:

- convex-solver vertex choices for non-unique, near-duplicate optima;
- accelerator reduction order and MLX/Torch kernels;
- wall-clock throughput;
- first-use model and corpus downloads;
- PDFs at the byte level when TeX engines or package bundles differ.

`MANIFEST.sha256` records the public snapshot bytes. Regenerate it with
`python reproducibility/build_manifest.py` after intentional source changes.
