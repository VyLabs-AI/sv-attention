# Completed sequential audit evidence

`sv_attention_sequential_evidence_v1.zip` contains the sequential-audit evidence for the current paper: both the initial audit and the guarded procedure, all measured numerical results (including failures), their exact execution code, protocols, analysis and verification code, focused scientific tests and licenses. It is stored as one immutable archive to avoid duplicating its large per-operation numerical records in Git. The large records are synthetic solver results, not personal or clinical data.

From the repository root, extract into an ignored disposable output directory:

```sh
python -m zipfile -e journal_evidence/sv_attention_sequential_evidence_v1.zip outputs/journal_review
cd outputs/journal_review/sv_attention_sequential_evidence_v1
python verify_bundle.py
python -m journal_studies.sv_attention.verify_guarded_results
python -m pytest -q tests/test_sv_journal_study.py -p no:cacheprovider
```

Run the manifest verifier before the other commands. The guarded verifier rewrites derived summaries in the disposable extraction; it does not change raw results or rerun a solver. Tests use small synthetic cases and injected failures; they do not repeat the full experiment grid. Set `PYTHONDONTWRITEBYTECODE=1` if rerunning the manifest verifier after imports. NumPy, SciPy, CVXPY and pytest are required by the scientific checks, with measured environment versions recorded in the supplement README and original JSON provenance.

## Rerun the numerical experiments

From the extracted directory, use new output filenames to preserve the supplied
measurements. The recorded environment includes Python 3.11, NumPy 2.4.6,
SciPy 1.17.1, CVXPY 1.9.2 and the CLARABEL QP solver.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m journal_studies.sv_attention.run_study --output original_reproduction.json
python -m journal_studies.sv_attention.run_guarded_study --output guarded_reproduction.json
```

The guarded runner binds to the included initial protocol and result hash; keep
that supplied parent result in place. The extracted README describes the full
protocol, analysis helpers and environment requirements. Numerical results and
timings can vary across solver and platform versions.

The original audit completed 66/80 trajectories, attempting 4,390/5,120 scheduled operations. The separately frozen guarded follow-up completed all 5,120 operations with eight initialization rescues and 84 update rescues at unchanged numerical acceptance thresholds. These are finite synthetic results with shared-backend QP comparisons, not an exact-arithmetic guarantee or demonstrated production speedup.

Run from the extraction so Python uses the code measured in this study. Its `cp_svm` package includes dependencies specific to the audit. Recorded absolute source paths identify experiment provenance and are not required local paths. The archive preserves the original study records and hashes; its packaging notes do not change the public availability of this repository.
