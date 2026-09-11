# Completed sequential audit evidence

`sv_attention_sequential_evidence_v1.zip` is the exact source-free reviewer supplement accompanying the September 11, 2026 Machine Learning manuscript. It contains both frozen audits, all measured numerical results (including failures), the original and guarded runtime snapshots, protocols, analysis/verification code, focused scientific contract tests and existing licenses. It is stored as one immutable archive to avoid duplicating its large per-operation numerical records in Git. The large records are synthetic solver results, not personal or clinical data.

From the repository root, extract into an ignored disposable output directory:

```sh
python -m zipfile -e journal_evidence/sv_attention_sequential_evidence_v1.zip outputs/journal_review
cd outputs/journal_review/sv_attention_sequential_evidence_v1
python verify_bundle.py
python -m journal_studies.sv_attention.verify_guarded_results
python -m pytest -q tests/test_sv_journal_study.py -p no:cacheprovider
```

Run the manifest verifier before the other commands. The guarded verifier rewrites derived summaries in the disposable extraction; it does not change raw results or rerun a solver. Tests use small synthetic cases and injected failures; they do not repeat the full experiment grid. Set `PYTHONDONTWRITEBYTECODE=1` if rerunning the manifest verifier after imports. NumPy, SciPy, CVXPY and pytest are required by the scientific checks, with measured environment versions recorded in the supplement README and original JSON provenance.

The original audit completed 66/80 trajectories, attempting 4,390/5,120 scheduled operations. The separately frozen guarded follow-up completed all 5,120 operations with eight initialization rescues and 84 update rescues at unchanged numerical acceptance thresholds. These are finite synthetic results with shared-backend QP comparisons, not an exact-arithmetic guarantee or demonstrated production speedup.

The ZIP's README describes its packaging-time status; this repository supplies the code/evidence location named by the journal manuscript. The complete supplement is accessible once the author makes this repository public. Its historical absolute source paths are provenance strings, not live path dependencies. Do not overlay its `cp_svm` package onto the legacy repository root: run from the extraction to use the exact measured snapshot.
