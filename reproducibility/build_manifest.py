"""Validate and hash the exact public GitHub snapshot allowlist."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"

PUBLIC_FILES = {
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements-apple.txt",
    "requirements-core.txt",
    "requirements-full.txt",
    "cp_svm/__init__.py",
    "cp_svm/kernels.py",
    "cp_svm/oneclass_fast.py",
    "cp_svm/oneclass_incremental.py",
    "cp_svm/oneclass_qp.py",
    "svattn/__init__.py",
    "svattn/baselines.py",
    "svattn/causal_sv_attention.py",
    "svattn/diff_svdd.py",
    "svattn/eviction_benchmark.py",
    "svattn/fast_diff_svdd.py",
    "svattn/figstyle.py",
    "svattn/mlx_svdd.py",
    "svattn/recall.py",
    "svattn/run_recall.py",
    "svattn/sv_attention.py",
    "svattn/nanogpt/__init__.py",
    "svattn/nanogpt/data.py",
    "svattn/nanogpt/model.py",
    "svattn/tasks/__init__.py",
    "svattn/tasks/mimic.py",
    "svattn/tasks/mqar.py",
    "experiments/__init__.py",
    "experiments/a2_quality.py",
    "experiments/deletion_latency.py",
    "experiments/forget_qa_nlp.py",
    "experiments/forgetting_rigor.py",
    "experiments/g1_throughput.py",
    "experiments/interpretable_demos.py",
    "experiments/p2_mimic.py",
    "clinical_seq/__init__.py",
    "clinical_seq/build_icu_sequences.py",
    "clinical_seq/e2_selection.py",
    "clinical_seq/e3_forget.py",
    "tests/test_causal_layer.py",
    "tests/test_clinical_deletion.py",
    "tests/test_clinical_selection.py",
    "tests/test_diff_svdd.py",
    "tests/test_fast_diff_svdd.py",
    "tests/test_fast_solver.py",
    "tests/test_train_toy.py",
    "paper/appendix.tex",
    "paper/body.tex",
    "paper/data/v2_evidence.json",
    "paper/fancyhdr.sty",
    "paper/figs/deletion_evidence_v2.png",
    "paper/figs/hero_contract.png",
    "paper/figs/selection_scope_v2.png",
    "paper/figs/two_numerical_paths.png",
    "paper/main.tex",
    "paper/make_v2_figures.py",
    "paper/references.bib",
    "paper/tmlr.bst",
    "paper/tmlr.sty",
    "reproducibility/HARDWARE.md",
    "reproducibility/PROVENANCE.md",
    "reproducibility/README.md",
    "reproducibility/build_manifest.py",
    "reproducibility/check_evidence.py",
    "reproducibility/deletion_latency_v2.json",
    "reproducibility/mimic_composite_selection.json",
    "reproducibility/mimic_held_out_spo2.json",
    "reproducibility/run_headline.sh",
    "reproducibility/run_mimic.sh",
    "reproducibility/run_models.sh",
    "reproducibility/run_quick.sh",
}

LOCAL_ONLY_PREFIXES = {
    ".git/",
    ".pytest_cache/",
    ".venv/",
    "clinical_seq/cache/",
    "outputs/",
    "releases/",
}
LOCAL_ONLY_NAMES = {
    ".DS_Store",
    "MANIFEST.sha256",
    "paper/main.aux",
    "paper/main.bbl",
    "paper/main.blg",
    "paper/main.log",
    "paper/main.out",
    "paper/main.pdf",
}
BANNED_TEXT = {
    "/" + "Users/",
    "BEGIN " + "OPENSSH PRIVATE KEY",
    "BEGIN " + "PRIVATE KEY",
    "api_" + "key=",
}


def _local_only(relative: str) -> bool:
    if relative in LOCAL_ONLY_NAMES:
        return True
    if "__pycache__/" in relative or relative.endswith((".pyc", ".pyo")):
        return True
    return any(relative.startswith(prefix) for prefix in LOCAL_ONLY_PREFIXES)


def _validate_text(path: Path) -> None:
    if path.suffix.lower() in {".png", ".pdf"}:
        return
    text = path.read_text()
    hits = sorted(token for token in BANNED_TEXT if token in text)
    if hits:
        raise RuntimeError(f"private or secret-like text in {path}: {hits}")


def main() -> None:
    missing = sorted(relative for relative in PUBLIC_FILES if not (ROOT / relative).is_file())
    if missing:
        raise RuntimeError(f"public allowlist files are missing: {missing}")

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    unexpected = sorted(
        relative
        for relative in actual - PUBLIC_FILES
        if not _local_only(relative)
    )
    if unexpected:
        raise RuntimeError(f"undeclared public-snapshot files: {unexpected}")

    rows = []
    for relative in sorted(PUBLIC_FILES):
        path = ROOT / relative
        if path.is_symlink():
            raise RuntimeError(f"symlinks are not allowed: {relative}")
        _validate_text(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {relative}")

    MANIFEST.write_text("\n".join(rows) + "\n")
    print(f"Wrote {MANIFEST} with {len(rows)} allowlisted files")


if __name__ == "__main__":
    main()
