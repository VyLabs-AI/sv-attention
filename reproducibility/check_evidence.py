"""Recompute statistics stored in the arXiv v2 aggregate evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reproducibility" / "aggregates" / "v2_evidence.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_language_modeling(evidence: dict) -> None:
    language_modeling = evidence["language_modeling"]
    row = language_modeling["enwik8_3.22m"]
    assert row["prefix_bytes"] == 5_000_000
    assert row["evaluation_every_steps"] == {
        "seed_0": 500,
        "seeds_1_to_6": 1000,
    }
    assert language_modeling["enwik8_10m"]["prefix_bytes"] == 20_000_000
    assert language_modeling["enwik8_32m"]["prefix_bytes"] == 20_000_000

    wrapper = (ROOT / "reproducibility" / "run_models.sh").read_text()
    assert "--n_bytes 5000000" in wrapper
    assert wrapper.count("--n_bytes 20000000") == 2
    assert '--eval_every "$EVAL_EVERY"' in wrapper

    sv = np.asarray(row["sv"], dtype=np.float64)
    swa = np.asarray(row["swa"], dtype=np.float64)
    relative = (swa - sv) / swa
    differences = swa - sv

    t_result = stats.ttest_rel(swa, sv)
    relative_interval = stats.t.interval(
        0.95,
        len(relative) - 1,
        loc=relative.mean(),
        scale=stats.sem(relative),
    )
    wilcoxon = stats.wilcoxon(swa, sv)
    cohen_dz = differences.mean() / differences.std(ddof=1)

    np.testing.assert_allclose(row["mean_sv"], sv.mean())
    np.testing.assert_allclose(row["mean_swa"], swa.mean())
    np.testing.assert_allclose(
        row["mean_paired_relative_improvement"],
        relative.mean(),
    )
    np.testing.assert_allclose(
        row["relative_improvement_t_interval_95"],
        relative_interval,
    )
    np.testing.assert_allclose(
        row["absolute_difference_paired_t"],
        t_result.statistic,
    )
    np.testing.assert_allclose(
        row["absolute_difference_paired_t_p"],
        t_result.pvalue,
    )
    np.testing.assert_allclose(row["wilcoxon_p"], wilcoxon.pvalue)
    np.testing.assert_allclose(row["cohen_dz"], cohen_dz)


def _check_clinical(evidence: dict) -> None:
    held_out = _load(ROOT / "reproducibility" / "mimic_held_out_spo2.json")
    expected = evidence["selection"]["mimic_held_out_spo2_retention"]
    aggregate = held_out["aggregate"]
    np.testing.assert_allclose(expected["mean_budget"], aggregate["budget"])
    np.testing.assert_allclose(expected["SV gate"], aggregate["svdd_retain"])
    np.testing.assert_allclose(expected["H2O"], aggregate["h2o_retain"])
    np.testing.assert_allclose(expected["Random"], aggregate["random_retain"])
    np.testing.assert_allclose(expected["Recency"], aggregate["recency_retain"])
    contrast = held_out["svdd_minus_h2o_retention"]
    np.testing.assert_allclose(expected["sv_minus_h2o"], contrast["mean"])
    np.testing.assert_allclose(
        expected["bootstrap_95_ci"],
        contrast["bootstrap_95_ci"],
    )
    assert expected["analysis_stays"] == held_out["completed_stays"]
    assert held_out["event_positive_attempts"] == held_out["completed_stays"]

    composite = _load(ROOT / "reproducibility" / "mimic_composite_selection.json")
    submitted = evidence["selection"]["mimic_deterioration_retention"]
    np.testing.assert_allclose(
        submitted["SV gate"],
        composite["aggregate"]["svdd_retain"],
        atol=0.005,
    )
    np.testing.assert_allclose(
        submitted["H2O"],
        composite["aggregate"]["h2o_retain"],
        atol=0.005,
    )


def _check_deletion_latency(evidence: dict) -> None:
    report = _load(ROOT / "reproducibility" / "deletion_latency_v2.json")
    expected = evidence["deletion_latency"]["summaries"]
    assert len(expected) == len(report["summaries"])
    for stored, reproduced in zip(expected, report["summaries"]):
        assert stored["n"] == reproduced["n"]
        for key in (
            "median_deletion_ms",
            "median_refit_ms",
            "median_per_trial_speedup",
        ):
            np.testing.assert_allclose(stored[key], reproduced[key])


def main() -> None:
    evidence = _load(EVIDENCE)
    _check_language_modeling(evidence)
    _check_clinical(evidence)
    _check_deletion_latency(evidence)
    print("Aggregate evidence checks passed.")


if __name__ == "__main__":
    main()
