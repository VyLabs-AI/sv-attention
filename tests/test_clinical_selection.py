import numpy as np
import pytest

from clinical_seq.e2_selection import (
    _trial_with_status,
    held_out_vital_label,
    paired_bootstrap,
)


@pytest.mark.parametrize(
    ("vital", "column", "values", "expected"),
    [
        ("hr", 0, [39.0, 80.0, 131.0], [True, False, True]),
        ("sbp", 1, [89.0, 120.0, 201.0], [True, False, True]),
        ("rr", 3, [5.0, 18.0, 31.0], [True, False, True]),
        ("spo2", 4, [89.0, 90.0, 98.0], [True, False, False]),
        ("temp_c", 5, [34.9, 37.0, 38.6], [True, False, True]),
    ],
)
def test_held_out_vital_label_matches_declared_thresholds(
    vital,
    column,
    values,
    expected,
):
    raw = np.zeros((3, 6), dtype=np.float64)
    raw[:, column] = values
    assert held_out_vital_label(raw, vital).tolist() == expected


def test_trial_status_separates_no_event_from_solver_coverage():
    result, status = _trial_with_status(
        np.zeros((4, 5), dtype=np.float64),
        np.zeros(4, dtype=bool),
        nu=0.3,
        kpar=3.0,
    )
    assert result is None
    assert status == "no_event"


def test_paired_bootstrap_records_stay_level_scope():
    rows = [
        {"svdd_retain": 0.8, "h2o_retain": 0.2},
        {"svdd_retain": 0.6, "h2o_retain": 0.4},
    ]
    result = paired_bootstrap(
        rows,
        "svdd_retain",
        "h2o_retain",
        samples=1000,
        seed=7,
    )
    assert result["mean"] == pytest.approx(0.4)
    assert result["resampling_unit"] == "stay"
    assert result["independent_unit_assumption"] is True
    assert result["bootstrap_95_ci"][0] <= result["mean"]
    assert result["bootstrap_95_ci"][1] >= result["mean"]
