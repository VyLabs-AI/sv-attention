import numpy as np

from clinical_seq.e3_forget import one_trial


def test_patient_decrement_matches_fixed_c_retained_refit():
    data_rng = np.random.RandomState(123)
    sequences = [
        data_rng.normal(loc=index * 2.0, scale=0.5, size=(16, 4))
        for index in range(8)
    ]
    row = one_trial(
        sequences,
        np.random.RandomState(0),
        n_patients=3,
        kpar=2.0,
        nu=0.5,
        cap_hours=16,
        dedup_eps=0.0,
    )

    assert row["kkt"] < 1e-7
    assert row["decision_deviation"] < 1e-6
    assert row["readout_deviation"] < 1e-6
    assert row["decay0.1_readout_deviation"] > 1e-2
