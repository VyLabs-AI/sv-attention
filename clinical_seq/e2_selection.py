"""E2 -- Fixed-context support selection on real ICU streams.

When the KV budget is smaller than the stream, something must be dropped. We ask
the clinically pointed question: at a MATCHED token budget, does each policy keep
the rare deterioration hours (the moments a clinician must not lose), or does it
spend the budget on the redundant stable hours?

All policies select the SAME number of tokens k (set by the point-in-time
support set so the comparison is budget-matched), and we score them two ways:
  - rare-hour retention: fraction of deterioration-labeled hours kept;
  - rare-hour retrieval : probe each retained set with the rare hours' own keys
    under uniform RBF attention and check the nearest retained token is itself a
    deterioration hour (does the kept context still surface the rare event?).

Policies:
  - svdd     : one-class support set (reserve is inert for the fixed solve),
               query-independent.
  - h2o      : H2O-style heavy hitters -- top-k tokens by accumulated self-
               attention mass (the dense, redundant hours).
  - recency  : last k hours.   - random : random k.

Deterioration hours are defined by physiologic thresholds at build time
(clinical_seq.build_icu_sequences.deterioration_label).

The optional held-out-vital control defines events using one vital and removes
that channel from every selection policy's input. It tests whether correlated
physiology still places those events near the support boundary without giving
the gate the label-defining measurement itself.

Run from the repository root:
    python -m clinical_seq.e2_selection
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from cp_svm import FastOneClassSVM
from cp_svm.kernels import radial_kernel

CACHE = Path(__file__).resolve().parent / "cache"

HELD_OUT_VITALS = ("hr", "sbp", "rr", "spo2", "temp_c")


def load(path):
    d = np.load(path, allow_pickle=True)
    return list(d["sequences"]), list(d["det_labels"]), d["stay_ids"]


def held_out_vital_label(raw, vital):
    """Threshold one cached unstandardized vital using the declared rule."""
    raw = np.asarray(raw, dtype=np.float64)
    column = {"hr": 0, "sbp": 1, "rr": 3, "spo2": 4, "temp_c": 5}[vital]
    value = raw[:, column]
    if vital == "hr":
        return (value > 130) | (value < 40)
    if vital == "sbp":
        return (value < 90) | (value > 200)
    if vital == "rr":
        return (value > 30) | (value < 6)
    if vital == "spo2":
        return value < 90
    return (value > 38.5) | (value < 35.0)


def load_held_out(path, vital):
    """Load a single-vital event label and remove that vital from gate inputs."""
    d = np.load(path, allow_pickle=True)
    features = [str(name) for name in d["features"]]
    held_out = features.index(vital)
    keep = [index for index in range(len(features)) if index != held_out]
    seqs = [np.asarray(sequence)[:, keep] for sequence in d["sequences"]]
    dets = [
        held_out_vital_label(raw, vital)
        for raw in d["raw_sequences"]
    ]
    return seqs, dets, d["stay_ids"]


def heavy_hitter_topk(X, k, kpar):
    """H2O-style: top-k tokens by total self-attention mass (RBF)."""
    Kmat = radial_kernel(X, X, kpar)
    np.fill_diagonal(Kmat, 0.0)
    mass = Kmat.sum(axis=0)
    return np.sort(np.argsort(-mass)[:k])


def rare_retrieval(sel, X, det, kpar):
    """For each rare hour, the nearest token within the retained set -- is it
    also a rare hour? Uniform RBF attention argmax over retained tokens."""
    rare_idx = np.where(det)[0]
    if len(rare_idx) == 0 or len(sel) == 0:
        return np.nan
    Q = X[rare_idx]
    w = radial_kernel(Q, X[sel], kpar)
    nearest = sel[w.argmax(axis=1)]
    hits = np.array([det[int(j)] for j in nearest])
    return float(hits.mean())


def _trial_with_status(X, det, nu, kpar):
    n = len(X)
    if det.sum() == 0:
        return None, "no_event"
    C = 1.0 / (nu * n)
    try:
        g = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X)
    except Exception as error:
        return None, f"solver:{type(error).__name__}"
    support = np.sort(np.array(g.S + g.E, dtype=int))
    k = len(support)
    if k == 0 or k >= n:
        return None, "support_size_boundary"

    rng = np.random.RandomState(n)  # deterministic per stay
    sels = {
        "svdd": support,
        "h2o": heavy_hitter_topk(X, k, kpar),
        "recency": np.arange(n - k, n),
        "random": np.sort(rng.choice(n, k, replace=False)),
    }
    rare = det.astype(bool)
    out = {"budget": k / n, "n_rare": int(rare.sum()), "n": n}
    for name, sel in sels.items():
        keep = np.zeros(n, bool); keep[sel] = True
        out[f"{name}_retain"] = float(keep[rare].mean())
        out[f"{name}_retrieval"] = rare_retrieval(sel, X, det, kpar)
    return out, "complete"


def trial(X, det, nu, kpar):
    """Run one stay; retained for callers that only need complete rows."""
    result, _status = _trial_with_status(X, det, nu, kpar)
    return result


def paired_bootstrap(rows, left, right, samples=10_000, seed=0):
    """Return a deterministic paired mean difference and percentile interval."""
    differences = np.asarray(
        [row[left] - row[right] for row in rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        differences,
        size=(samples, len(differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_samples": samples,
        "seed": seed,
        "method": "paired percentile bootstrap",
        "resampling_unit": "stay",
        "independent_unit_assumption": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE / "icu_vitals_n1500.npz"))
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--kpar", type=float, default=3.0)
    ap.add_argument("--max-stays", type=int, default=400)
    ap.add_argument(
        "--held-out-vital",
        choices=HELD_OUT_VITALS,
        default=None,
        help="define events from this unstandardized cached vital and exclude it",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="optional path for a machine-readable aggregate JSON report",
    )
    args = ap.parse_args()

    if args.held_out_vital:
        seqs, dets, ids = load_held_out(args.cache, args.held_out_vital)
    else:
        seqs, dets, ids = load(args.cache)
    rows = []
    statuses = Counter()
    for X, det in zip(seqs[:args.max_stays], dets[:args.max_stays]):
        r, status = _trial_with_status(
            X,
            np.asarray(det),
            args.nu,
            args.kpar,
        )
        statuses[status] += 1
        if r:
            rows.append(r)
    if not rows:
        print("no valid stays"); return

    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    retention_contrast = paired_bootstrap(
        rows,
        "svdd_retain",
        "h2o_retain",
    )
    mode = (
        f"held-out {args.held_out_vital} threshold "
        f"({args.held_out_vital} excluded from policy inputs)"
        if args.held_out_vital
        else "five-threshold composite label on six-channel policy inputs"
    )
    print(f"Matched-budget selection on real ICU streams "
          f"({len(rows)} stays with >=1 deterioration hour, nu={args.nu}, "
          f"RBF kpar={args.kpar})")
    print(f"  mode: {mode}")
    event_attempts = sum(
        count for status, count in statuses.items() if status != "no_event"
    )
    print(f"  event-positive coverage: {len(rows)}/{event_attempts}; "
          f"statuses={dict(sorted(statuses.items()))}")
    print(f"  mean budget {agg['budget']*100:.0f}% of stream, "
          f"mean {agg['n_rare']:.1f} rare hours / stay\n")
    print(f"  {'policy':<12}{'rare-hour retention':>22}{'rare-hour retrieval':>22}")
    for name in ["svdd", "h2o", "recency", "random"]:
        print(f"  {name:<12}{agg[f'{name}_retain']:>22.3f}"
              f"{agg[f'{name}_retrieval']:>22.3f}")
    lo, hi = retention_contrast["bootstrap_95_ci"]
    print(
        f"\n  paired SVDD-H2O retention difference "
        f"{retention_contrast['mean']:.3f} "
        f"(95% stay-level percentile bootstrap CI [{lo:.3f}, {hi:.3f}])"
    )
    print("\nAll policies keep the same #tokens (budget-matched to the point-in-time "
          "support set). Retention = fraction of deterioration hours kept; "
          "retrieval = a rare hour's nearest retained token is also rare.")
    if args.report:
        report = {
            "mode": "held_out_vital" if args.held_out_vital else "composite",
            "held_out_vital": args.held_out_vital,
            "max_stays": args.max_stays,
            "event_positive_attempts": event_attempts,
            "completed_stays": len(rows),
            "status_counts": dict(sorted(statuses.items())),
            "nu": args.nu,
            "kernel": "RBF",
            "kernel_width": args.kpar,
            "aggregate": agg,
            "svdd_minus_h2o_retention": retention_contrast,
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
