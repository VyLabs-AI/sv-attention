"""E3 -- Fixed-C patient-record deletion audit on real ICU streams.

A shared context holds the hourly tokens of several ICU stays. A request comes
in to remove one patient's records. We delete those tokens through the
Cauwenberghs--Poggio reverse path and audit:

  (1) whether a completed decrement agrees with a retained-key refit under the
      same fixed C, together with KKT residuals and explicit coverage; and
  (2) a coefficient-decay baseline on the same finite readout probes.

Smooth, oversampled clinical vitals are numerically harder than the synthetic
well-separated case: a small fraction of patient combinations drive the margin
set empty during the reverse sweep, an edge case the current decrement path does
not cover. We report that coverage honestly and aggregate the agreement metrics
over the trials that complete.

This is a state/decision/readout audit, not deletion from model weights, a
privacy guarantee, or a legal-compliance claim.

Run from the repository root:
    python -m clinical_seq.e3_forget
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from cp_svm import FastOneClassSVM
from cp_svm.kernels import radial_kernel

CACHE = Path(__file__).resolve().parent / "cache"


def load(path):
    d = np.load(path, allow_pickle=True)
    return list(d["sequences"]), d["stay_ids"]


def dedupe(X, eps):
    """Collapse near-identical (forward-filled) hours: keep an hour only if it
    is > eps std-units from every kept hour. This declared numerical
    preprocessing changes the context and reduces duplicate-induced singularity."""
    if eps <= 0 or len(X) <= 1:
        return X
    keep = [0]
    for i in range(1, len(X)):
        if np.min(np.sum((X[keep] - X[i]) ** 2, axis=1)) > eps * eps:
            keep.append(i)
    return X[keep]


def kkt_residual(m: FastOneClassSVM):
    """Maximum equality, box, margin, and set-inequality KKT violation."""
    K, a = m._K, m.alpha
    g = m.rho - np.diag(K) + 2.0 * (K @ a)
    r_margin = float(np.max(np.abs(g[m.S]))) if m.S else 0.0
    r_sum = float(abs(a.sum() - 1.0))
    r_error = max(float(np.max(g[m.E])), 0.0) if m.E else 0.0
    r_reserve = max(float(-np.min(g[m.R])), 0.0) if m.R else 0.0
    r_box = max(
        float(np.max(-a, initial=0.0)),
        float(np.max(a - m.C, initial=0.0)),
    )
    return max(r_margin, r_sum, r_error, r_reserve, r_box)


def readout(alpha, keys, values, Q, kpar):
    w = radial_kernel(Q, keys, kpar) * alpha[None, :]
    denom = w.sum(axis=1, keepdims=True)
    denom[denom < 1e-12] = 1e-12
    return w @ values / denom


def one_trial(seqs, rng, n_patients, kpar, nu, cap_hours, dedup_eps):
    pick = rng.choice(len(seqs), n_patients, replace=False)
    blocks = [dedupe(seqs[i][:cap_hours], dedup_eps) for i in pick]
    sizes = [len(b) for b in blocks]
    X = np.vstack(blocks)
    n = len(X)
    pid = np.concatenate([[j] * s for j, s in enumerate(sizes)])
    # one-hot patient-id values, so the readout reveals whose token answers a probe
    V = np.eye(n_patients)[pid]

    forget_j = int(rng.randint(n_patients))
    fmask = pid == forget_j
    keep_keys = X[~fmask]
    keep_vals = V[~fmask]
    C = 1.0 / (nu * n)

    # full layer over all patients, then forget the target decrementally
    m = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X)
    a_full = np.array(m.alpha)
    forget_rows = np.where(fmask)[0]
    for c in sorted(forget_rows, reverse=True):
        m.remove_point(c)            # may raise on the S-empty edge case

    # Compare with an independent retained-key refit under the identical C.
    refit = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(keep_keys)
    Qf = X[fmask] + 0.02 * rng.randn(int(fmask.sum()), X.shape[1])
    jitter_base = keep_keys[: min(32, len(keep_keys))]
    probes = np.vstack(
        [
            keep_keys,
            Qf,
            jitter_base + 0.02 * rng.randn(*jitter_base.shape),
        ]
    )

    out_refit = readout(refit.alpha, refit.X, keep_vals, probes, kpar)
    out_decrement = readout(m.alpha, m.X, keep_vals, probes, kpar)
    res = {
        "kkt": kkt_residual(m),
        "guards": m.guard_events,
        "decision_deviation": float(
            np.max(
                np.abs(
                    m.decision_function(probes)
                    - refit.decision_function(probes)
                )
            )
        ),
        "readout_deviation": float(np.max(np.abs(out_decrement - out_refit))),
    }
    # Decay keeps the target rows and is evaluated against the same refit.
    for gamma in (0.1, 0.01):
        a_dec = a_full.copy()
        a_dec[fmask] *= gamma
        out_dec = readout(a_dec, X, V, probes, kpar)
        res[f"decay{gamma}_readout_deviation"] = float(
            np.max(np.abs(out_dec - out_refit))
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE / "icu_vitals_n1500.npz"))
    ap.add_argument("--nu", type=float, default=0.5)
    ap.add_argument("--kpar", type=float, default=2.0)
    ap.add_argument("--n-patients", type=int, default=5)
    ap.add_argument("--cap-hours", type=int, default=48)
    ap.add_argument("--dedup-eps", type=float, default=0.1,
                    help="collapse hours within this many std-units (0 disables)")
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    seqs, ids = load(args.cache)
    rng = np.random.RandomState(0)
    rows, failures = [], Counter()
    for _ in range(args.trials):
        try:
            rows.append(one_trial(seqs, rng, args.n_patients, args.kpar,
                                  args.nu, args.cap_hours, args.dedup_eps))
        except RuntimeError as error:
            message = str(error).lower()
            reason = "margin_empty" if "margin" in message and "empty" in message else type(error).__name__
            failures[reason] += 1
    if not rows:
        raise RuntimeError(f"no completed trials; failures={dict(failures)}")
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    worst_kkt = max(r["kkt"] for r in rows)

    print(f"Fixed-C patient-record deletion audit on real ICU streams "
          f"({len(rows)}/{args.trials} trials completed, {args.n_patients} "
          f"patients/context, <= {args.cap_hours}h each, dedup_eps={args.dedup_eps}, "
          f"nu={args.nu}, RBF kpar={args.kpar})")
    print(f"  decrement coverage: {len(rows)}/{args.trials} "
          f"(failures={dict(failures)})\n")
    print("  agreement with an independent retained-key refit:")
    print(f"    mean decision-function deviation {agg['decision_deviation']:.2e}")
    print(f"    mean readout deviation           {agg['readout_deviation']:.2e}")
    print(f"    mean KKT residual {agg['kkt']:.2e}   (worst {worst_kkt:.2e})")
    print(f"    mean refactor guard events / trial: {agg['guards']:.2f}\n")
    print("  decay readout deviation from the same retained-key refit:")
    for g in (0.1, 0.01):
        key = f"decay{g}_readout_deviation"
        print(f"    {f'decay gamma={g}':<22}{agg[key]:>12.2e}")
    print("\nThis reports finite-probe state/readout agreement and numerical "
          "coverage; it is not a privacy or legal-unlearning guarantee.")


if __name__ == "__main__":
    main()
