"""Audit fixed-C decrement against a retained-set refit.

The algebraic C&P path targets the same fixed-C optimum as solving again on the
retained keys. This experiment reports how closely the numerical implementation
reaches that target and how often it completes without an explicit edge case.

  R1  State agreement when the optimum is unique: alpha and the active-set
      partition agree with a retained-set refit under the identical C.
  R2  Function agreement when the optimum is non-unique: decision functions are
      compared on retained, removed, and jittered probes.
  R3  Coverage and conditioning are first-class results. We report attempted,
      completed, and structured edge-case counts over {Gaussian distinct,
      redundant duplicates, real MIMIC-IV vitals, trained-model keys}, together
      with median, mean, and worst deviation.
  R4  Coefficient decay is measured against the same refit target.

Run from the repository root:
    python -m experiments.forgetting_rigor
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from cp_svm import FastOneClassSVM
from cp_svm.kernels import radial_kernel
from cp_svm.oneclass_qp import recover_rho

MIMIC = Path(__file__).resolve().parent.parent / "clinical_seq" / "cache" / "icu_vitals_n1500.npz"


# ----------------------------------------------------------------- regimes
def regime_gaussian(rng, n=24, d=6):
    return rng.standard_normal((n, d))


def regime_redundant(rng, n_clusters=4, per=6, d=6):
    centers = rng.standard_normal((n_clusters, d)) * 2.0
    X = np.repeat(centers, per, axis=0) + 0.02 * rng.standard_normal((n_clusters * per, d))
    return X


_MIMIC = None
def regime_mimic(rng, cap=40, dedup_eps=0.1):
    global _MIMIC
    if _MIMIC is None:
        d = np.load(MIMIC, allow_pickle=True)
        _MIMIC = list(d["sequences"])
    X = _MIMIC[rng.integers(len(_MIMIC))][:cap]
    # dedupe near-identical forward-filled hours (the documented ill-conditioning)
    keep = [0]
    for i in range(1, len(X)):
        if np.min(np.sum((X[keep] - X[i]) ** 2, 1)) > dedup_eps ** 2:
            keep.append(i)
    return X[keep]


_TRAINED = None
def regime_trained(rng, n=24):
    """Keys from a trained SV recall model's learned Wk projection (real learned reps)."""
    global _TRAINED
    if _TRAINED is None:
        import torch
        from experiments.forget_qa_nlp import build
        m = build()
        m.load_state_dict(torch.load("outputs/qa_nlp_model.pt", weights_only=True)); m.eval()
        _TRAINED = m
    import torch
    d_in = _TRAINED.embed.in_features
    X = torch.tensor(rng.standard_normal((1, n, d_in)), dtype=torch.float32)
    with torch.no_grad():
        emb = _TRAINED.embed(X)
        K = _TRAINED.layer.Wk(emb)[0].numpy().astype(np.float64)
    return K


REGIMES = {"gaussian": regime_gaussian, "redundant": regime_redundant,
           "mimic": regime_mimic, "trained": regime_trained}


# ----------------------------------------------------------------- core
def cond_number(K):
    try:
        return float(np.linalg.cond(K))
    except Exception:
        return float("inf")


def _failure_code(stage: str, error: Exception) -> str:
    message = str(error).casefold()
    if "margin set emptied" in message or "margin set empty" in message:
        reason = "margin_empty"
    elif "no feasible" in message:
        reason = "no_feasible_step"
    elif "exceeded max_iter" in message or "exceeded max iter" in message:
        reason = "max_iterations"
    elif "singular" in message:
        reason = "singular_system"
    else:
        reason = type(error).__name__.casefold()
    return f"{stage}:{reason}"


def one_trial(X, rng, nu, kpar, n_forget, forget=None):
    """Run one fixed-C decrement/refit comparison with structured status."""
    n = len(X)
    if n < n_forget + 4:
        return {"status": "skipped", "failure": "input:insufficient_points"}
    C = 1.0 / (nu * n)                                    # fixed box, identical for both
    forget = (
        sorted(rng.choice(n, n_forget, replace=False).tolist())
        if forget is None
        else sorted({int(index) for index in forget})
    )
    n_forget = len(forget)
    keep = [i for i in range(n) if i not in set(forget)]
    retained_capacity = len(keep) * C
    base = {
        "n_points": int(n),
        "n_retained": len(keep),
        "n_forget": n_forget,
        "C": float(C),
        "retained_capacity": float(retained_capacity),
    }
    if retained_capacity < 1.0 - 1e-12:
        return {
            **base,
            "status": "failed",
            "failure": "fixed_c_preflight:infeasible",
            "refit_status": "infeasible",
        }

    try:
        # (1) seed the full problem.
        m = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X)
        a_full = np.array(m.alpha)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {
            **base,
            "status": "failed",
            "failure": _failure_code("full_seed", error),
            "refit_status": "not_run",
        }

    try:
        # (2) solve the retained reference independently even if decrement fails.
        f = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X[keep])
        a_fresh = np.array(f.alpha)
        S_f, E_f = set(f.S), set(f.E)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {
            **base,
            "status": "failed",
            "failure": _failure_code("refit", error),
            "refit_status": "failed",
        }

    try:
        # (3) decrement from the full solution via the C&P reverse path.
        for c in sorted(forget, reverse=True):
            m.remove_point(c)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {
            **base,
            "status": "failed",
            "failure": _failure_code("decrement", error),
            "refit_status": "completed",
        }

    a_dec = np.array(m.alpha)                              # aligned to `keep` order
    S_dec, E_dec = set(m.S), set(m.E)

    d_alpha = float(np.max(np.abs(a_dec - a_fresh)))
    partition_exact = (S_dec == S_f) and (E_dec == E_f)
    # Primary agreement metric: finite-probe decision function f(x) over retained,
    # removed, and jittered locations. This is less sensitive than raw coefficients
    # to non-unique optima and compares decrement with the declared retained refit.
    Xk = X[keep]
    probes = np.vstack([Xk, X[forget], Xk + 0.1 * rng.standard_normal(Xk.shape)])
    f_dev = float(np.max(np.abs(m.decision_function(probes) - f.decision_function(probes))))
    # R4: decay baseline on the SAME functional metric -- keep forgotten tokens with a
    # scaled coefficient; its decision function still differs from retrain-without
    # (decay never re-solved), so its functional deviation is large.
    try:
        a_decay = a_full.copy()
        a_decay[forget] *= 0.01
        rho_decay = recover_rho(radial_kernel(X, X, kpar), a_decay, C)
        f_decay = 2.0 * (radial_kernel(probes, X, kpar) @ a_decay) - rho_decay
        decay_fdev = float(np.max(np.abs(f_decay - f.decision_function(probes))))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {"status": "failed", "failure": _failure_code("decay", error)}

    cond = cond_number(radial_kernel(Xk, Xk, kpar))        # RBF gate Gram conditioning
    return {
        **base,
        "status": "completed",
        "refit_status": "completed",
        "alpha_deviation": d_alpha,
        "partition_match": bool(partition_exact),
        "function_deviation": f_dev,
        "decay_function_deviation": decay_fdev,
        "condition_number": cond,
    }


def _distribution(values):
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "worst": float(np.max(array)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--nu", type=float, default=0.4)
    ap.add_argument("--kpar", type=float, default=2.0)
    ap.add_argument("--n-forget", type=int, default=2)
    ap.add_argument(
        "--regimes",
        nargs="+",
        choices=tuple(REGIMES),
        default=list(REGIMES),
    )
    ap.add_argument("--fig", default="outputs/forgetting_rigor.png")
    ap.add_argument("--report", default="outputs/forgetting_rigor.json")
    args = ap.parse_args()

    print("=== Fixed-C decrement versus retained-set refit ===")
    print(f"(nu={args.nu}, kpar={args.kpar}, forget {args.n_forget} point(s)/trial, "
          f"{args.trials} trials/regime; C fixed identical for both)\n")
    print("Function agreement = max |decision_function| deviation, decrement vs "
          "retained-set refit.\nPartition== = state-level alpha/(S,E,R) match when the optimum is "
          "unique; non-unique on exact duplicates).\n")
    header = (f"{'regime':<11}{'cover':>7}{'state==':>9}"
              f"{'decrement/refit f-dev (median / mean / worst)':>47}"
              f"{'decay median':>14}")
    print(header)
    all_fd, summary, cond_rows, trial_reports = [], {}, [], {}
    for name in args.regimes:
        gen = REGIMES[name]
        rng = np.random.default_rng(0)
        outcomes = []
        for _ in range(args.trials):
            X = np.atleast_2d(gen(rng))
            outcomes.append(
                one_trial(X, rng, args.nu, args.kpar, args.n_forget)
            )
        trial_reports[name] = outcomes
        completed = [row for row in outcomes if row["status"] == "completed"]
        failures = Counter(
            row["failure"] for row in outcomes if row["status"] != "completed"
        )
        done = len(completed)
        fds = [row["function_deviation"] for row in completed]
        parts = [row["partition_match"] for row in completed]
        decs = [row["decay_function_deviation"] for row in completed]
        for row in completed:
            cond = row["condition_number"]
            if np.isfinite(cond):
                cond_rows.append((cond, row["function_deviation"], name))
        if done == 0:
            print(f"{name:<11}{'0/'+str(args.trials):>7}  (no completed trials)")
            summary[name] = {
                "attempted": args.trials,
                "completed": 0,
                "coverage": 0.0,
                "failure_counts": dict(failures),
            }
            continue
        all_fd += fds
        finite_conditions = [
            row["condition_number"]
            for row in completed
            if np.isfinite(row["condition_number"])
        ]
        summary[name] = {
            "attempted": args.trials,
            "completed": done,
            "coverage": done / args.trials,
            "failure_counts": dict(failures),
            "partition_match_rate": float(np.mean(parts)),
            "function_deviation": _distribution(fds),
            "decay_function_deviation": _distribution(decs),
            "condition_number": (
                _distribution(finite_conditions)
                if finite_conditions
                else None
            ),
        }
        print(f"{name:<11}{f'{done}/{args.trials}':>7}{f'{100*np.mean(parts):.0f}%':>9}"
              f"{f'{np.median(fds):.1e} / {np.mean(fds):.1e} / {np.max(fds):.1e}':>47}"
              f"{np.median(decs):>13.2e}")
        if failures:
            print(
                " " * 11
                + "edge cases: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(failures.items())
                )
            )

    if all_fd:
        print(f"\nAcross completed trials: worst decision-function deviation = "
              f"{np.max(all_fd):.1e}.")
    print(
        "The algebraic target is fixed-C refit equivalence. The values above "
        "are the measured numerical agreement and solver coverage; failures "
        "are not silently discarded."
    )

    report = {
        "schema_version": 2,
        "claim": "fixed-C decrement/refit agreement to logged numerical tolerance",
        "parameters": {
            "trials_per_regime": args.trials,
            "nu": args.nu,
            "kpar": args.kpar,
            "n_forget": args.n_forget,
            "regimes": args.regimes,
        },
        "summary": summary,
        "trials": trial_reports,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"saved report -> {report_path}")
    if args.fig:
        render(summary, cond_rows, args.fig)
        print(f"\nsaved figure -> {args.fig}")


def render(summary, cond_rows, path):
    import matplotlib.pyplot as plt
    from svattn import figstyle
    figstyle.apply()
    names = [name for name, row in summary.items() if row["completed"]]
    if not names:
        raise RuntimeError("cannot render without a completed regime")
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14, 3.8))
    x = np.arange(len(names))
    coverage = [100.0 * summary[n]["coverage"] for n in names]
    axA.bar(x, coverage, color=figstyle.BLUE)
    axA.set_ylim(0, 105)
    axA.set_xticks(x); axA.set_xticklabels(names, fontsize=9)
    axA.set_ylabel("completed trials (%)")
    figstyle.caps_title(axA, "(a) numerical-path coverage")

    median = [
        max(summary[n]["function_deviation"]["median"], 1e-16)
        for n in names
    ]
    worst = [
        max(summary[n]["function_deviation"]["worst"], 1e-16)
        for n in names
    ]
    decay = [
        max(summary[n]["decay_function_deviation"]["median"], 1e-16)
        for n in names
    ]
    axB.bar(
        x - 0.25, median, 0.25, color=figstyle.BLUE,
        label="decrement median",
    )
    axB.bar(
        x, worst, 0.25, color=figstyle.GREEN,
        label="decrement worst",
    )
    axB.bar(
        x + 0.25, decay, 0.25, color=figstyle.RED,
        label="decay median",
    )
    axB.set_yscale("log")
    axB.set_xticks(x); axB.set_xticklabels(names, fontsize=9)
    axB.set_ylabel("deviation from fixed-C refit")
    figstyle.caps_title(axB, "(b) typical and worst agreement")
    axB.legend(fontsize=7, frameon=False)

    if cond_rows:
        c = np.array([row[0] for row in cond_rows])
        d = np.array([max(row[1], 1e-16) for row in cond_rows])
        axC.scatter(c, d, s=8, color=figstyle.BLUE, alpha=0.4)
        axC.set_xscale("log"); axC.set_yscale("log")
        axC.set_xlabel("gate Gram condition number")
        axC.set_ylabel("decrement/refit deviation")
        figstyle.caps_title(axC, "(c) conditioning explains the tail")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    main()
