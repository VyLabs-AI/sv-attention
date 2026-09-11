"""Minimum-norm tie-break for the fixed-C deletion certificate.

The rigor audit (``forgetting_rigor.py``) observed larger deviations between
maintained decrement and retained-key refit for some near-duplicate inputs.
This experiment tests minimum-norm projection within each side's declared
objective tolerance; the observed improvements do not establish that every
deviation comes from exact non-uniqueness or that the projection is exact.
This script applies the same tie-break to both sides: project each coefficient
vector to the minimum-norm point of the epsilon-optimal set

    min ||alpha||^2  s.t.  sum alpha = 1,  0 <= alpha <= C,
                           alpha^T K alpha - diag(K)^T alpha <= f_side + eps,

where f_side is the objective the side itself achieved (the executor projects
from the decrement, an independent auditor from the refit; neither sees the
other's value) and eps is relative to max(1, |f_side|). If the two solutions
are tied to within eps, the two epsilon-optimal sets nearly coincide and so do
their minimum-norm points, which are unique; agreement after projection then
tests whether the tail really is a tie. Trials, seeds, regimes, probes, and the gate-score metric are identical
to the rigor audit. A readout deviation is added: each key carries a random
value vector, the readout is the alpha-weighted, kernel-weighted mean of the
values at each probe, and the deviation is reported as a fraction of the
readout range over the probes, before and after the tie-break.

Run: PYTHONPATH=. python -m experiments.forgetting_tiebreak
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cvxpy as cp
import numpy as np

from cp_svm import FastOneClassSVM
from cp_svm.kernels import radial_kernel
from cp_svm.oneclass_qp import recover_rho, partition_sets
from experiments.forgetting_rigor import REGIMES, _distribution


def objective(K: np.ndarray, a: np.ndarray) -> float:
    return float(a @ K @ a - np.diag(K) @ a)


def _spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    return float(np.corrcoef(rx, ry)[0, 1])


def min_norm_projection(K: np.ndarray, a_star: np.ndarray, C: float, f_side: float, eps: float) -> np.ndarray | None:
    n = len(a_star)
    a = cp.Variable(n)
    Ksym = 0.5 * (K + K.T)
    constraints = [cp.sum(a) == 1, a >= 0, a <= C, cp.quad_form(a, cp.psd_wrap(Ksym)) - np.diag(K) @ a <= f_side + eps]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(a)), constraints)
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            prob.solve(solver=solver, **({"eps": 1e-10, "max_iters": 200000} if solver == cp.SCS else {}))
        except (cp.error.SolverError, ValueError, ArithmeticError):
            continue
        if a.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            out = np.clip(np.asarray(a.value, dtype=np.float64), 0.0, C)
            return out / out.sum()
    return None


def gate_scores(K_probe: np.ndarray, K_keys: np.ndarray, a: np.ndarray, C: float) -> np.ndarray:
    rho = recover_rho(K_keys, a, C)
    return 2.0 * (K_probe @ a) - rho


def readout(K_probe: np.ndarray, a: np.ndarray, V: np.ndarray) -> np.ndarray:
    w = K_probe * a[None, :]
    return (w @ V) / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)


def one_trial(X, rng, nu, kpar, n_forget, eps_rels, value_rng):
    n = len(X)
    C = 1.0 / (nu * n)
    forget = sorted(rng.choice(n, n_forget, replace=False).tolist())
    keep = [i for i in range(n) if i not in set(forget)]
    if len(keep) * C < 1.0 - 1e-12:
        return {"status": "failed", "failure": "infeasible"}
    try:
        m = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X)
        f = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X[keep])
        for c in sorted(forget, reverse=True):
            m.remove_point(c)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {"status": "failed", "failure": type(error).__name__}
    a_dec = np.array(m.alpha)
    a_fresh = np.array(f.alpha)
    Xk = X[keep]
    probes = np.vstack([Xk, X[forget], Xk + 0.1 * rng.standard_normal(Xk.shape)])
    K_keys = radial_kernel(Xk, Xk, kpar)
    K_probe = radial_kernel(probes, Xk, kpar)
    V = value_rng.standard_normal((len(keep), 4))

    def compare(a1, a2):
        g1, g2 = gate_scores(K_probe, K_keys, a1, C), gate_scores(K_probe, K_keys, a2, C)
        r1, r2 = readout(K_probe, a1, V), readout(K_probe, a2, V)
        rng_r = float(np.max(np.ptp(r2, axis=0)))
        S1, E1, _ = partition_sets(a1, C)
        S2, E2, _ = partition_sets(a2, C)
        return {
            "gate_score_deviation": float(np.max(np.abs(g1 - g2))),
            "gate_score_range_fraction": float(np.max(np.abs(g1 - g2)) / max(np.ptp(g2), 1e-300)),
            "readout_deviation_fraction": float(np.max(np.abs(r1 - r2)) / max(rng_r, 1e-300)),
            "alpha_deviation": float(np.max(np.abs(a1 - a2))),
            "partition_match": bool(set(S1) == set(S2) and set(E1) == set(E2)),
        }

    out = {"status": "completed", "n_retained": len(keep), "raw": compare(a_dec, a_fresh), "tiebreak": {}}
    f_dec, f_fresh = objective(K_keys, a_dec), objective(K_keys, a_fresh)
    out["objective_gap"] = float(abs(f_dec - f_fresh))
    out["objective_gap_relative"] = float(abs(f_dec - f_fresh) / max(1.0, abs(f_fresh)))
    for eps_rel in eps_rels:
        # Each side projects from its own achieved objective; neither sees the other's.
        p_dec = min_norm_projection(K_keys, a_dec, C, f_dec, eps_rel * max(1.0, abs(f_dec)))
        p_fresh = min_norm_projection(K_keys, a_fresh, C, f_fresh, eps_rel * max(1.0, abs(f_fresh)))
        if p_dec is None or p_fresh is None:
            out["tiebreak"][str(eps_rel)] = {"status": "solver_failed"}
            continue
        rec = compare(p_dec, p_fresh)
        rec["status"] = "ok"
        rec["projection_moved_decrement"] = float(np.max(np.abs(p_dec - a_dec)))
        rec["projection_moved_refit"] = float(np.max(np.abs(p_fresh - a_fresh)))
        out["tiebreak"][str(eps_rel)] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--nu", type=float, default=0.4)
    ap.add_argument("--kpar", type=float, default=2.0)
    ap.add_argument("--n-forget", type=int, default=2)
    ap.add_argument("--eps-rels", default="1e-10,1e-8,1e-6")
    ap.add_argument("--regimes", nargs="+", choices=tuple(REGIMES), default=list(REGIMES))
    ap.add_argument("--report", default="outputs/forgetting_tiebreak.json")
    ap.add_argument("--summary", default="outputs/forgetting_tiebreak_summary.json")
    args = ap.parse_args()
    eps_rels = [float(x) for x in args.eps_rels.split(",")]

    summary, trials = {}, {}
    for name in args.regimes:
        gen = REGIMES[name]
        rng = np.random.default_rng(0)
        value_rng = np.random.default_rng(12345)
        rows = []
        for _ in range(args.trials):
            X = np.atleast_2d(gen(rng))
            rows.append(one_trial(X, rng, args.nu, args.kpar, args.n_forget, eps_rels, value_rng))
        trials[name] = rows
        done = [r for r in rows if r["status"] == "completed"]
        s = {"attempted": args.trials, "completed": len(done), "raw": {}, "tiebreak": {}}
        for key in ("gate_score_deviation", "gate_score_range_fraction", "readout_deviation_fraction", "alpha_deviation"):
            s["raw"][key] = _distribution([r["raw"][key] for r in done])
        s["raw"]["partition_match_rate"] = float(np.mean([r["raw"]["partition_match"] for r in done]))
        s["objective_gap"] = _distribution([r["objective_gap"] for r in done])
        s["objective_gap_relative"] = _distribution([r["objective_gap_relative"] for r in done])
        # Trials whose two solvers disagree most in objective value are the trials
        # with the largest raw gate-score deviation; the rank correlation records it.
        s["objective_gap_vs_raw_gate_score_spearman"] = _spearman(
            [r["objective_gap_relative"] for r in done], [r["raw"]["gate_score_deviation"] for r in done]
        )
        for eps_rel in eps_rels:
            ok = [r["tiebreak"][str(eps_rel)] for r in done if r["tiebreak"][str(eps_rel)]["status"] == "ok"]
            block = {"solved": len(ok), "of": len(done)}
            for key in (
                "gate_score_deviation",
                "gate_score_range_fraction",
                "readout_deviation_fraction",
                "alpha_deviation",
                "projection_moved_decrement",
                "projection_moved_refit",
            ):
                block[key] = _distribution([r[key] for r in ok]) if ok else None
            block["partition_match_rate"] = float(np.mean([r["partition_match"] for r in ok])) if ok else None
            s["tiebreak"][str(eps_rel)] = block
        summary[name] = s
        raw, tb = s["raw"], s["tiebreak"][str(eps_rels[1])]
        print(
            f"{name:<12} n={len(done)} raw gate-score worst {raw['gate_score_deviation']['worst']:.1e} "
            f"(readout worst {100*raw['readout_deviation_fraction']['worst']:.1f}% of range, partition {100*raw['partition_match_rate']:.0f}%) | "
            f"tie-break eps={eps_rels[1]:g}: worst {tb['gate_score_deviation']['worst']:.1e} "
            f"(readout worst {100*tb['readout_deviation_fraction']['worst']:.1f}%, partition {100*tb['partition_match_rate']:.0f}%, solved {tb['solved']}/{tb['of']})"
        )
    report = {
        "schema_version": 1,
        "claim": "minimum-norm epsilon-optimal tie-break applied to both decrement and refit",
        "parameters": {"trials_per_regime": args.trials, "nu": args.nu, "kpar": args.kpar, "n_forget": args.n_forget, "eps_rels": eps_rels, "values": "N(0,1) 4-vector per retained key, seed 12345"},
        "summary": summary,
        "trials": trials,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
    Path(args.summary).write_text(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=1) + "\n")
    print(f"saved {args.report} and {args.summary}")


if __name__ == "__main__":
    main()
