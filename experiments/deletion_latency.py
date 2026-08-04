"""Latency of maintained fixed-C deletion versus a retained-set refit.

For every deterministic trial, this benchmark:

1. fits a full one-class SVDD state;
2. chooses one token uniformly, representing an ordinary erasure request;
3. times only that maintained-state deletion; and
4. times an equivalent from-scratch batch-QP fit on the retained tokens.

The box constraint C is computed from the original context length and held
identical in both solves.  The from-scratch timing includes construction of the
retained-set Gram matrix.  Full-state setup is excluded from deletion latency
because it is work already maintained by the online layer.

Run from the repository root:
    python -m experiments.deletion_latency
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import platform
from pathlib import Path
import subprocess
import time
from dataclasses import dataclass

import numpy as np

from cp_svm import FastOneClassSVM


@dataclass(frozen=True)
class TrialResult:
    deletion_ms: float
    refit_ms: float
    active_deletion: bool
    alpha_deviation: float
    function_deviation: float
    partition_match: bool


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if name:
                return name
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def _timed(callable_):
    """Time one operation with cyclic GC excluded from the measured region."""
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        result = callable_()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    finally:
        if was_enabled:
            gc.enable()
    return result, elapsed_ms


def _partition(model: FastOneClassSVM) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sorted(group)) for group in (model.S, model.E, model.R))


def _assert_feasible(model: FastOneClassSVM, tol: float) -> None:
    if abs(float(model.alpha.sum()) - 1.0) > tol:
        raise AssertionError(f"sum(alpha) violation: {model.alpha.sum() - 1.0:.3e}")
    violation = max(
        float(np.max(-model.alpha, initial=0.0)),
        float(np.max(model.alpha - model.C, initial=0.0)),
    )
    if violation > tol:
        raise AssertionError(f"box-constraint violation: {violation:.3e}")


def run_trial(
    n: int,
    d: int,
    nu: float,
    kpar: float,
    seed: int,
    exactness_tol: float,
    refit_first: bool,
) -> TrialResult:
    rng = np.random.default_rng(seed)
    X = 1.5 * rng.standard_normal((n, d))
    C = 1.0 / (nu * n)

    decrement = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X)
    victim = int(rng.integers(n))
    active_deletion = victim in decrement.S or victim in decrement.E
    keep = np.arange(n) != victim

    def delete():
        decrement.remove_point(victim)
        return decrement

    def refit():
        return FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X[keep])

    # Alternate order to avoid assigning all systematic timing drift to one path.
    if refit_first:
        fresh, refit_ms = _timed(refit)
        decrement, deletion_ms = _timed(delete)
    else:
        decrement, deletion_ms = _timed(delete)
        fresh, refit_ms = _timed(refit)

    _assert_feasible(decrement, exactness_tol)
    _assert_feasible(fresh, exactness_tol)

    alpha_deviation = float(np.max(np.abs(decrement.alpha - fresh.alpha)))
    partition_match = _partition(decrement) == _partition(fresh)
    jitter_base = X[keep][: min(32, n - 1)]
    probes = np.vstack(
        [
            X[keep],
            X[victim : victim + 1],
            jitter_base + 0.05 * rng.standard_normal(jitter_base.shape),
        ]
    )
    function_deviation = float(
        np.max(
            np.abs(
                decrement.decision_function(probes)
                - fresh.decision_function(probes)
            )
        )
    )
    if function_deviation > exactness_tol:
        raise AssertionError(
            f"n={n}, seed={seed}: decrement/refit decision-function deviation "
            f"{function_deviation:.3e} exceeds {exactness_tol:.1e}"
        )

    return TrialResult(
        deletion_ms=deletion_ms,
        refit_ms=refit_ms,
        active_deletion=active_deletion,
        alpha_deviation=alpha_deviation,
        function_deviation=function_deviation,
        partition_match=partition_match,
    )


def _summary(n: int, results: list[TrialResult]) -> dict[str, float]:
    deletion = np.asarray([result.deletion_ms for result in results])
    refit = np.asarray([result.refit_ms for result in results])
    speedup = refit / deletion
    return {
        "n": float(n),
        "deletion_ms": float(np.median(deletion)),
        "refit_ms": float(np.median(refit)),
        "speedup": float(np.median(speedup)),
        "active_deletions": float(np.mean([r.active_deletion for r in results])),
        "partition_match": float(np.mean([r.partition_match for r in results])),
        "max_alpha_deviation": float(max(r.alpha_deviation for r in results)),
        "max_function_deviation": float(max(r.function_deviation for r in results)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[120, 256, 384, 512])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--nu", type=float, default=0.4)
    parser.add_argument("--kpar", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report",
        default=None,
        help="optional path for machine-readable summaries and per-trial values",
    )
    parser.add_argument(
        "--exactness-tol",
        type=float,
        default=1e-3,
        help="decision-function/feasibility tolerance; actual maxima are reported",
    )
    args = parser.parse_args()

    if args.trials < 1 or args.warmup < 0:
        parser.error("--trials must be positive and --warmup nonnegative")
    if any(n < 3 for n in args.sizes):
        parser.error("all context sizes must be at least 3")
    if not 0.0 < args.nu < 1.0:
        parser.error("--nu must lie strictly between zero and one")

    print("=== Fixed-C deletion latency: maintained decrement vs retained-set refit ===")
    print(
        f"platform={platform.platform()} | cpu={_cpu_name()} | numpy={np.__version__}\n"
        f"d={args.d}, nu={args.nu:g}, kpar={args.kpar:g}, "
        f"{args.trials} measured trials/size, {args.warmup} warmup; "
        "uniform random-token deletion; C fixed within each pair\n"
    )
    print(
        f"{'tokens':>7} {'delete ms':>11} {'refit ms':>11} {'speedup':>10} "
        f"{'active':>8} {'partition':>11} {'max |dalpha|':>13} {'max |df|':>11}"
    )

    summaries = []
    trials_by_size = {}
    for n in args.sizes:
        for warmup in range(args.warmup):
            run_trial(
                n=n,
                d=args.d,
                nu=args.nu,
                kpar=args.kpar,
                seed=args.seed + n * 100_000 + args.trials + warmup,
                exactness_tol=args.exactness_tol,
                refit_first=bool(warmup % 2),
            )

        results = []
        for trial in range(args.trials):
            # Independent per-trial streams make measured inputs invariant to
            # the number of warmups and to other context sizes.
            trial_seed = args.seed + n * 100_000 + trial
            result = run_trial(
                n=n,
                d=args.d,
                nu=args.nu,
                kpar=args.kpar,
                seed=trial_seed,
                exactness_tol=args.exactness_tol,
                refit_first=bool(trial % 2),
            )
            results.append(result)

        row = _summary(n, results)
        summaries.append(row)
        trials_by_size[str(n)] = [asdict(result) for result in results]
        print(
            f"{n:7d} {row['deletion_ms']:11.3f} {row['refit_ms']:11.3f} "
            f"{row['speedup']:9.1f}x {100.0 * row['active_deletions']:7.1f}% "
            f"{100.0 * row['partition_match']:10.1f}% "
            f"{row['max_alpha_deviation']:13.2e} "
            f"{row['max_function_deviation']:11.2e}"
        )

    if args.report:
        report = {
            "platform": platform.platform(),
            "cpu": _cpu_name(),
            "numpy": np.__version__,
            "configuration": {
                "sizes": args.sizes,
                "trials_per_size": args.trials,
                "warmups_per_size": args.warmup,
                "dimension": args.d,
                "nu": args.nu,
                "kernel": "RBF",
                "kernel_width": args.kpar,
                "seed": args.seed,
                "exactness_tolerance": args.exactness_tol,
                "victim_sampling": "uniform random token",
                "box_policy": "C fixed within each deletion/refit pair",
            },
            "summaries": summaries,
            "trials": trials_by_size,
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
