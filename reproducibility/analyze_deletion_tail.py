"""Summarize saved standalone deletion-audit records without rerunning a solver.

Accepts the JSON written by experiments.forgetting_rigor. The output contains
regime-level statistics only: no keys, clinical records or per-trial rows.
This recovers the condition/partition diagnostics, not the separate historical
scalar-readout experiment or the four-dimensional projection experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def summarize_regime(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("status") == "completed"]
    result = {
        "attempted": len(rows),
        "status_counts": dict(sorted(Counter(row.get("status", "missing") for row in rows).items())),
        "completed": len(completed),
    }
    if not completed:
        return result

    deviation = np.asarray([row["function_deviation"] for row in completed], dtype=float)
    condition = np.asarray([row["condition_number"] for row in completed], dtype=float)
    partition_match = np.asarray([row["partition_match"] for row in completed], dtype=bool)
    if not np.isfinite(deviation).all():
        raise ValueError("Completed rows contain non-finite gate-score deviations")
    finite = np.isfinite(condition)
    rho = spearmanr(condition[finite], deviation[finite]).statistic if finite.sum() > 1 else float("nan")
    top_count = max(1, math.ceil(len(completed) / 10))
    order = np.argsort(deviation, kind="stable")
    top = order[-top_count:]
    rest = order[:-top_count]

    def stats(mask):
        values = deviation[mask]
        return {
            "count": int(len(values)),
            "gate_score_median": _number(np.median(values)) if len(values) else None,
            "gate_score_worst": _number(values.max()) if len(values) else None,
        }

    result.update({
        "spearman_condition_vs_gate_score": _number(rho),
        "spearman_finite_pairs": int(finite.sum()),
        "spearman_excluded_nonfinite_condition": int((~finite).sum()),
        "partition_agreeing": stats(partition_match),
        "partition_disagreeing": stats(~partition_match),
        "largest_ten": {
            "count": min(10, len(completed)),
            "partition_disagreeing": int((~partition_match[order[-10:]]).sum()),
        },
        "largest_decile": {
            "definition": "largest ceil(completed/10) gate-score deviations; stable row-order tie break",
            "count": top_count,
            "condition_median": _number(np.median(condition[top])),
            "remaining_condition_median": _number(np.median(condition[rest])) if len(rest) else None,
        },
        "gate_score": {
            "median": _number(np.median(deviation)),
            "p95": _number(np.percentile(deviation, 95, method="linear")),
            "worst": _number(deviation.max()),
        },
    })
    return result


def analyze(path: Path) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw)
    trials = data.get("trials")
    if not isinstance(trials, dict) or not trials:
        raise ValueError("Expected nonempty trials mapping from experiments.forgetting_rigor")
    return {
        "schema_version": 1,
        "calculation": "Post-processing of saved standalone audit records; no solver execution",
        "source_file": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_schema_version": data.get("schema_version"),
        "metric_scope": "Condition/partition and gate-score diagnostics. Does not reconstruct scalar readouts or certify stored states.",
        "regimes": {name: summarize_regime(rows) for name, rows in trials.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local forgetting_rigor JSON with per-trial scalar audit records")
    parser.add_argument("--output", type=Path, help="Write aggregate JSON here; otherwise print to stdout")
    args = parser.parse_args()
    report = json.dumps(analyze(args.input), indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.resolve() == args.input.resolve():
            parser.error("Output must differ from the source audit JSON")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
