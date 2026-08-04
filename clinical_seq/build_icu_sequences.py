"""Build per-stay hourly vital-sign sequences from MIMIC-IV ICU chartevents.

Each ICU stay becomes a token stream: one token per hour, holding a small set
of standardized vital signs (Harutyunyan-style benchmark vitals). Long stretches
of physiologically stable hours are near-duplicate tokens (heavy redundancy);
deterioration episodes are the rare, informative tokens. This is exactly the
redundancy regime tested by SV-Attention's fixed-context support selection.

We stream the (3.3 GB gzipped) chartevents file in chunks, keep only the vital
itemids for a fixed cohort of stays, bin to hourly means per stay, forward-fill
within a stay, impute remaining gaps with the cohort median, and standardize per
feature using cohort statistics. We also keep the RAW (unstandardized) hourly
values and a per-hour "rare deterioration" label (physiologic thresholds) so the
downstream experiments can ask whether the gate retains the clinically rare
hours that heavy-hitter eviction drops.

Outputs an .npz cache (object arrays of variable-length per-stay sequences).

Run:
    python -m clinical_seq.build_icu_sequences \
        --icu-path /path/to/mimiciv/3.1/icu --n-stays 1500

The ICU path can instead be supplied through MIMICIV_ICU_PATH. This program
requires independently credentialed access to MIMIC-IV and does not download
or redistribute records or derived caches.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Feature set (Harutyunyan-style core vitals). Several MIMIC itemids map to one
# logical feature (e.g. arterial + non-invasive blood pressure).
FEATURES = ["hr", "sbp", "dbp", "rr", "spo2", "temp_c"]
ITEM2FEAT = {
    220045: "hr",
    220050: "sbp", 220179: "sbp",      # arterial + non-invasive systolic
    220051: "dbp", 220180: "dbp",      # arterial + non-invasive diastolic
    220210: "rr",
    220277: "spo2",
    223761: "temp_f", 223762: "temp_c",  # temp_f converted to C below
}
VITAL_ITEMIDS = list(ITEM2FEAT.keys())

# Plausible physiologic ranges; values outside are dropped as artifacts.
VALID_RANGE = {
    "hr": (10, 300), "sbp": (20, 300), "dbp": (5, 220),
    "rr": (2, 80), "spo2": (30, 100), "temp_c": (25, 45),
}

# A deterioration hour: any vital crosses a clinically alarming threshold.
def deterioration_label(raw: np.ndarray) -> np.ndarray:
    """raw: (T, 6) unstandardized [hr, sbp, dbp, rr, spo2, temp_c]. Returns (T,) bool."""
    hr, sbp, dbp, rr, spo2, temp = (raw[:, i] for i in range(6))
    flag = (
        (hr > 130) | (hr < 40)
        | (sbp < 90) | (sbp > 200)
        | (rr > 30) | (rr < 6)
        | (spo2 < 90)
        | (temp > 38.5) | (temp < 35.0)
    )
    return flag


def select_cohort(
    icu_path: Path,
    n_stays: int,
    min_los_days: float,
    seed: int,
) -> pd.DataFrame:
    stays = pd.read_csv(icu_path / "icustays.csv.gz",
                        usecols=["stay_id", "subject_id", "intime", "outtime", "los"],
                        parse_dates=["intime", "outtime"])
    stays = stays[stays.los >= min_los_days].copy()
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(stays))[:n_stays]
    cohort = stays.iloc[np.sort(idx)].reset_index(drop=True)
    return cohort


def scan_chartevents(
    icu_path: Path,
    stay_ids: set[int],
    chunksize: int,
    verbose: bool,
) -> pd.DataFrame:
    """One streaming pass over chartevents; keep vital rows for cohort stays."""
    keep = []
    t0 = time.time()
    n_rows = 0
    reader = pd.read_csv(
        icu_path / "chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=chunksize,
    )
    item_set = set(VITAL_ITEMIDS)
    for ci, chunk in enumerate(reader):
        n_rows += len(chunk)
        m = chunk.itemid.isin(item_set) & chunk.stay_id.isin(stay_ids) & chunk.valuenum.notna()
        if m.any():
            keep.append(chunk.loc[m, ["stay_id", "itemid", "charttime", "valuenum"]])
        if verbose and ci % 20 == 0:
            kept = sum(len(k) for k in keep)
            print(f"  chunk {ci:4d}  scanned {n_rows/1e6:6.1f}M rows  "
                  f"kept {kept/1e3:8.1f}k  ({time.time()-t0:5.0f}s)", flush=True)
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(
        columns=["stay_id", "itemid", "charttime", "valuenum"])
    if verbose:
        print(f"  scan done: {n_rows/1e6:.1f}M rows in {time.time()-t0:.0f}s, "
              f"kept {len(df)/1e3:.1f}k vital readings", flush=True)
    return df


def build_hourly(df: pd.DataFrame, cohort: pd.DataFrame, min_hours: int,
                 verbose: bool) -> dict:
    """Pivot filtered readings into per-stay (T, 6) hourly RAW arrays."""
    df = df.copy()
    df["feat"] = df.itemid.map(ITEM2FEAT)
    # Convert Fahrenheit temperature to Celsius, fold into temp_c.
    is_f = df.feat == "temp_f"
    df.loc[is_f, "valuenum"] = (df.loc[is_f, "valuenum"] - 32.0) * 5.0 / 9.0
    df.loc[is_f, "feat"] = "temp_c"

    intime = cohort.set_index("stay_id").intime
    df = df.join(intime, on="stay_id")
    df["hour"] = ((df.charttime - df.intime).dt.total_seconds() // 3600).astype(int)
    df = df[df.hour >= 0]

    # Drop physiologically impossible values before aggregation.
    for feat, (lo, hi) in VALID_RANGE.items():
        bad = (df.feat == feat) & ((df.valuenum < lo) | (df.valuenum > hi))
        df = df[~bad]

    # Hourly mean per (stay, hour, feature), then pivot to columns.
    g = (df.groupby(["stay_id", "hour", "feat"]).valuenum.mean()
           .unstack("feat").reindex(columns=FEATURES))

    sequences, raw_sequences, stay_ids, det_labels = [], [], [], []
    # Cohort-level medians for imputing features never measured in a stay.
    cohort_median = g.median(axis=0)
    for stay_id, sub in g.groupby(level="stay_id"):
        sub = sub.droplevel("stay_id")
        full = sub.reindex(range(sub.index.min(), sub.index.max() + 1))
        T = len(full)
        if T < min_hours:
            continue
        filled = full.ffill().bfill().fillna(cohort_median)
        raw = filled[FEATURES].to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all():
            continue
        raw_sequences.append(raw)
        stay_ids.append(int(stay_id))
        det_labels.append(deterioration_label(raw))
    if verbose:
        print(f"  built {len(stay_ids)} stays (>= {min_hours}h)", flush=True)

    # Standardize per feature using pooled cohort statistics.
    allraw = np.vstack(raw_sequences)
    mu = allraw.mean(axis=0)
    sd = allraw.std(axis=0)
    sd[sd < 1e-6] = 1.0
    sequences = [(r - mu) / sd for r in raw_sequences]
    return dict(
        sequences=np.array(sequences, dtype=object),
        raw_sequences=np.array(raw_sequences, dtype=object),
        det_labels=np.array(det_labels, dtype=object),
        stay_ids=np.array(stay_ids, dtype=np.int64),
        feat_mu=mu, feat_sd=sd, features=np.array(FEATURES),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Build hourly vital-sign sequences from credentialed MIMIC-IV ICU files."
    )
    env_icu = os.environ.get("MIMICIV_ICU_PATH")
    ap.add_argument(
        "--icu-path",
        type=Path,
        default=Path(env_icu).expanduser() if env_icu else None,
        help="MIMIC-IV v3.1 ICU directory; defaults to MIMICIV_ICU_PATH",
    )
    ap.add_argument("--n-stays", type=int, default=1500)
    ap.add_argument("--min-los-days", type=float, default=2.0)
    ap.add_argument("--min-hours", type=int, default=48)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    verbose = not args.quiet

    if args.icu_path is None:
        ap.error("provide --icu-path or set MIMICIV_ICU_PATH")
    icu_path = args.icu_path.resolve()
    required = [icu_path / "icustays.csv.gz", icu_path / "chartevents.csv.gz"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        ap.error("missing required MIMIC-IV ICU file(s): " + ", ".join(missing))

    out = Path(args.out) if args.out else CACHE_DIR / f"icu_vitals_n{args.n_stays}.npz"
    out = out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Selecting cohort: {args.n_stays} stays, LOS >= {args.min_los_days}d", flush=True)
    cohort = select_cohort(icu_path, args.n_stays, args.min_los_days, args.seed)
    print(f"  cohort size {len(cohort)}", flush=True)

    print("Scanning chartevents (one streaming pass)...", flush=True)
    df = scan_chartevents(icu_path, set(cohort.stay_id.tolist()), args.chunksize, verbose)

    print("Building hourly per-stay sequences...", flush=True)
    data = build_hourly(df, cohort, args.min_hours, verbose)

    np.savez_compressed(out, **data)
    lens = [len(s) for s in data["sequences"]]
    det_frac = np.mean([lbl.mean() for lbl in data["det_labels"]])
    print(f"\nSaved {len(lens)} stays -> {out}")
    print(f"  hours/stay: median {int(np.median(lens))}, "
          f"max {max(lens)}, total {sum(lens)} tokens")
    print(f"  mean deterioration-hour fraction: {det_frac:.3f}")
    print(f"  features: {list(data['features'])}")


if __name__ == "__main__":
    main()
