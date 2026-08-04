"""Reproducible throughput benchmarks for the causal SV layer.

The default command is the historical forward-only micro sweep.  ``--paper``
runs the publication benchmark: a full forward/backward step through the 3.22M
CharGPT at the paper configuration.  It reports the current mixed
PyTorch-CPU/MLX-GPU hybrid (SV-gated long range + local q.k softmax) alongside
*separate* PyTorch CPU and MPS softmax references.  ``--sv-readout rbf`` retains
the old pure-RBF current-path measurement as an optional diagnostic.  The slow
re-stream and persistent active-set paths are retained as explicitly labelled
historical RBF batch extrapolations.

Examples:
    PYTHONPATH=. python -m experiments.g1_throughput
    PYTHONPATH=. python -m experiments.g1_throughput --paper
    PYTHONPATH=. python -m experiments.g1_throughput --lm  # compatibility alias

The timed "training step" is zero_grad + forward + cross-entropy backward.  It
does not include an optimizer update, data loading, evaluation, or checkpointing.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from cp_svm.oneclass_fast import FastOneClassSVM
from svattn.baselines import CausalSoftmaxAttention, CausalLinearAttention, DeltaNet
from svattn.causal_sv_attention import CausalSVAttention
from svattn.nanogpt import CharGPT, GPTConfig

try:  # MLX remains optional off Apple silicon.
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised on non-Apple CI
    mx = None


PAPER_LAYERS = 4
PAPER_D_MODEL = 256
PAPER_HEADS = 4
PAPER_BLOCK = 256
PAPER_BATCH = 16
PAPER_CHUNK = 64
PAPER_C = 0.25
PAPER_KPAR = 2.0
PAPER_FISTA_ITERS = 80
PAPER_VOCAB = 256


@dataclass(frozen=True)
class Timing:
    """Independent, synchronized wall-clock samples."""

    samples_s: tuple[float, ...]

    @property
    def median_s(self) -> float:
        return float(np.median(self.samples_s))

    @property
    def q1_s(self) -> float:
        return float(np.percentile(self.samples_s, 25))

    @property
    def q3_s(self) -> float:
        return float(np.percentile(self.samples_s, 75))


@dataclass(frozen=True)
class BenchmarkRow:
    method: str
    framework: str
    device: str
    measurement: str
    timing: Timing
    scale_to_target: float = 1.0

    @property
    def step_s(self) -> float:
        return self.timing.median_s * self.scale_to_target

    @property
    def q1_s(self) -> float:
        return self.timing.q1_s * self.scale_to_target

    @property
    def q3_s(self) -> float:
        return self.timing.q3_s * self.scale_to_target


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if mx is not None:
        mx.random.seed(seed)


def _synchronize(*, use_mps: bool = False, use_mlx: bool = False) -> None:
    """Finish queued accelerator work before reading the wall clock."""
    if use_mps:
        torch.mps.synchronize()
    if use_mlx:
        if mx is None:  # Defensive: callers should have skipped the MLX row.
            raise RuntimeError("MLX synchronization requested but MLX is unavailable")
        mx.synchronize()


def _measure(
    operation: Callable[[], None],
    *,
    warmups: int,
    repetitions: int,
    use_mps: bool = False,
    use_mlx: bool = False,
) -> Timing:
    """Warm up, then collect one synchronized wall-clock sample per operation."""
    gc.collect()
    _synchronize(use_mps=use_mps, use_mlx=use_mlx)
    for _ in range(warmups):
        operation()
        _synchronize(use_mps=use_mps, use_mlx=use_mlx)

    samples = []
    for _ in range(repetitions):
        _synchronize(use_mps=use_mps, use_mlx=use_mlx)
        t0 = time.perf_counter()
        operation()
        _synchronize(use_mps=use_mps, use_mlx=use_mlx)
        samples.append(time.perf_counter() - t0)
    return Timing(tuple(samples))


def _benchmark_training_step(
    model: torch.nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    *,
    warmups: int,
    repetitions: int,
    use_mps: bool = False,
    use_mlx: bool = False,
) -> Timing:
    model.train()

    def step() -> None:
        model.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        loss.backward()

    return _measure(
        step,
        warmups=warmups,
        repetitions=repetitions,
        use_mps=use_mps,
        use_mlx=use_mlx,
    )


def _paper_config() -> GPTConfig:
    return GPTConfig(
        vocab=PAPER_VOCAB,
        d_model=PAPER_D_MODEL,
        n_heads=PAPER_HEADS,
        n_layers=PAPER_LAYERS,
        block=PAPER_BLOCK,
    )


def _build_paper_model(
    kind: str,
    *,
    seed: int,
    device: str,
    sv_readout: str = "softmax",
) -> CharGPT:
    _seed_all(seed)
    cfg = _paper_config()
    if kind == "softmax":
        model = CharGPT(cfg, attn_name="softmax")
    elif kind == "sv-mlx":
        model = CharGPT(
            cfg,
            attn_name="sv",
            C=PAPER_C,
            kpar=PAPER_KPAR,
            chunk=PAPER_CHUNK,
            solver="mlx",
            persistent=False,
            fista_iters=PAPER_FISTA_ITERS,
            readout=sv_readout,
        )
    elif kind in {"sv-restream", "sv-persistent"}:
        # The sequential active-set branches in CausalSVAttention predate the
        # hybrid and always call the RBF readout; self.readout is only consumed
        # by the batched MLX branch. Keep these as labelled historical probes.
        model = CharGPT(
            cfg,
            attn_name="sv",
            C=PAPER_C,
            kpar=PAPER_KPAR,
            chunk=PAPER_CHUNK,
            solver="exact",
            persistent=(kind == "sv-persistent"),
            readout="rbf",
        )
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"unknown paper benchmark model: {kind}")
    return model.to(device)


def _paper_batch(batch: int, *, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    idx = torch.randint(PAPER_VOCAB, (batch, PAPER_BLOCK), generator=generator)
    targets = torch.randint(PAPER_VOCAB, (batch, PAPER_BLOCK), generator=generator)
    return idx.to(device), targets.to(device)


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _sw_vers(flag: str) -> str:
    try:
        return subprocess.check_output(
            ["sw_vers", flag], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _print_runtime_metadata() -> None:
    chip = _sysctl("machdep.cpu.brand_string")
    if chip == "unknown":
        chip = platform.processor() or platform.machine()
    model = _sysctl("hw.model")
    cores = _sysctl("hw.physicalcpu")
    memory = _sysctl("hw.memsize")
    memory_text = (
        f"{int(memory) / 2**30:.0f} GiB" if memory.isdigit() else memory
    )
    mlx_version = _package_version("mlx")
    mlx_device = str(mx.default_device()) if mx is not None else "unavailable"

    print("Hardware/runtime metadata")
    print(
        f"  Hardware: {chip}; model={model}; physical cores={cores}; "
        f"memory={memory_text}; arch={platform.machine()}"
    )
    print(
        f"  OS: macOS {_sw_vers('-productVersion')} "
        f"(build {_sw_vers('-buildVersion')})"
    )
    print(
        f"  Runtime: Python {platform.python_version()}; PyTorch {torch.__version__}; "
        f"NumPy {np.__version__}; MLX {mlx_version}; CVXPY {_package_version('cvxpy')}"
    )
    print(
        f"  PyTorch: CPU threads={torch.get_num_threads()}, "
        f"interop threads={torch.get_num_interop_threads()}, "
        f"MPS available={torch.backends.mps.is_available()}"
    )
    print(f"  MLX default device: {mlx_device}")
    print(f"  Command: {shlex.join([sys.executable, *sys.argv])}")


def _format_duration(seconds: float) -> str:
    hours = seconds / 3600.0
    if hours < 24:
        return f"{hours:.2f} h"
    return f"{hours / 24.0:.2f} d"


def _print_paper_table(rows: list[BenchmarkRow], *, step_budget: int) -> None:
    cpu_softmax = next(
        row
        for row in rows
        if row.method == "Softmax reference" and row.device == "CPU (PyTorch)"
    )
    mps_softmax = next(
        (
            row
            for row in rows
            if row.method == "Softmax reference"
            and row.device == "Apple GPU via MPS (PyTorch)"
        ),
        None,
    )
    tokens = PAPER_BATCH * PAPER_BLOCK
    print("\nPaper-ready results (Markdown)")
    print(
        "| Method | Framework | Device(s) | Measurement | "
        "median ms/step [IQR] | tokens/s | x CPU-softmax time | "
        "x MPS-softmax time | "
        f"{step_budget:,}-step wall-clock |"
    )
    print("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        step_ms = row.step_s * 1e3
        iqr = f"{row.q1_s * 1e3:,.1f}-{row.q3_s * 1e3:,.1f}"
        throughput = tokens / row.step_s
        cpu_ratio = row.step_s / cpu_softmax.step_s
        mps_ratio = (
            f"{row.step_s / mps_softmax.step_s:.2f}x"
            if mps_softmax is not None
            else "n/a"
        )
        wall = _format_duration(row.step_s * step_budget)
        print(
            f"| {row.method} | {row.framework} | {row.device} | "
            f"{row.measurement} | {step_ms:,.1f} [{iqr}] | "
            f"{throughput:,.0f} | {cpu_ratio:.2f}x | {mps_ratio} | {wall} |"
        )


def main_paper(
    *,
    warmups: int = 3,
    repetitions: int = 7,
    probe_batch: int = 2,
    step_budget: int = 50_000,
    seed: int = 0,
    include_legacy: bool = True,
    sv_readout: str = "softmax",
) -> None:
    """Benchmark a complete 3.22M CharGPT forward/backward training step."""
    _print_runtime_metadata()
    cfg = _paper_config()
    tokens = PAPER_BATCH * PAPER_BLOCK
    readout_description = (
        "hybrid: SV-gated long range + local causal q.k softmax"
        if sv_readout == "softmax"
        else "pure RBF diagnostic (no local softmax path)"
    )
    print(
        "\nPaper configuration\n"
        f"  CharGPT: reported params=3.22M; layers={cfg.n_layers}; "
        f"d_model={cfg.d_model}; heads={cfg.n_heads}; block={cfg.block}; "
        f"batch={PAPER_BATCH}; tokens/step={tokens:,}\n"
        f"  SV: C={PAPER_C}; kpar={PAPER_KPAR}; chunk={PAPER_CHUNK}; "
        f"FISTA iterations={PAPER_FISTA_ITERS}; readout={sv_readout} "
        f"({readout_description})\n"
        "  Timed step: zero_grad + full-model forward + cross-entropy backward "
        "(no optimizer/data/eval)\n"
        f"  Statistic: median of {repetitions} separately synchronized repetitions "
        f"after {warmups} warmups; IQR shown; RNG seed={seed}"
    )

    torch_version = str(torch.__version__)
    numpy_version = str(np.__version__)
    mlx_version = _package_version("mlx")
    cvxpy_version = _package_version("cvxpy")
    rows: list[BenchmarkRow] = []

    idx_cpu, targets_cpu = _paper_batch(PAPER_BATCH, seed=seed, device="cpu")
    softmax_cpu = _build_paper_model("softmax", seed=seed, device="cpu")
    if softmax_cpu.n_params() != 3_220_992:
        raise RuntimeError(
            f"paper model changed: expected 3,220,992 reported params, "
            f"got {softmax_cpu.n_params():,}"
        )
    timing = _benchmark_training_step(
        softmax_cpu,
        idx_cpu,
        targets_cpu,
        warmups=warmups,
        repetitions=repetitions,
    )
    rows.append(
        BenchmarkRow(
            "Softmax reference",
            f"PyTorch {torch_version}",
            "CPU (PyTorch)",
            f"direct B={PAPER_BATCH}",
            timing,
        )
    )
    del softmax_cpu
    gc.collect()

    if torch.backends.mps.is_available():
        idx_mps, targets_mps = _paper_batch(PAPER_BATCH, seed=seed, device="mps")
        softmax_mps = _build_paper_model("softmax", seed=seed, device="mps")
        timing = _benchmark_training_step(
            softmax_mps,
            idx_mps,
            targets_mps,
            warmups=warmups,
            repetitions=repetitions,
            use_mps=True,
        )
        rows.append(
            BenchmarkRow(
                "Softmax reference",
                f"PyTorch {torch_version}",
                "Apple GPU via MPS (PyTorch)",
                f"direct B={PAPER_BATCH}",
                timing,
            )
        )
        del softmax_mps, idx_mps, targets_mps
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
    else:
        print("\n[skip] PyTorch MPS softmax: MPS is unavailable in this runtime.")

    if mx is not None:
        mx.set_default_device(mx.gpu)
        _seed_all(seed)
        mixed = _build_paper_model(
            "sv-mlx",
            seed=seed,
            device="cpu",
            sv_readout=sv_readout,
        )
        timing = _benchmark_training_step(
            mixed,
            idx_cpu,
            targets_cpu,
            warmups=warmups,
            repetitions=repetitions,
            use_mlx=True,
        )
        rows.append(
            BenchmarkRow(
                (
                    "SV hybrid, batched FISTA (current path)"
                    if sv_readout == "softmax"
                    else "SV RBF, batched FISTA (diagnostic)"
                ),
                f"PyTorch {torch_version} + NumPy {numpy_version} + MLX {mlx_version}",
                "CPU (PyTorch/KKT) + Apple GPU (MLX FISTA)",
                f"direct B={PAPER_BATCH}",
                timing,
            )
        )
        del mixed
        gc.collect()
        mx.synchronize()
    else:
        print("\n[skip] Mixed SV row: MLX is unavailable in this runtime.")

    if include_legacy:
        idx_probe, targets_probe = _paper_batch(probe_batch, seed=seed, device="cpu")
        scale = PAPER_BATCH / probe_batch
        exact_framework = (
            f"PyTorch {torch_version} + NumPy {numpy_version} + "
            f"CVXPY {cvxpy_version}"
        )
        for kind, label in (
            ("sv-restream", "SV RBF, re-stream active-set (legacy)"),
            ("sv-persistent", "SV RBF, persistent rank-one (legacy)"),
        ):
            model = _build_paper_model(kind, seed=seed, device="cpu")
            timing = _benchmark_training_step(
                model,
                idx_probe,
                targets_probe,
                warmups=warmups,
                repetitions=repetitions,
            )
            rows.append(
                BenchmarkRow(
                    label,
                    exact_framework,
                    "CPU (all components)",
                    f"extrapolated: direct B={probe_batch} median "
                    f"{timing.median_s * 1e3:,.1f} ms (x{scale:g})",
                    timing,
                    scale_to_target=scale,
                )
            )
            del model
            gc.collect()

    order = {
        ("Softmax reference", "CPU (PyTorch)"): 0,
        ("Softmax reference", "Apple GPU via MPS (PyTorch)"): 1,
        ("SV RBF, re-stream active-set (legacy)", "CPU (all components)"): 2,
        ("SV RBF, persistent rank-one (legacy)", "CPU (all components)"): 3,
        ("SV hybrid, batched FISTA (current path)",
         "CPU (PyTorch/KKT) + Apple GPU (MLX FISTA)"): 4,
        ("SV RBF, batched FISTA (diagnostic)",
         "CPU (PyTorch/KKT) + Apple GPU (MLX FISTA)"): 4,
    }
    rows.sort(key=lambda row: order.get((row.method, row.device), 99))
    _print_paper_table(rows, step_budget=step_budget)

    print(
        "\nCaveats\n"
        "  - Rows intentionally identify different frameworks/devices. Ratio columns "
        "name the directly measured CPU and MPS softmax references; they are "
        "cross-framework/device comparisons, not same-stack speedups.\n"
        "  - Re-stream and persistent rows are NOT direct B=16 measurements. Their full-model "
        f"B={probe_batch} medians are scaled linearly by {PAPER_BATCH / probe_batch:g}x; "
        "non-gate model work need not scale linearly with batch. These historical "
        "active-set branches implement only the RBF readout, so they are not hybrid "
        "end-to-end measurements.\n"
        "  - MPS and MLX are synchronized before and after every timed sample. "
        "The mixed row includes host/device transfers, MLX FISTA, PyTorch KKT/readout, "
        "and backward.\n"
        "  - Softmax is the repository's unfused masked PyTorch implementation, "
        "not FlashAttention or another fused kernel.\n"
        "  - Wall-clock projections exclude optimizer updates, data loading, evaluation, "
        "checkpointing, and thermal/load variation."
    )


def bench_layer(
    layer: torch.nn.Module,
    B: int,
    T: int,
    d_model: int,
    *,
    n_warmup: int = 2,
    n_iter: int = 5,
) -> Timing:
    X = torch.randn(B, T, d_model)

    def forward() -> None:
        with torch.no_grad():
            layer(X)

    return _measure(forward, warmups=n_warmup, repetitions=n_iter)


def bench_persistent_stream(
    T: int,
    d: int,
    C: float,
    kpar: float,
    *,
    n_warmup: int = 2,
    n_iter: int = 5,
) -> Timing:
    """Achievable per-sequence gate cost: seed once, then stream T points with a
    single maintained solver (rank-1 updates), no per-chunk re-streaming."""
    rng = np.random.RandomState(0)
    X = rng.randn(T, d)
    n_seed = int(np.ceil(1.0 / C)) + 2
    def stream_once() -> None:
        solver = FastOneClassSVM(C=C, ktype="r", kpar=kpar)
        solver.seed_from_qp(X[:n_seed])
        for i in range(n_seed, T):
            solver.add_point(X[i])

    return _measure(stream_once, warmups=n_warmup, repetitions=n_iter)


def main(*, warmups: int = 2, repetitions: int = 5, seed: int = 0) -> None:
    """Historical forward-only CPU micro sweep, now with robust timing labels."""
    _seed_all(seed)
    d_model, n_heads, hd = 64, 4, 16
    C, kpar, chunk, B = 0.1, 2.0, 64, 4
    torch_version = str(torch.__version__)
    exact_framework = (
        f"PyTorch {torch_version} + NumPy {np.__version__} + "
        f"CVXPY {_package_version('cvxpy')}"
    )

    def make_layers():
        return {
            "softmax": CausalSoftmaxAttention(d_model, n_heads, hd),
            "linear": CausalLinearAttention(d_model, n_heads, hd),
            "deltanet": DeltaNet(d_model, n_heads, hd),
            "sv (v1 re-stream)": CausalSVAttention(d_model, n_heads, hd, C=C,
                                                   kpar=kpar, chunk=chunk),
        }

    for T in (128, 256, 512):
        print(f"\n=== T={T}  B={B}  d_model={d_model}  heads={n_heads}  chunk={chunk} ===")
        print(
            f"median of {repetitions} samples after {warmups} warmups; "
            "forward only\n"
            "| Method | Framework | Device | Measurement | median ms/fwd [IQR] | tokens/s |"
        )
        print("|---|---|---|---|---:|---:|")
        res: dict[str, float] = {}
        for name, layer in make_layers().items():
            timing = bench_layer(
                layer,
                B,
                T,
                d_model,
                n_warmup=warmups,
                n_iter=repetitions,
            )
            dt = timing.median_s
            tps = B * T / dt
            res[name] = tps
            framework = exact_framework if name == "sv (v1 re-stream)" else f"PyTorch {torch_version}"
            print(
                f"| {name} | {framework} | CPU | direct B={B} | "
                f"{dt * 1e3:,.1f} [{timing.q1_s * 1e3:,.1f}-"
                f"{timing.q3_s * 1e3:,.1f}] | {tps:,.1f} |"
            )

        solver_timing = bench_persistent_stream(
            T,
            hd,
            C,
            kpar,
            n_warmup=warmups,
            n_iter=repetitions,
        )
        dt_solver = solver_timing.median_s
        # B*H independent gates, each ~dt_solver; tokens processed = B*T
        achievable = B * T / (B * n_heads * dt_solver)
        estimated_fwd = B * n_heads * dt_solver
        print(
            f"| sv (persistent gate estimate) | NumPy {np.__version__} + "
            f"CVXPY {_package_version('cvxpy')} | CPU | extrapolated from one "
            f"maintained solver/head | {estimated_fwd * 1e3:,.1f} "
            f"[{solver_timing.q1_s * B * n_heads * 1e3:,.1f}-"
            f"{solver_timing.q3_s * B * n_heads * 1e3:,.1f}] | {achievable:,.1f} |"
        )

        sm = res["softmax"]
        print(f"  -> vs softmax: v1 {sm / res['sv (v1 re-stream)']:.0f}x slower; "
              f"persistent-est {sm / achievable:.0f}x slower")

    print("\nDecision guide: viable if the persistent-stream estimate is within "
          "~10x of softmax/deltanet (batch across heads to close more); "
          "descope to selection-only / inference-time if even persistent is >~20x.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--paper",
        action="store_true",
        help="run the robust full-model paper benchmark",
    )
    mode.add_argument(
        "--lm",
        action="store_true",
        help="compatibility alias for --paper",
    )
    ap.add_argument(
        "--warmups",
        type=int,
        default=None,
        help="warmup steps per row (paper default: 3; micro default: 2)",
    )
    ap.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="independent timed samples per row (paper default: 7; micro default: 5)",
    )
    ap.add_argument("--seed", type=int, default=0, help="NumPy/PyTorch/MLX RNG seed")
    ap.add_argument(
        "--sv-readout",
        choices=["softmax", "rbf"],
        default="softmax",
        help="current-path SV readout for --paper; softmax is the paper's hybrid model",
    )
    ap.add_argument(
        "--probe-batch",
        type=int,
        default=2,
        help="direct batch used for slow legacy rows before extrapolating to B=16",
    )
    ap.add_argument(
        "--step-budget",
        type=int,
        default=50_000,
        help="step count used only for wall-clock projections",
    )
    ap.add_argument(
        "--skip-legacy",
        action="store_true",
        help="skip slow re-stream/persistent rows (not for paper reporting)",
    )
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="override PyTorch CPU intra-op thread count",
    )
    args = ap.parse_args()
    if args.warmups is not None and args.warmups < 0:
        ap.error("--warmups must be nonnegative")
    if args.repetitions is not None and args.repetitions < 1:
        ap.error("--repetitions must be positive")
    if not 1 <= args.probe_batch <= PAPER_BATCH:
        ap.error(f"--probe-batch must be in [1, {PAPER_BATCH}]")
    if args.step_budget < 1:
        ap.error("--step-budget must be positive")
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            ap.error("--torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)

    if args.paper or args.lm:
        main_paper(
            warmups=3 if args.warmups is None else args.warmups,
            repetitions=7 if args.repetitions is None else args.repetitions,
            probe_batch=args.probe_batch,
            step_budget=args.step_budget,
            seed=args.seed,
            include_legacy=not args.skip_legacy,
            sv_readout=args.sv_readout,
        )
    else:
        main(
            warmups=2 if args.warmups is None else args.warmups,
            repetitions=5 if args.repetitions is None else args.repetitions,
            seed=args.seed,
        )
