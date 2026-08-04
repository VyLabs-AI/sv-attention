"""Matched-state byte-level language-model quality experiments.

Trains a pluggable CharGPT and reports best validation bits per character.
SV's effective state (mean active |S u E|) sets the matched sliding-window
Transformer budget. Full softmax is an unbounded reference; linear attention
and DeltaNet carry their native fixed states.

Run from the repository root:
    python -m experiments.a2_quality
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from svattn.causal_sv_attention import CausalSVAttention
from svattn.nanogpt import CharGPT, GPTConfig, Enwik8, batch_iter
from svattn.nanogpt.data import LN2

# order matters for the run loop: sv first (sets matched window + saves model),
# fast baselines next, deltanet last (its naive token loop is the slow pole).
VARIANTS = ["sv", "swa", "linear", "softmax", "deltanet"]
NOTES = {"deltanet": "native (d_k x d_v)", "linear": "native (d_k)",
         "softmax": "full attn (upper bound)"}


@torch.no_grad()
def eval_bpc(model, data, block, batch, n_batches, rng):
    model.eval()
    it = batch_iter(data, block, batch, rng)
    tot = 0.0
    for _ in range(n_batches):
        x, y = next(it)
        tot += float(model(x, y)[1])
    return tot / n_batches / LN2


@torch.no_grad()
def measure_state(model, data, block, batch, n_batches, rng):
    """Mean active |S u E| across the SV layers (the effective token state)."""
    sv_layers = [m for m in model.modules() if isinstance(m, CausalSVAttention)]
    for m in sv_layers:
        m.record_state, m.state_log = True, []
    model.eval()
    it = batch_iter(data, block, batch, rng)
    for _ in range(n_batches):
        model(next(it)[0])
    logs = [v for m in sv_layers for v in m.state_log]
    for m in sv_layers:
        m.record_state = False
    return float(np.mean(logs)) if logs else float("nan")


def train(model, opt, sched, ds, *, steps, batch, block, n_eval, eval_every, seed,
          tag="", start=0, best=float("inf"), save_cb=None, ckpt_every=500):
    """Train from `start` to `steps`, periodically checkpointing via save_cb(step,
    best). Data windows are drawn i.i.d., so resuming with a fresh stream is fine."""
    it = batch_iter(ds.train, block, batch, np.random.default_rng(seed))
    ev_rng = np.random.default_rng(12345)
    t0 = time.perf_counter()
    for step in range(start + 1, steps + 1):
        model.train()
        x, y = next(it)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            best = min(best, eval_bpc(model, ds.val, block, batch, n_eval, ev_rng))
            print(f"    [{tag:9s}] step {step:5d}/{steps}  train {float(loss)/LN2:.3f} "
                  f"val-best {best:.3f} bpc  ({time.perf_counter()-t0:.0f}s)", flush=True)
        if save_cb is not None and step % ckpt_every == 0 and step != steps:
            save_cb(step, best)
    return best


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_model(name, cfg, sv_kw, window):
    if name == "sv":
        return CharGPT(cfg, attn_name="sv", **sv_kw)
    if name == "swa":
        return CharGPT(cfg, attn_name="swa", window=window)
    return CharGPT(cfg, attn_name=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_bytes", type=int, default=2_000_000)
    ap.add_argument("--data", default=None,
                    help="path to a byte-level corpus (default: enwik8); e.g. a modern "
                         "text file to show the matched-state result is not corpus-specific")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--solver", choices=["mlx", "exact"], default="mlx",
                    help="SV gate solver: 'mlx' (batched GPU, fast) or 'exact' (CPU)")
    ap.add_argument("--fista_iters", type=int, default=80)
    ap.add_argument("--readout", choices=["rbf", "softmax"], default="rbf",
                    help="SV readout: 'rbf' (distance) or 'softmax' (hybrid: gated "
                         "long-range + local causal window)")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="subset of variants to run (default: all). 'sv' must be "
                         "included to set the matched swa window.")
    ap.add_argument("--ckpt", default="outputs/a2_ckpt.pt")
    ap.add_argument("--save_sv", default="outputs/sv_model.pt",
                    help="save the trained SV model for frozen-key audits")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    args = ap.parse_args()

    ds = Enwik8(n_bytes=args.n_bytes, path=args.data)
    cfg = GPTConfig(vocab=ds.vocab, d_model=args.d_model, n_heads=args.n_heads,
                    n_layers=args.n_layers, block=args.block)
    tk = dict(steps=args.steps, batch=args.batch, block=args.block,
              eval_every=args.eval_every, n_eval=args.n_eval, seed=args.seed,
              ckpt_every=args.ckpt_every)
    sv_kw = dict(C=0.25, kpar=2.0, chunk=args.chunk, solver=args.solver,
                 persistent=(args.solver == "exact"), fista_iters=args.fista_iters,
                 readout=args.readout)
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    ck = {}
    if os.path.exists(args.ckpt) and not args.fresh:
        ck = torch.load(args.ckpt, weights_only=False)
        print(f"[resume] checkpoint at variant idx {ck.get('vi')} "
              f"step {ck.get('step')} ({args.ckpt})", flush=True)
    results = ck.get("results", {})
    window = ck.get("window")
    sv_state = ck.get("sv_state")
    nparams = ck.get("nparams")

    print(f"=== Matched-state language-model quality: {ds} ===")
    print(f"    CharGPT d_model={cfg.d_model} heads={cfg.n_heads} layers={cfg.n_layers} "
          f"block={cfg.block}  steps={args.steps} batch={args.batch} chunk={args.chunk} "
          f"solver={args.solver}\n", flush=True)

    run_variants = [v for v in VARIANTS if args.variants is None or v in args.variants]
    if "sv" not in run_variants:                            # sv sets the matched window
        run_variants = ["sv"] + run_variants
    for vi in range(ck.get("vi", 0), len(run_variants)):
        name = run_variants[vi]
        torch.manual_seed(args.seed)
        model = build_model(name, cfg, sv_kw, window)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                weight_decay=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        start, best = 0, float("inf")
        if ck.get("vi") == vi and ck.get("model") is not None:   # resume this variant
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start, best = ck["step"], ck["best"]
        if name == "sv":
            nparams = model.n_params()

        def save_cb(step, best, _vi=vi, _model=model, _opt=opt, _sched=sched):
            _atomic_save({"vi": _vi, "step": step, "best": best, "results": results,
                          "window": window, "sv_state": sv_state, "nparams": nparams,
                          "model": _model.state_dict(), "opt": _opt.state_dict(),
                          "sched": _sched.state_dict()}, args.ckpt)

        bpc = train(model, opt, sched, ds, **tk, tag=name, start=start, best=best,
                    save_cb=save_cb)

        if name == "sv":
            sv_state = measure_state(model, ds.val, args.block, args.batch, 10,
                                     np.random.default_rng(999))
            if not np.isfinite(sv_state):                    # no gated boundary (block<=chunk)
                sv_state = float(args.chunk)
            # hybrid also keeps a local causal window of size `chunk`, so the fair
            # matched budget for swa is the gated long-range state + the local window.
            local = args.chunk if args.readout == "softmax" else 0
            window = max(1, round(sv_state) + local)
            note = f"state {sv_state:.0f}" + (f"+local{local}" if local else "")
            results["sv"] = [bpc, note]
            _atomic_save({"model": model.state_dict(), "cfg": vars(cfg),
                          "sv_kw": sv_kw, "state": sv_state, "val_bpc": bpc},
                         args.save_sv)
            print(f"  sv: val {bpc:.3f} bpc  | {note} -> swa window {window}  "
                  f"(saved {args.save_sv})", flush=True)
        else:
            note = f"window {window} (matched)" if name == "swa" else NOTES[name]
            results[name] = [bpc, note]
            print(f"  {name:9s} val {bpc:.3f} bpc  | {note}", flush=True)
        # advance the checkpoint to the next variant (clears model state)
        _atomic_save({"vi": vi + 1, "step": 0, "best": float("inf"), "results": results,
                      "window": window, "sv_state": sv_state, "nparams": nparams,
                      "model": None, "opt": None, "sched": None}, args.ckpt)

    print(f"\n=== Matched-state summary ({(nparams or 0)/1e6:.2f}M params, seed {args.seed}) ===")
    print(f"{'variant':>10} {'val bpc':>9}  state/note")
    for name in run_variants:
        bpc, note = results[name]
        print(f"{name:>10} {bpc:>9.3f}  {note}")
    cands = [n for n in ("swa", "deltanet", "linear") if n in results]
    if cands and "sv" in results:
        ttl = min(results[n][0] for n in cands)
        gap = 100.0 * (results["sv"][0] - ttl) / ttl
        bar = "MEETS bar (<=2%)" if gap <= 2.0 else "BELOW bar (>2%)"
        print(f"\nSV vs best of {cands}: {gap:+.2f}%  =>  {bar}")
    if "swa" in results and "sv" in results:
        g = 100.0 * (results["sv"][0] - results["swa"][0]) / results["swa"][0]
        print(f"SV vs matched-state swa: {g:+.2f}%")


if __name__ == "__main__":
    main()
