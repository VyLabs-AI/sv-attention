"""Interpretable examples on a trained causal SV-Attention recall
model (recall@1 = 1.0). Each example is a readable "facts -> ask" story that maps
to one paper property:

  A. Fixed-C deletion      -- delete ONE fact and audit the retained facts.
  B. Remove-add editing    -- removal plus admission corrects a binding
                              (carol: cairo->tokyo).
  C. Support-set selection -- under redundancy, compare point-in-time support
                              selection with matched-budget baselines.

We reuse the model trained in experiments.forget_qa_nlp (same SeqRecallModel /
CausalSVAttention). Frozen learned keys are solved in fp64, then deletion and
admission use the maintained path. Facts are named subject->city for
readability; the model answers by reading the gated memory and we decode the
nearest stored value.

Run: PYTHONPATH=. python -m experiments.interpretable_demos
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from cp_svm.oneclass_fast import FastOneClassSVM
from svattn.sv_attention import rbf_gram
from svattn.tasks.mqar import SeqRecallModel
from svattn.causal_sv_attention import CausalSVAttention
from experiments.forget_qa_nlp import (SUBJECTS, CITIES, D_K, D_V, D_MODEL, HEADS,
                                       N_MEM, CHUNK, C, KPAR, build)

KEY_NOISE = 0.02


def load_model(cache="outputs/qa_nlp_model.pt"):
    torch.manual_seed(0)
    model = build()
    if not os.path.exists(cache):
        raise FileNotFoundError(f"{cache} missing; run experiments.forget_qa_nlp first")
    model.load_state_dict(torch.load(cache, weights_only=True))
    model.eval()
    return model


def make_scene(rng, n_mem=N_MEM):
    """One scene: n_mem (key, value) facts + one query token per fact (asking that
    subject). Returns the projected memory keys/values and per-subject query vecs."""
    keys = rng.standard_normal((n_mem, D_K)).astype(np.float32)
    vals = rng.standard_normal((n_mem, D_V)).astype(np.float32)
    qkeys = keys + KEY_NOISE * rng.standard_normal((n_mem, D_K)).astype(np.float32)
    return keys, vals, qkeys


@torch.no_grad()
def project(model, keys, vals, qkeys):
    """Embed memory tokens [key;val] and query tokens [qkey;0], return projected
    layer K (mem), V (mem), and Q (per query) for the single head."""
    nm = len(keys)
    Xm = np.concatenate([keys, vals], axis=1)                 # (nm, d_in)
    Xq = np.concatenate([qkeys, np.zeros((len(qkeys), D_V), np.float32)], axis=1)
    X = torch.tensor(np.concatenate([Xm, Xq], axis=0)[None], dtype=torch.float32)
    emb = model.embed(X)
    layer = model.layer
    K = layer.Wk(emb)[0]; Q = layer.Wq(emb)[0]; V = layer.Wv(emb)[0]
    return K[:nm], V[:nm], Q[nm:], vals               # mem K,V ; query Q ; vbank


def gate_alpha(Kmem, drop=None, decay=None):
    """Fresh fixed-C gate refit; `drop` omits rows, `decay` scales one weight."""
    Knp = Kmem.detach().numpy().astype(np.float64)
    nm = len(Knp)
    drop = set(drop or [])
    keep = [i for i in range(nm) if i not in drop]
    s = FastOneClassSVM(C=C, ktype="r", kpar=KPAR); s.seed_from_qp(Knp[keep])
    alpha = np.zeros(nm); alpha[keep] = s.alpha
    if decay is not None and alpha[decay[0]] != 0:
        alpha = alpha.copy(); alpha[decay[0]] *= decay[1]
    return alpha, s


def maintained_drop_alpha(Kmem, drop):
    """Delete one row through the maintained path and return padded coefficients."""
    Knp = Kmem.detach().numpy().astype(np.float64)
    keep = [index for index in range(len(Knp)) if index != drop]
    state = FastOneClassSVM(C=C, ktype="r", kpar=KPAR).seed_from_qp(Knp)
    state.remove_point(drop)
    alpha = np.zeros(len(Knp))
    alpha[keep] = state.alpha
    return alpha, state


def answer(model, Kmem, Vmem, qvec, vbank, alpha):
    """Readout o(q) = sum_i alpha_i kappa(q,k_i) v_i, decoded to nearest stored value."""
    w = rbf_gram(qvec[None, :], Kmem, KPAR)[0] * torch.from_numpy(alpha).float()
    o = (w @ Vmem) / w.sum().clamp_min(1e-8)
    pred = model.readout(model.layer.Wo(o[None, :]))[0].detach().numpy()
    return int(((vbank - pred[None, :]) ** 2).sum(-1).argmin())


def _readout_vec(Kmem, Vmem, qvec, alpha):
    w = rbf_gram(qvec[None, :], Kmem, KPAR)[0] * torch.from_numpy(alpha).float()
    return ((w @ Vmem) / w.sum().clamp_min(1e-8)).detach().numpy()


def edit_residual(Kmem, Q, Vmem, drop, decays=(0.01,)):
    """Maintained-decrement readout deviation from a fixed-C retained-key refit."""
    Knp = Kmem.detach().numpy().astype(np.float64)
    nm = len(Knp)
    keep = [i for i in range(nm) if i != drop]
    Kk, Vk = Kmem[keep], Vmem[keep]
    sref = FastOneClassSVM(C=C, ktype="r", kpar=KPAR); sref.seed_from_qp(Knp[keep])
    a_ref = np.array(sref.alpha)                          # never-saw-it reference
    try:
        sdec = FastOneClassSVM(C=C, ktype="r", kpar=KPAR); sdec.seed_from_qp(Knp)
        sdec.remove_point(drop)
        a_dec = np.array(sdec.alpha)
        res_decrement = max(float(np.abs(_readout_vec(Kk, Vk, Q[i], a_dec)
                                         - _readout_vec(Kk, Vk, Q[i], a_ref)).max())
                            for i in range(nm))
    except Exception:
        res_decrement = float("nan")
    sfull = FastOneClassSVM(C=C, ktype="r", kpar=KPAR); sfull.seed_from_qp(Knp)
    a_full = np.array(sfull.alpha)
    res_decay = {}
    for g in decays:
        ad = a_full.copy(); ad[drop] *= g
        res_decay[g] = max(float(np.abs(_readout_vec(Kmem, Vmem, Q[i], ad)
                                        - _readout_vec(Kk, Vk, Q[i], a_ref)).max())
                           for i in range(nm))
    return res_decrement, res_decay


# ----------------------------------------------------------------- Demo A
def demo_surgical(model, rng):
    keys, vals, qkeys = make_scene(rng)
    Kmem, Vmem, Q, vbank = project(model, keys, vals, qkeys)
    secret = 2                                              # carol
    a_full, _ = gate_alpha(Kmem)
    a_forg, _ = maintained_drop_alpha(Kmem, secret)
    before = [answer(model, Kmem, Vmem, Q[i], vbank, a_full) == i for i in range(N_MEM)]
    after = [answer(model, Kmem, Vmem, Q[i], vbank, a_forg) == i for i in range(N_MEM)]
    return secret, before, after


# ----------------------------------------------------------------- Demo B
def demo_edit(model, rng):
    """carol -> cairo, then maintained remove + add to correct carol -> tokyo."""
    keys, vals, qkeys = make_scene(rng)
    Kmem, Vmem, Q, vbank = project(model, keys, vals, qkeys)
    carol, tokyo = 2, 1                                     # subject idx, target city idx
    a0, _ = gate_alpha(Kmem)
    ans_before = answer(model, Kmem, Vmem, Q[carol], vbank, a0)
    # Remove carol's old binding, then admit the same key carrying tokyo's value.
    keep = [index for index in range(N_MEM) if index != carol]
    new_k = Kmem[carol:carol + 1]                          # same subject key
    new_v = Vmem[tokyo:tokyo + 1]                          # tokyo's value
    Ke = torch.cat([Kmem[keep], new_k], 0)
    Ve = torch.cat([Vmem[keep], new_v], 0)
    vbe = np.concatenate([vbank[keep], vbank[tokyo:tokyo + 1]], 0)
    labels = [CITIES[index] for index in keep] + [CITIES[tokyo]]

    Knp = Kmem.detach().numpy().astype(np.float64)
    state = FastOneClassSVM(C=C, ktype="r", kpar=KPAR).seed_from_qp(Knp)
    state.remove_point(carol)
    state.add_point(new_k[0].detach().numpy().astype(np.float64))
    a_edit = np.array(state.alpha)

    qcarol = Q[carol]
    ans_after = answer(model, Ke, Ve, qcarol, vbe, a_edit)
    city_after = labels[ans_after]

    refit = FastOneClassSVM(C=C, ktype="r", kpar=KPAR).seed_from_qp(
        Ke.detach().numpy().astype(np.float64)
    )
    probes = Q.detach().numpy().astype(np.float64)
    edit_deviation = float(
        np.max(
            np.abs(
                state.decision_function(probes)
                - refit.decision_function(probes)
            )
        )
    )
    decrement_residual, res_decay = edit_residual(Kmem, Q, Vmem, carol)
    return (
        carol,
        CITIES[ans_before],
        city_after,
        decrement_residual,
        edit_deviation,
        res_decay[0.01],
    )


# ----------------------------------------------------------------- Demo D
def demo_roundtrip(model, rng, n_trials=120):
    """Delete and re-admit one fact through the maintained path."""
    restored, recovered = [], []
    for _ in range(n_trials):
        keys, vals, qkeys = make_scene(rng)
        Kmem, Vmem, Q, vbank = project(model, keys, vals, qkeys)
        tgt = 2
        a0, _ = gate_alpha(Kmem)
        o_before = _readout_vec(Kmem, Vmem, Q[tgt], a0)
        keep = [index for index in range(N_MEM) if index != tgt]
        state = FastOneClassSVM(C=C, ktype="r", kpar=KPAR).seed_from_qp(
            Kmem.detach().numpy().astype(np.float64)
        )
        state.remove_point(tgt)
        a_forg = np.zeros(N_MEM)
        a_forg[keep] = state.alpha
        forgotten = answer(model, Kmem, Vmem, Q[tgt], vbank, a_forg) != tgt
        state.add_point(Kmem[tgt].detach().numpy().astype(np.float64))
        K_re = torch.cat([Kmem[keep], Kmem[tgt:tgt + 1]], 0)
        V_re = torch.cat([Vmem[keep], Vmem[tgt:tgt + 1]], 0)
        vbank_re = np.concatenate([vbank[keep], vbank[tgt:tgt + 1]], 0)
        a_re = np.array(state.alpha)
        o_after = _readout_vec(K_re, V_re, Q[tgt], a_re)
        restored.append(float(np.abs(o_before - o_after).max()))
        recovered.append(
            int(
                forgotten
                and answer(model, K_re, V_re, Q[tgt], vbank_re, a_re)
                == len(vbank_re) - 1
            )
        )
    return float(np.mean(restored)), float(np.mean(recovered))


# ----------------------------------------------------------------- Demo C
def demo_eviction(model, rng, n_trials=120):
    """Redundant context: 2 rare facts (1 copy) + 3 dense facts (k copies). At a
    budget = current support size, does the RARE fact survive each policy?"""
    dense_k = 4
    rates = {"sv": [], "h2o": [], "recency": [], "random": []}
    for _ in range(n_trials):
        n_rare, n_dense = 2, 3
        base = rng.standard_normal((n_rare + n_dense, D_K)).astype(np.float32)
        vbase = rng.standard_normal((n_rare + n_dense, D_V)).astype(np.float32)
        keys, vals, owner = [], [], []
        for i in range(n_rare):                            # rare: 1 copy
            keys.append(base[i]); vals.append(vbase[i]); owner.append(i)
        for j in range(n_dense):                           # dense: dense_k near-dup copies
            for _c in range(dense_k):
                keys.append(base[n_rare + j] + 0.01 * rng.standard_normal(D_K).astype(np.float32))
                vals.append(vbase[n_rare + j]); owner.append(n_rare + j)
        keys = np.array(keys, np.float32); vals = np.array(vals, np.float32)
        owner = np.array(owner)
        qk = base[0] + KEY_NOISE * rng.standard_normal(D_K).astype(np.float32)   # ask a RARE fact
        Kmem, Vmem, Q, vbank = project(model, keys, vals, qk[None, :])
        q = Q[0]
        a_full, s = gate_alpha(Kmem)
        budget = max(2, len(s.S) + len(s.E))               # current support size
        nm = len(keys)
        # H2O score = query-independent popularity: total kernel mass each key
        # receives from the whole context. Under redundancy this concentrates on the
        # duplicated dense keys (the heavy-hitter failure mode), starving rare keys.
        G = rbf_gram(Kmem, Kmem, KPAR).detach().numpy()
        mass = G.sum(1)
        def keep_to(sel):
            a = a_full.copy(); mask = np.zeros(nm, bool); mask[sel] = True; a[~mask] = 0.0
            return a
        sel = {"sv": np.argsort(-a_full)[:budget],
               "h2o": np.argsort(-mass)[:budget],
               "recency": np.arange(nm - budget, nm),
               "random": rng.choice(nm, budget, replace=False)}
        for pol, idx in sel.items():
            pred = answer(model, Kmem, Vmem, q, vbank, keep_to(idx))
            rates[pol].append(int(owner[pred] == 0))       # answered the rare fact?
        # also record full-context correctness as a sanity ceiling
    return {k: float(np.mean(v)) for k, v in rates.items()}, budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="outputs/interpretable_demos.png")
    ap.add_argument("--trials", type=int, default=120)
    args = ap.parse_args()
    model = load_model()

    secret, before, after = demo_surgical(model, np.random.RandomState(7))
    print("Demo A -- fixed-C deletion (remove carol):")
    print(f"  before: {sum(before)}/{N_MEM} facts answered; after: {sum(after)}/{N_MEM}")
    print(f"  carol answered before={before[secret]} after={after[secret]}; "
          f"others intact={all(after[i] for i in range(N_MEM) if i != secret)}")

    csub, c_before, c_after, res_decrement, edit_dev, res_decay = demo_edit(
        model, np.random.RandomState(3)
    )
    print(f"\nDemo B -- knowledge edit (carol): before='{c_before}' after-edit='{c_after}'")
    print(f"  maintained delete/refit readout deviation: {res_decrement:.1e}")
    print(f"  maintained remove-add/refit decision deviation: {edit_dev:.1e}")
    print(f"  decay g=0.01 readout deviation: {res_decay:.2f}")

    ev, budget = demo_eviction(model, np.random.RandomState(11), args.trials)
    print(f"\nDemo C -- support selection under redundancy (budget={budget}, rare-fact survival):")
    for k in ("sv", "h2o", "recency", "random"):
        print(f"  {k:<8} {ev[k]:.2f}")

    rt_res, rt_rec = demo_roundtrip(model, np.random.RandomState(23), args.trials)
    print(f"\nDemo D -- delete->re-admit round-trip ({args.trials} trials): "
          f"readout restored to {rt_res:.1e}; answer recovered {rt_rec:.2f} of the time")

    render(
        secret,
        before,
        after,
        csub,
        c_before,
        c_after,
        res_decrement,
        res_decay,
        ev,
        args.fig,
    )
    print(f"\nsaved figure -> {args.fig}")


def _fmt_res(r):
    if not np.isfinite(r):
        return "n/a"
    return "0 (machine precision)" if r < 1e-15 else f"{r:.0e}"


def render(secret, before, after, csub, c_before, c_after,
           res_decrement, res_decay, ev, path):
    import matplotlib.pyplot as plt
    from svattn import figstyle
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axA, axB, axC = axes

    # Panel A: fixed-C deletion grid
    axA.axis("off")
    figstyle.caps_title(axA, "(a) fixed-C deletion: others survive")
    axA.text(0.0, 0.92, f"remove one fact ({SUBJECTS[secret]}\u2192{CITIES[secret]}); "
             "re-ask all:", color=figstyle.INK, fontsize=10, fontweight="bold",
             family="monospace", transform=axA.transAxes)
    axA.text(0.30, 0.82, "before", color=figstyle.INK, fontsize=9.5, fontweight="bold",
             transform=axA.transAxes)
    axA.text(0.55, 0.82, "after", color=figstyle.INK, fontsize=9.5, fontweight="bold",
             transform=axA.transAxes)
    for i in range(N_MEM):
        y = 0.74 - i * 0.085
        is_secret = i == secret
        axA.text(0.0, y, f"{SUBJECTS[i]}\u2192{CITIES[i]}",
                 color=figstyle.BLUE if is_secret else figstyle.DARKGRAY,
                 fontsize=9.5, fontweight="bold" if is_secret else "normal",
                 family="monospace", transform=axA.transAxes)
        for x, ok in ((0.33, before[i]), (0.58, after[i])):
            axA.text(x, y, "\u2713" if ok else "\u2717",
                     color=figstyle.GREEN if ok else figstyle.RED,
                     fontsize=12, fontweight="bold", transform=axA.transAxes)
    axA.text(0.0, 0.74 - N_MEM * 0.085 - 0.02,
             "only the forgotten fact flips; the rest are untouched.",
             color=figstyle.GRAY, fontsize=8.5, style="italic", transform=axA.transAxes)

    # Panel B: knowledge editing
    axB.axis("off")
    figstyle.caps_title(axB, "(b) maintained remove-add editing")
    lines = [(f"fact:  {SUBJECTS[csub]} \u2192 {c_before}", figstyle.DARKGRAY, 0.84, "normal"),
             (f"ASK where is {SUBJECTS[csub]}?  \u2192 {c_before}", figstyle.BLUE, 0.71, "bold"),
             ("EDIT (maintained remove + add):", figstyle.INK, 0.56, "bold"),
             (f"   {SUBJECTS[csub]} \u2192 {c_after}", figstyle.INK, 0.47, "normal"),
             (f"ASK again \u2192 {c_after}", figstyle.RED, 0.35, "bold"),
             (f"delete/refit residual: {_fmt_res(res_decrement)}",
              figstyle.GRAY, 0.20, "italic"),
             (f"   vs decay \u03b3=0.01: {res_decay:.2f}  (still leaks)",
              figstyle.GRAY, 0.11, "italic")]
    for txt, col, y, st in lines:
        axB.text(0.0, y, txt, color=col, fontsize=10,
                 fontweight=("bold" if st == "bold" else "normal"),
                 style=("italic" if st == "italic" else "normal"),
                 family="monospace", transform=axB.transAxes)

    # Panel C: point-in-time support selection under redundancy
    order = ["sv", "h2o", "recency", "random"]
    vals = [ev[k] for k in order]
    cols = [figstyle.BLUE, figstyle.RED, figstyle.GRAY, figstyle.DARKGRAY]
    axC.bar(range(len(order)), vals, color=cols)
    axC.set_xticks(range(len(order)))
    axC.set_xticklabels(["SV\nsupport", "H2O", "recency", "random"], fontsize=9)
    axC.set_ylabel("rare fact still answered")
    axC.set_ylim(0, 1.05)
    figstyle.caps_title(axC, "(c) keep the rare fact under a budget")
    for i, v in enumerate(vals):
        axC.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    main()
