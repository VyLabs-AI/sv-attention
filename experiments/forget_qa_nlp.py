"""NLP-style "tell facts incl. a secret -> ask -> forget the secret -> ask again"
on a TRAINED causal SV-Attention recall model.

We use the paper's working recall format (MQAR / SeqRecallModel): each fact is one
token carrying a (key, value) pair, so retrieval is content-addressed and needs no
induction circuit (the separate-token version stalls because the chunk-frozen layer
can't form induction -- the documented blind spot). We name the memories as facts
(subject -> city) for readability. The query asks for one subject's city; the model
recalls it. We then remove that fact and solve the retained-key refit under the
same fixed C, whereas coefficient decay leaves residual readout influence.

Run from the repository root:
    python -m experiments.forget_qa_nlp --steps 500
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from cp_svm.oneclass_fast import FastOneClassSVM
from svattn.causal_sv_attention import CausalSVAttention
from svattn.sv_attention import rbf_gram
from svattn.tasks.mqar import make_mqar_batch, SeqRecallModel, train_recall, eval_recall

SUBJECTS = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"]
CITIES = ["paris", "tokyo", "cairo", "lima", "oslo", "delhi", "rome", "perth"]
D_K = D_V = 8
D_MODEL = 32
HEADS = 1
N_MEM = 8
CHUNK = 8
C, KPAR = 0.2, 2.0


def build():
    layer = CausalSVAttention(D_MODEL, HEADS, D_MODEL // HEADS, C=C, kpar=KPAR,
                              chunk=CHUNK, solver="mlx", fista_iters=120, readout="rbf")
    return SeqRecallModel(layer, d_in=D_K + D_V, d_model=D_MODEL, d_v=D_V)


@torch.no_grad()
def query_value(model, X, qpos, forget_m=None, decay=None):
    """Predicted value vector at query position `qpos`, reading the gate fit on the
    N_MEM memory tokens; `forget_m` removes one row and refits under the same C,
    while `decay`=(m, gamma) scales its existing coefficient."""
    layer = model.layer
    emb = model.embed(X)                                  # (1, T, d_model)
    K = layer.Wk(emb)[0]; Q = layer.Wq(emb)[0]; V = layer.Wv(emb)[0]   # heads=1
    Kp, Vp, q = K[:N_MEM], V[:N_MEM], Q[qpos]
    Knp = Kp.detach().numpy().astype(np.float64)
    if forget_m is not None:
        keep = [i for i in range(N_MEM) if i != forget_m]
        s = FastOneClassSVM(C=C, ktype="r", kpar=KPAR); s.seed_from_qp(Knp[keep])
        alpha = np.zeros(N_MEM); alpha[keep] = s.alpha
    else:
        s = FastOneClassSVM(C=C, ktype="r", kpar=KPAR); s.seed_from_qp(Knp)
        alpha = np.array(s.alpha)
        if decay is not None:
            alpha = alpha.copy(); alpha[decay[0]] *= decay[1]
    w = rbf_gram(q[None, :], Kp, KPAR)[0] * torch.from_numpy(alpha).float()
    o = (w @ Vp) / w.sum().clamp_min(1e-8)
    return model.readout(layer.Wo(o[None, :]))[0]         # (d_v,)


def predicted_mem(pred, vbank):
    return int(((vbank - pred[None, :]) ** 2).sum(-1).argmin())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--cache", default="outputs/qa_nlp_model.pt")
    ap.add_argument("--fig", default="outputs/forget_qa_nlp.png")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = build()
    if os.path.exists(args.cache) and not args.retrain:
        model.load_state_dict(torch.load(args.cache, weights_only=True)); model.eval()
        print(f"[loaded {args.cache}]")
    else:
        print(f"training causal SV recall model ({args.steps} steps)...")
        train_recall(model, n_mem=N_MEM, n_query=4, d_k=D_K, d_v=D_V,
                     steps=args.steps, batch=16, lr=3e-3, seed=0)
        model.eval()
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.cache)

    acc = eval_recall(model, n_mem=N_MEM, n_query=4, d_k=D_K, d_v=D_V, n_eval=400)
    print(f"\nheld-out recall@1: {acc:.3f}  (chance = {1.0/N_MEM:.3f})")
    if acc < 0.6:
        print("recall not learned; increase --steps. Skipping figure.")
        return

    rng = np.random.RandomState(123)
    rates = {"told": [], "retained_refit": [], "decay0.1": [], "decay0.01": []}
    for _ in range(args.trials):
        X, _t, vbank, qidx = make_mqar_batch(1, N_MEM, 1, D_K, D_V, rng)
        secret = int(qidx[0, 0]); qpos = N_MEM            # the single query
        vb = vbank[0]

        def asks(**kw):
            return int(predicted_mem(query_value(model, X, qpos, **kw), vb) == secret)
        rates["told"].append(asks())
        rates["retained_refit"].append(asks(forget_m=secret))
        rates["decay0.1"].append(asks(decay=(secret, 0.1)))
        rates["decay0.01"].append(asks(decay=(secret, 0.01)))
    agg = {k: float(np.mean(v)) for k, v in rates.items()}
    print("\nsecret-recall rate (trained causal SV model):")
    for k in ("told", "retained_refit", "decay0.1", "decay0.01"):
        print(f"  {k:<10} {agg[k]:.2f}")

    # one transcript instance (relabel memories as subject->city facts)
    rng2 = np.random.RandomState(7)
    X, _t, vbank, qidx = make_mqar_batch(1, N_MEM, 1, D_K, D_V, rng2)
    secret = int(qidx[0, 0]); vb = vbank[0]
    names = list(zip(SUBJECTS[:N_MEM], CITIES[:N_MEM]))
    before = predicted_mem(query_value(model, X, N_MEM), vb)
    after = predicted_mem(query_value(model, X, N_MEM, forget_m=secret), vb)
    render(names, secret, before, after, agg, args.fig)
    print(f"saved figure -> {args.fig}")


def render(names, secret, before, after, agg, path):
    import matplotlib.pyplot as plt
    from svattn import figstyle
    figstyle.apply()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 3.6),
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    axA.axis("off")
    figstyle.caps_title(axA, "(a) tell facts incl. a secret, forget it, ask again")
    others = [f"{s}\u2192{c}" for i, (s, c) in enumerate(names) if i != secret]
    ohalf = len(others) // 2
    others1 = ",  ".join(others[:ohalf]); others2 = ",  ".join(others[ohalf:])
    ssub, scity = names[secret]
    rows = [("FACTS told to the model:", figstyle.INK, 0.94, "bold", 10.5),
            (f"   secret:  {ssub} \u2192 {scity}", figstyle.BLUE, 0.85, "bold", 10.5),
            (f"   {others1}", figstyle.DARKGRAY, 0.77, "normal", 9.5),
            (f"   {others2}", figstyle.DARKGRAY, 0.70, "normal", 9.5),
            (f"ASK:  where is {ssub} ?", figstyle.INK, 0.56, "bold", 10.5),
            (f"BEFORE forgetting:   \u201c{names[before][1]}\u201d   "
             f"\u2713 recalls the secret", figstyle.BLUE, 0.41, "bold", 10.5),
            (f"AFTER retained refit: \u201c{names[after][1]}\u201d   \u2717 target absent",
             figstyle.RED, 0.29, "bold", 10.5),
            ("   retrieves a neighbor under the retained-key reference",
             figstyle.GRAY, 0.21, "normal", 9.5),
            (f"AFTER decay \u03b3=0.01: still answers the secret {agg['decay0.01']*100:.0f}% "
             "of the time", figstyle.DARKGRAY, 0.09, "normal", 9.5)]
    for txt, col, y, wt, fs in rows:
        axA.text(0.0, y, txt, color=col, fontsize=fs, fontweight=wt,
                 family="monospace", transform=axA.transAxes)

    order = ["told", "retained_refit", "decay0.1", "decay0.01"]
    vals = [agg[k] for k in order]
    cols = [figstyle.BLUE, figstyle.RED, figstyle.GRAY, figstyle.DARKGRAY]
    axB.bar(range(len(order)), vals, color=cols)
    axB.set_xticks(range(len(order)))
    axB.set_xticklabels(["told", "retained\nrefit", "decay\n0.1", "decay\n0.01"], fontsize=9)
    axB.set_ylabel("answers the secret (rate)")
    axB.set_ylim(0, 1.05)
    figstyle.caps_title(axB, "(b) does the model still answer the secret?")
    for i, v in enumerate(vals):
        axB.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    main()
