"""P2 MIMIC flagship: next-6h ICU deterioration prediction, causal layers.

CausalSVAttention vs DeltaNet / softmax / linear + an LSTM clinical baseline, on
real MIMIC-IV ICU vital streams (local DUA-covered cohort). Reported as
AUROC/AUPRC over held-out (stay, hour) pairs. SV is expected to be competitive
(prediction needs local trend, which the chunk-frozen gate is not specialized
for). This is the downstream predictive control; selection and fixed-C deletion
are evaluated separately.

Run from the repository root:
    python -m experiments.p2_mimic
"""
from __future__ import annotations

import torch

from svattn.baselines import CausalSoftmaxAttention, CausalLinearAttention, DeltaNet
from svattn.causal_sv_attention import CausalSVAttention
from svattn.tasks.mimic import (load_cohort, next_h_labels, ClinicalModel,
                                LSTMLayer, train_clinical, eval_clinical)

D_MODEL, HEADS, HORIZON, T = 32, 2, 6, 56


def build(name):
    hd = D_MODEL // HEADS
    layer = {
        "lstm": lambda: LSTMLayer(D_MODEL),
        "softmax": lambda: CausalSoftmaxAttention(D_MODEL, HEADS, hd),
        "deltanet": lambda: DeltaNet(D_MODEL, HEADS, hd),
        "linear": lambda: CausalLinearAttention(D_MODEL, HEADS, hd),
        "sv": lambda: CausalSVAttention(D_MODEL, HEADS, hd, C=0.1, kpar=2.0, chunk=14),
    }[name]()
    return ClinicalModel(layer, n_feat=6, d_model=D_MODEL)


def main():
    X, det = load_cohort(t_fixed=T, n_max=400, seed=0)
    y, valid = next_h_labels(det, HORIZON)
    ntr = 250
    tr = (X[:ntr], y[:ntr], valid[:ntr])
    te = (X[ntr:], y[ntr:], valid[ntr:])
    pos = (y[:ntr] * valid[:ntr]).sum() / valid[:ntr].sum()
    print(f"MIMIC next-{HORIZON}h deterioration | train={ntr} test={len(X) - ntr} "
          f"T={T} pos_rate={pos:.2f}\n")
    print(f"{'model':<10}{'AUROC':>9}{'AUPRC':>9}", flush=True)
    for name in ("lstm", "softmax", "deltanet", "linear", "sv"):
        torch.manual_seed(0)
        m = build(name)
        train_clinical(m, *tr, steps=120, batch=8, lr=2e-3, seed=0)
        au, ap = eval_clinical(m, *te)
        print(f"{name:<10}{au:>9.3f}{ap:>9.3f}", flush=True)


if __name__ == "__main__":
    main()
