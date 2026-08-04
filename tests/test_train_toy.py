"""End-to-end learning smoke test for Support Vector Attention.

Proves the implicit-gradient pipeline is usable for optimization, not merely
numerically correct: a student SV-attention layer with learnable key/value/query
projections is trained by Adam (gradients flowing through the max-margin solve)
to match a teacher SV-attention mapping. Training loss must drop substantially.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpy")

from svattn import SVAttention


def test_sv_attention_trains_to_fit_teacher():
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    n, m, d_in, d, dv = 14, 8, 5, 3, 3

    T = torch.tensor(rng.randn(n, d_in), dtype=torch.float64)        # context tokens
    Tq = torch.tensor(rng.randn(m, d_in), dtype=torch.float64)       # query tokens
    layer = SVAttention(C=0.3, kpar=2.0)

    def project(T_, Tq_, Wk, Wv, Wq):
        return T_ @ Wk, T_ @ Wv, Tq_ @ Wq

    # Teacher: fixed random projections generate the target mapping.
    Wk_t = torch.tensor(rng.randn(d_in, d), dtype=torch.float64)
    Wv_t = torch.tensor(rng.randn(d_in, dv), dtype=torch.float64)
    Wq_t = torch.tensor(rng.randn(d_in, d), dtype=torch.float64)
    with torch.no_grad():
        Xk, Vv, Q = project(T, Tq, Wk_t, Wv_t, Wq_t)
        Y = layer(Xk, Vv, Q)

    # Student: different init, learnable.
    Wk = torch.tensor(rng.randn(d_in, d) * 0.8, dtype=torch.float64, requires_grad=True)
    Wv = torch.tensor(rng.randn(d_in, dv) * 0.8, dtype=torch.float64, requires_grad=True)
    Wq = torch.tensor(rng.randn(d_in, d) * 0.8, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([Wk, Wv, Wq], lr=0.03)

    losses = []
    for _ in range(200):
        opt.zero_grad()
        Xk, Vv, Q = project(T, Tq, Wk, Wv, Wq)
        O = layer(Xk, Vv, Q)
        loss = ((O - Y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"\n  loss: init={losses[0]:.4f} -> final={losses[-1]:.4f} "
          f"({100*(1-losses[-1]/losses[0]):.0f}% reduction)")
    assert losses[-1] < 0.4 * losses[0], "SV-attention failed to learn the teacher mapping"
    assert losses[-1] < losses[0]
