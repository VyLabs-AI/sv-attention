"""Gradient verification for the differentiable one-class SVDD layer.

The implicit-gradient backward pass (reusing the bordered KKT inverse R) must
match finite differences. We verify with torch.autograd.gradcheck in double
precision, differentiating through K = rbf(X) w.r.t. the points X (so the
partition is locally stable and perturbations keep K symmetric/PSD).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpy")

from svattn import OneClassSVDD


def rbf_K(X, kpar):
    d2 = torch.cdist(X, X) ** 2
    return torch.exp(-d2 / (kpar * kpar))


def _make_X(n=12, d=3, seed=0):
    rng = np.random.RandomState(seed)
    return torch.tensor(rng.randn(n, d) * 1.2, dtype=torch.float64, requires_grad=True)


def test_svdd_alpha_grad_matches_finite_diff():
    torch.manual_seed(0)
    X = _make_X(n=12, d=3, seed=1)
    svdd = OneClassSVDD(C=0.25, tol=1e-7)

    def f(Xin):
        K = rbf_K(Xin, kpar=2.0)
        alpha, rho = svdd(K)
        # arbitrary smooth scalarization of the dual solution
        w = torch.linspace(-1.0, 1.0, alpha.shape[0], dtype=torch.float64)
        return (alpha * w).sum() + 0.5 * rho

    assert torch.autograd.gradcheck(f, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_svdd_decision_score_grad_matches_finite_diff():
    X = _make_X(n=14, d=4, seed=2)
    svdd = OneClassSVDD(C=0.2, tol=1e-7)

    def f(Xin):
        K = rbf_K(Xin, kpar=2.5)
        alpha, rho = svdd(K)
        # decision scores on the training points: f_i = 2 (K alpha)_i - rho
        scores = 2.0 * (K @ alpha) - rho
        return (scores ** 2).mean()

    assert torch.autograd.gradcheck(f, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_sv_attention_layer_grad_matches_finite_diff():
    """End-to-end gradient through the SV-attention readout (alpha load-bearing)."""
    from svattn import SVAttention

    rng = np.random.RandomState(5)
    X = torch.tensor(rng.randn(12, 3) * 1.2, dtype=torch.float64, requires_grad=True)
    V = torch.tensor(rng.randn(12, 4), dtype=torch.float64, requires_grad=True)
    Q = torch.tensor(rng.randn(5, 3) * 1.2, dtype=torch.float64, requires_grad=True)
    layer = SVAttention(C=0.25, kpar=2.0)

    def f(Xin, Vin, Qin):
        O = layer(Xin, Vin, Qin)
        return (O ** 2).sum()

    assert torch.autograd.gradcheck(f, (X, V, Q), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_sv_attention_is_sparse():
    """Reserve tokens get exactly zero attention weight (certified sparsity)."""
    from svattn import SVAttention
    rng = np.random.RandomState(6)
    X = torch.tensor(rng.randn(40, 4) * 1.3, dtype=torch.float64)
    layer = SVAttention(C=0.1, kpar=2.0)
    mask = layer.support_mask(X)
    assert mask.sum().item() < 40, "expected some reserve (zero-weight) tokens"
    assert mask.sum().item() >= 1
