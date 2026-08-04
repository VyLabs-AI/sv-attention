"""Gradient verification for the streaming-solver-backed differentiable SVDD.

FastDiffSVDD must (1) pass double-precision gradcheck (the implicit VJP reusing
the rank-1-maintained inverse R is correct) and (2) produce the same alpha/rho
and the same gradient as the cvxpy-based OneClassSVDD oracle on the same keys --
confirming the fast incremental partition matches the batch QP.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpy")

from svattn import OneClassSVDD
from svattn.fast_diff_svdd import fast_diff_svdd
from svattn.sv_attention import rbf_gram


def _make_X(n=12, d=3, seed=1):
    rng = np.random.RandomState(seed)
    return torch.tensor(rng.randn(n, d) * 1.2, dtype=torch.float64, requires_grad=True)


def test_fast_diff_svdd_gradcheck():
    """Implicit VJP matches finite differences (double precision)."""
    X = _make_X(n=12, d=3, seed=1)

    def f(Xin):
        alpha, rho = fast_diff_svdd(Xin, C=0.25, kpar=2.0)
        w = torch.linspace(-1.0, 1.0, alpha.shape[0], dtype=torch.float64)
        return (alpha * w).sum() + 0.5 * rho

    assert torch.autograd.gradcheck(f, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_fast_matches_cvxpy_oracle_forward_and_grad():
    """Same partition => same alpha/rho and same dL/dX as the cvxpy oracle."""
    X = _make_X(n=14, d=4, seed=2)
    C, kpar = 0.2, 2.5
    w = torch.linspace(-1.0, 1.0, X.shape[0], dtype=torch.float64)

    Xc = X.detach().clone().requires_grad_(True)
    a_ref, rho_ref = OneClassSVDD(C=C, tol=1e-7)(rbf_gram(Xc, Xc, kpar))
    ((a_ref * w).sum() + 0.3 * rho_ref).backward()

    Xf = X.detach().clone().requires_grad_(True)
    a_fast, rho_fast = fast_diff_svdd(Xf, C=C, kpar=kpar)
    ((a_fast * w).sum() + 0.3 * rho_fast).backward()

    assert torch.allclose(a_fast, a_ref, atol=1e-6), float((a_fast - a_ref).abs().max())
    assert abs(rho_fast.item() - rho_ref.item()) < 1e-6
    assert torch.allclose(Xf.grad, Xc.grad, atol=1e-6), float((Xf.grad - Xc.grad).abs().max())


def test_fast_diff_svdd_is_sparse():
    """Reserve tokens get exactly zero weight (certified sparsity)."""
    rng = np.random.RandomState(6)
    X = torch.tensor(rng.randn(40, 4) * 1.3, dtype=torch.float64)
    alpha, _ = fast_diff_svdd(X, C=0.1, kpar=2.0)
    nz = int((alpha.abs() > 0).sum())
    assert 1 <= nz < 40, f"expected sparse support, got {nz}/40 nonzero"
