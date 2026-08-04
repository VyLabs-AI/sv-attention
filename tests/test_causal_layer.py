"""Correctness tests for the causal chunk-frozen SV-Attention layer.

Validates the three things that define the layer: (1) causality -- a token's
output never depends on later tokens; (2) the chunk-frozen readout formula --
each chunk reads only its causal prefix, first chunk is context-free; (3) the
layer is end-to-end differentiable (gradcheck through the gate).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpy")

from svattn.causal_sv_attention import (
    CausalSVAttention,
    _raw_fista_gate,
    causal_sv_readout,
    causal_sv_readout_persistent,
)
from svattn.sv_attention import rbf_gram


def test_causal_masking_no_future_leak():
    """Perturbing the last token changes only its own output."""
    torch.manual_seed(0)
    layer = CausalSVAttention(d_model=4, n_heads=1, head_dim=4, C=0.4,
                              kpar=2.0, chunk=2).double()
    X = torch.randn(1, 6, 4, dtype=torch.float64)
    O1 = layer(X)
    X2 = X.clone()
    X2[0, -1] += 1.7                                   # perturb only the last token
    O2 = layer(X2)
    assert torch.allclose(O1[0, :-1], O2[0, :-1], atol=1e-10), \
        float((O1[0, :-1] - O2[0, :-1]).abs().max())
    assert not torch.allclose(O1[0, -1], O2[0, -1])    # its own output should move


def test_chunk_frozen_readout_formula():
    """First chunk is zeros; later chunk reads only the causal prefix (warmup)."""
    torch.manual_seed(1)
    T, chunk, kpar = 4, 2, 2.0
    K = torch.randn(T, 3, dtype=torch.float64)
    V = torch.randn(T, 5, dtype=torch.float64)
    Q = torch.randn(T, 3, dtype=torch.float64)
    # C tiny enough that the 2-token prefix is below the gate threshold -> uniform.
    O = causal_sv_readout(K, V, Q, C=0.25, kpar=kpar, chunk=chunk, normalize=True)

    assert torch.allclose(O[:2], torch.zeros(2, 5, dtype=torch.float64))  # chunk 0
    w = rbf_gram(Q[2:4], K[:2], kpar)                  # queries 2,3 read prefix 0,1 only
    ref = (w @ V[:2]) / w.sum(1, keepdim=True)
    assert torch.allclose(O[2:4], ref, atol=1e-12)


def test_raw_fista_proxy_preserves_feasible_projected_iterate():
    reference = torch.zeros(2, 3, dtype=torch.float64)
    values = np.array([[0.4, 0.3, 0.3], [0.7, 0.2, 0.1]])

    alpha, valid = _raw_fista_gate(values, reference, box_C=0.5)

    assert torch.equal(alpha, torch.tensor(values, dtype=torch.float64))
    assert valid.tolist() == [True, False]


def test_persistent_matches_restream_forward():
    """The maintained rank-1 solver path reproduces the re-stream chunk-frozen
    readout to floating-point tolerance (the G1 regression contract)."""
    torch.manual_seed(3)
    T, dk, dv, C, kpar, chunk = 40, 4, 5, 0.3, 2.0, 8
    K = torch.randn(T, dk, dtype=torch.float64)
    V = torch.randn(T, dv, dtype=torch.float64)
    Q = torch.randn(T, dk, dtype=torch.float64)
    O_ref = causal_sv_readout(K, V, Q, C, kpar, chunk, normalize=True)
    O_per = causal_sv_readout_persistent(K, V, Q, C, kpar, chunk, normalize=True)
    assert torch.allclose(O_ref, O_per, atol=1e-7, rtol=1e-5), \
        float((O_ref - O_per).abs().max())


def test_persistent_matches_restream_backward():
    """Gradients through the persistent path match the re-stream path."""
    torch.manual_seed(4)
    T, dk, dv, C, kpar, chunk = 40, 4, 5, 0.3, 2.0, 8
    V = torch.randn(T, dv, dtype=torch.float64)
    Q = torch.randn(T, dk, dtype=torch.float64)

    def grad_of(fn):
        K = torch.randn(T, dk, dtype=torch.float64, generator=torch.Generator().manual_seed(5),
                        requires_grad=True)
        (fn(K, V, Q, C, kpar, chunk, normalize=True) ** 2).sum().backward()
        return K.grad

    g_ref = grad_of(causal_sv_readout)
    g_per = grad_of(causal_sv_readout_persistent)
    assert torch.allclose(g_ref, g_per, atol=1e-6, rtol=1e-4), \
        float((g_ref - g_per).abs().max())


def test_mlx_matches_maintained_forward_within_tolerance():
    """Compare the batched approximation with the maintained chunk-frozen path."""
    mx = pytest.importorskip("mlx.core")
    torch.manual_seed(0)
    layer_e = CausalSVAttention(16, 2, 8, C=0.3, kpar=2.0, chunk=8,
                                persistent=True, solver="exact").double()
    layer_m = CausalSVAttention(16, 2, 8, C=0.3, kpar=2.0, chunk=8,
                                solver="mlx", fista_iters=250).double()
    layer_m.load_state_dict(layer_e.state_dict())
    X = torch.randn(2, 48, 16, dtype=torch.float64)
    with torch.no_grad():
        err = float((layer_e(X) - layer_m(X)).abs().max())
    assert err < 5e-3, err


def test_mlx_layer_gradcheck():
    """End-to-end gradient through the batched-MLX layer."""
    pytest.importorskip("mlx.core")
    torch.manual_seed(1)
    layer = CausalSVAttention(8, 1, 8, C=0.4, kpar=2.0, chunk=4,
                              solver="mlx", fista_iters=250).double()
    X = torch.randn(1, 12, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: (layer(z) ** 2).sum(), (X,),
                                    eps=1e-6, atol=1e-4, rtol=1e-3)


def test_hybrid_softmax_readout_causal_and_diff():
    """Hybrid readout (SV-gated long-range + local causal window) is causal and
    end-to-end differentiable."""
    pytest.importorskip("mlx.core")
    torch.manual_seed(1)
    layer = CausalSVAttention(16, 2, 8, C=0.3, kpar=2.0, chunk=4, solver="mlx",
                              fista_iters=200, readout="softmax").double()
    X = torch.randn(1, 12, 16, dtype=torch.float64)
    with torch.no_grad():
        o1 = layer(X)
        X2 = X.clone(); X2[0, -1] += 1.5
        o2 = layer(X2)
    assert torch.allclose(o1[0, :-1], o2[0, :-1], atol=1e-9), \
        float((o1[0, :-1] - o2[0, :-1]).abs().max())        # no future leak
    Xg = torch.randn(1, 12, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: (layer(z) ** 2).sum(), (Xg,),
                                    eps=1e-6, atol=1e-4, rtol=1e-3)


def test_persistent_layer_gradcheck():
    """End-to-end gradient through the persistent-solver layer."""
    torch.manual_seed(0)
    layer = CausalSVAttention(d_model=4, n_heads=1, head_dim=4, C=0.4,
                              kpar=2.0, chunk=2, persistent=True).double()
    X = torch.randn(1, 8, 4, dtype=torch.float64, requires_grad=True)

    def f(Xin):
        return (layer(Xin) ** 2).sum()

    assert torch.autograd.gradcheck(f, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_causal_layer_gradcheck():
    """End-to-end gradient through the layer (incl. a gated chunk)."""
    torch.manual_seed(0)
    layer = CausalSVAttention(d_model=4, n_heads=1, head_dim=4, C=0.4,
                              kpar=2.0, chunk=2).double()
    X = torch.randn(1, 8, 4, dtype=torch.float64, requires_grad=True)

    def f(Xin):
        return (layer(Xin) ** 2).sum()

    assert torch.autograd.gradcheck(f, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)
