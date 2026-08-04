"""Batch QP ground truth for the one-class (nu-SVDD) problem.

This is the solver-independent reference the incremental algorithm is verified
against. The dual problem implemented by the MATLAB code (derivable from the
gradient in Compute_grad.m) is:

    minimize_alpha   alpha^T K alpha - diag(K)^T alpha
    subject to       sum(alpha) = 1,   0 <= alpha_i <= C

with the KKT stationarity gradient (matching Compute_grad.m, y_i == 1):

    g_i = rho - K_ii + 2 * sum_j K_ij alpha_j

Active-set membership:
    g_i = 0, 0 < alpha_i < C   -> margin set S
    g_i < 0, alpha_i = C        -> error set E (outliers)
    g_i > 0, alpha_i = 0        -> reserve set R (inert)
"""
from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None


class SVDDQPInfeasibleError(RuntimeError):
    """The box constraints cannot carry unit alpha mass."""


def solve_svdd_qp(K: np.ndarray, C: float) -> np.ndarray:
    """Solve the nu-SVDD dual to high accuracy and return alpha."""
    if cp is None:
        raise ImportError("cvxpy is required for the batch-QP ground truth")
    n = K.shape[0]
    if n * float(C) < 1.0 - 1e-12:
        raise SVDDQPInfeasibleError(
            f"SVDD QP is infeasible: n*C={n * float(C):.12g} < 1 "
            f"(n={n}, C={float(C):.12g})"
        )
    # Symmetrize to keep the QP numerically PSD.
    Ksym = 0.5 * (K + K.T)
    alpha = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(alpha, cp.psd_wrap(Ksym)) - np.diag(K) @ alpha)
    constraints = [cp.sum(alpha) == 1, alpha >= 0, alpha <= C]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
               tol_feas=1e-10, tol_infeas_abs=1e-10, tol_infeas_rel=1e-10)
    if alpha.value is None:
        if prob.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
            raise SVDDQPInfeasibleError(
                f"SVDD QP solver reported infeasible despite n*C="
                f"{n * float(C):.12g}: status={prob.status}"
            )
        raise RuntimeError(f"QP did not solve: status={prob.status}")
    a = np.asarray(alpha.value, dtype=np.float64)
    # Clean tiny numerical violations of the box.
    a = np.clip(a, 0.0, C)
    return a


def recover_rho(K: np.ndarray, alpha: np.ndarray, C: float, tol: float = 1e-6) -> float:
    """Recover the offset rho from any margin support vector.

    rho = K_ii - 2 (K alpha)_i for any i with 0 < alpha_i < C.
    Falls back to averaging over the most-interior points if none are strictly free.
    """
    Ka = K @ alpha
    g_const = np.diag(K) - 2.0 * Ka  # = rho at margin points
    margin = (alpha > tol) & (alpha < C - tol)
    if np.any(margin):
        return float(np.mean(g_const[margin]))
    # Degenerate fallback: use points nearest the interior.
    order = np.argsort(np.abs(alpha - C / 2.0))
    return float(g_const[order[0]])


def partition_sets(alpha: np.ndarray, C: float, tol: float = 1e-6):
    """Partition indices into (S margin, E error, R reserve) by alpha value."""
    S = np.where((alpha > tol) & (alpha < C - tol))[0].tolist()
    E = np.where(alpha >= C - tol)[0].tolist()
    R = np.where(alpha <= tol)[0].tolist()
    return S, E, R
