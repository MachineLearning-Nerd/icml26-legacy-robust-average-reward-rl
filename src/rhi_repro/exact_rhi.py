"""Exact (model-known) Robust Halpern Iteration (Theorem 5.2).

Q^{k+1} = (1-beta_{k+1}) Q^0 + beta_{k+1} T_P(Q^k),  beta_k = k/(k+2).

Theorem 5.2: Sp(T_P(Q^k) - Q^k) -> 0  and  g*_P - g^{pi_k} -> 0  as k -> inf,
for the EXACT iteration with known uncertainty set.  This is the asymptotic-
convergence object (the finite-sample version is rhi.py / Theorem 5.3).
"""
from __future__ import annotations

import numpy as np

from . import mdp as M


def beta_schedule(k: int) -> float:
    """beta_k = k / (k + 2)."""
    return k / (k + 2)


def exact_halpern(r: np.ndarray, P: np.ndarray, unc: M.Uncertainty,
                  Q0: np.ndarray, K: int, record_every: int = 1,
                  gain_every: int = 200
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run exact RHI for K iterations.

    Returns (Q_final, span_history, gain_gap_history) where span_history[k] =
    Sp(T_P(Q^k) - Q^k) (cheap, every step) and gain_gap_history[k] = g* -
    g_P^{greedy(Q^k)} (expensive RRVI eval, only at multiples of gain_every;
    NaN between checkpoints)."""
    g_star, _, pi_star = M.robust_optimal_span_rvi(r, P, unc)
    Q = Q0.copy()
    span_hist = np.empty(K + 1)
    gap_hist = np.full(K + 1, np.nan)
    for k in range(K + 1):
        TQ = M.robust_bellman_T(Q, r, P, unc)
        span_hist[k] = M.span(TQ - Q)
        if k % gain_every == 0 or k == K:
            pi_k = Q.argmax(axis=1)
            g_pk, _ = M.robust_policy_gain(pi_k, r, P, unc)
            gap_hist[k] = g_star - g_pk
        if k < K:
            b = beta_schedule(k + 1)
            Q = (1 - b) * Q0 + b * TQ
    return Q, span_hist, gap_hist
