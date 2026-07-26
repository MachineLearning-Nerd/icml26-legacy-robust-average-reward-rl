"""Model-free Robust Halpern Iteration with R-SAMPLE (Algorithm 1, Theorem 5.3).

Estimates the robust Bellman operator T_P(Q^k) from NOMINAL samples only, via
the recursive difference scheme (R-SAMPLE, Algorithm 2 / Appendix B):

    D^k(s,a) = (1/m_k) sum_j d^k(s_j) - penalty_diff(s,a),  s_j ~ P_0(.|s,a)
    T^k = T^{k-1} + D^k        (telescoping; E[T^k] = T_P(Q^k))
    Q^k = (1-beta_k) Q^0 + beta_k T^{k-1}

Crucially Q^k is formed from the SAMPLED T^{k-1}, NOT the true Bellman operator
(this is the point the previous (rejected) evidence missed).  The penalty term
implements the Kumar et al. 2023 dual so that sampling the nominal kernel +
penalty is equivalent to sampling the worst-case kernel.

Per-iteration budget m_k = max(ceil(alpha * c_k * Sp(d^k)^2 / eps^2), 1) makes
the martingale concentration (Proposition C.2) hold, giving the
Õ(SA H^2 / eps^2) total sample complexity (Theorem 5.3).
"""
from __future__ import annotations

import numpy as np

from . import mdp as M


def c_schedule(k: int) -> float:
    """c_k = 5 (k+2) ln^2(k+2)  (Theorem 5.3)."""
    return 5.0 * (k + 2) * np.log(k + 2) ** 2


def r_sample(h_k: np.ndarray, h_km1: np.ndarray, m_k: int, P0: np.ndarray,
             unc: M.Uncertainty, rng: np.random.Generator) -> np.ndarray:
    """R-SAMPLE (Algorithm 2): estimate D^k = sigma(h^k) - sigma(h^{k-1}).

    Returns the (S,A) matrix D^k.  Samples m_k next-states per (s,a) from the
    NOMINAL kernel P0 and applies the uncertainty penalty difference so that
    E[D^k(s,a)] = sigma_{P_sa}(h^k) - sigma_{P_sa}(h^{k-1}) exactly.
    """
    S, A = P0.shape[:2]
    d = h_k - h_km1  # the difference whose expectation under P0 we estimate
    D = np.empty((S, A))
    for s in range(S):
        for a in range(A):
            Prow = P0[s, a]
            R = unc.radius if np.isscalar(unc.radius) else unc.radius[s, a]
            # empirical mean of d over m_k nominal samples
            idx = rng.choice(len(Prow), size=m_k, p=Prow)
            emp = float(np.mean(d[idx]))
            if unc.kind == "contamination":
                # sigma(h) = (1-R) P.h + R min h  ->  diff penalty = R(min h^k - min h^{k-1})
                penalty = R * (float(np.min(h_k)) - float(np.min(h_km1)))
                D[s, a] = (1.0 - R) * emp + penalty
            else:
                # lp: sigma(h) = P.h - R kappa_q(h)  ->  penalty diff = R (kappa_q(h^k)-kappa_q(h^km1))
                reach = Prow > 0
                kdiff = M.kappa_q(h_k, unc.p, reach) - M.kappa_q(h_km1, unc.p, reach)
                D[s, a] = emp - R * kdiff
    return D


def rhi_model_free(r: np.ndarray, P0: np.ndarray, unc: M.Uncertainty,
                   eps: float, n: int, delta: float = 0.05,
                   Q0: np.ndarray | None = None, rng: np.random.Generator | None = None,
                   track_estimation_error: bool = False
                   ) -> dict:
    """Run model-free RHI (Algorithm 1).

    Returns a dict with the final policy, Q, total samples consumed, per-iter
    span residual of the TRUE operator (for diagnostics), and (optionally) the
    estimation error ||T^k - T_P(Q^k)||_inf measured against the known model.
    """
    rng = np.random.default_rng() if rng is None else rng
    S, A = r.shape
    if Q0 is None:
        Q0 = np.zeros((S, A))
    alpha = float(np.log(2 * S * A * (n + 1) / delta))
    T_prev = r.copy()        # T^{-1} = r
    h_prev = np.zeros(S)     # h^{-1} = 0
    Q = Q0.copy()
    total_samples = 0
    m_history = []
    span_resid_true = []     # Sp(T_P(Q^k) - Q^k), uses true model (diagnostic)
    est_err = []             # ||T^{k-1} - T_P(Q^k)||_inf, uses true model (diag)
    h_history = []           # h^k = max_A Q^k trajectory
    if track_estimation_error:
        TQ0_true = M.robust_bellman_T(Q0, r, P0, unc)
    for k in range(n + 1):
        bk = k / (k + 2)
        Q = (1 - bk) * Q0 + bk * T_prev      # line 5: uses SAMPLED T^{k-1}
        h = Q.max(axis=1)                     # line 6
        h_history.append(h.copy())
        sp_d = M.span(h - h_prev)
        m_k = int(max(np.ceil(alpha * c_schedule(k) * sp_d ** 2 / eps ** 2), 1))
        m_history.append(m_k)
        Dk = r_sample(h, h_prev, m_k, P0, unc, rng)   # line 8
        T_prev = T_prev + Dk                 # line 9: T^k = T^{k-1} + D^k
        total_samples += S * A * m_k
        h_prev = h
        if track_estimation_error:
            TQ_true = M.robust_bellman_T(Q, r, P0, unc)
            span_resid_true.append(M.span(TQ_true - Q))
            est_err.append(float(np.max(np.abs(T_prev - TQ_true))))
    pi = Q.argmax(axis=1)
    return dict(pi=pi, Q=Q, total_samples=total_samples, n=n,
                m_history=np.array(m_history),
                span_resid_true=np.array(span_resid_true),
                est_err=np.array(est_err),
                h_history=np.array(h_history))
