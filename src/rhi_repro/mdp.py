"""Robust average-reward MDP primitives (arXiv 2505.12462, Sec. 3).

Implements the support function sigma_{P_sa} for the (s,a)-rectangular
contamination and l_p-norm uncertainty sets, the penalty term kappa_q (dual
representation, Kumar et al. 2023 / Appendix C), the robust Bellman operator
T_P(Q) = r + sigma_P(max_A Q), and exact solvers for the optimal robust gain
(RRVI span-relative-value-iteration) and per-policy robust gain evaluation.

All operators are EXACT (model-known): they are the ground truth the model-free
RHI algorithm (rhi.py) is measured against, and the objects Lemma 5.1 /
Theorem 5.2 / Theorem 4.2 are stated over.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def span(v: np.ndarray) -> float:
    """Span semi-norm Sp(v) = max_i v_i - min_i v_i (Eq. 9)."""
    return float(np.max(v) - np.min(v))


def holder_q(p: float) -> float:
    """Hölder conjugate q with 1/p + 1/q = 1."""
    if p == np.inf:
        return 1.0
    if p == 1.0:
        return np.inf
    return p / (p - 1.0)


def kappa_q(h: np.ndarray, p: float, reachable: np.ndarray | None = None) -> float:
    """kappa_q(h) = min_omega || u - omega * 1 ||_q  (Appendix C, eq. near 929).

    u(s) = h(s) * 1(s reachable); q is the Hölder conjugate of p.
    Closed forms: q=1 -> median, q=2 -> mean, q=inf -> midrange.
    """
    if reachable is None:
        u = np.asarray(h, dtype=float)
    else:
        u = np.where(reachable, h, 0.0)
    q = holder_q(p)
    if q == 1.0:
        omega = np.median(u)
    elif q == 2.0:
        omega = float(np.mean(u))
    elif q == np.inf:
        omega = 0.5 * (float(np.max(u)) + float(np.min(u)))
    else:
        # convex 1-D minimization
        lo, hi = float(np.min(u)), float(np.max(u))
        if hi <= lo:
            omega = lo
        else:
            res = minimize_scalar(
                lambda w: np.sum(np.abs(u - w) ** q), bounds=(lo, hi), method="bounded"
            )
            omega = float(res.x)
    diff = u - omega
    if q == np.inf:
        return float(np.max(np.abs(diff)))
    return float(np.sum(np.abs(diff) ** q) ** (1.0 / q))


def sigma_contamination(Prow: np.ndarray, h: np.ndarray, R: float) -> float:
    """sigma for contamination set {(1-R)P + R q : q in Delta} (Eq. 4).

    min_q E[(1-R)P + R q][h] = (1-R) P.h + R min_s h(s).
    """
    return float((1.0 - R) * np.dot(Prow, h) + R * np.min(h))


def sigma_lp(Prow: np.ndarray, h: np.ndarray, R: float, p: float,
             reachable: np.ndarray | None = None) -> float:
    """sigma for l_p-norm ball {q : ||q-P||_p <= R} (Eq. 5, Appendix C eq. 74).

    = P.h - R * kappa_q(h).  Valid when the ball is interior to the simplex.
    """
    return float(np.dot(Prow, h) - R * kappa_q(h, p, reachable))


class Uncertainty:
    """Container for an (s,a)-rectangular uncertainty set."""

    KINDS = ("contamination", "lp")

    def __init__(self, kind: str, radius: float | np.ndarray, p: float = 2.0):
        assert kind in self.KINDS, kind
        self.kind = kind
        self.radius = radius
        self.p = p

    def sigma(self, Prow: np.ndarray, h: np.ndarray, R: float,
              reachable: np.ndarray | None = None) -> float:
        if self.kind == "contamination":
            return sigma_contamination(Prow, h, R)
        return sigma_lp(Prow, h, R, self.p, reachable)


def robust_bellman_T(Q: np.ndarray, r: np.ndarray, P: np.ndarray,
                     unc: Uncertainty) -> np.ndarray:
    """Robust Bellman operator T_P(Q)(s,a) = r(s,a) + sigma_{P_sa}(max_A Q).

    T_P,g(Q) = T_P(Q) - g; the residual T_P(Q) - Q drives Lemma 5.1 / Thm 5.2.
    """
    h = Q.max(axis=1)  # max_A Q  (the V vector)
    return r + _sigma_matrix(P, h, unc)


# ---------------- exact robust solvers (RRVI, Wang et al. 2023a) -------------

def robust_discounted_vi(r: np.ndarray, P: np.ndarray, unc: Uncertainty,
                         gamma: float, iters: int = 5000, tol: float = 1e-12
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Optimal robust *discounted* value/value via value iteration (contractive).

    V(s) = max_a { r(s,a) + gamma sigma_{P_sa}(V) }.  Used by the reduction
    approach (Theorem 4.2) and as a gamma->1 cross-check on the optimal gain.
    """
    S, A = r.shape
    V = np.zeros(S)
    for _ in range(iters):
        Q = r + gamma * _sigma_matrix(P, V, unc)
        Vn = Q.max(axis=1)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    Q = r + gamma * _sigma_matrix(P, V, unc)
    return V, Q


def _sigma_matrix(P: np.ndarray, h: np.ndarray, unc: Uncertainty) -> np.ndarray:
    """Compute sigma_{P_sa}(h) for all (s,a) -> (S,A) array (vectorized)."""
    S, A = P.shape[:2]
    Ph = np.einsum("sap,p->sa", P, h)  # P[s,a] . h  for all (s,a)
    if unc.kind == "contamination":
        R = unc.radius
        if not np.isscalar(R):
            R = R  # per-(s,a) radius matrix
        return (1.0 - R) * Ph + R * float(np.min(h))
    # lp-norm: sigma = P.h - R * kappa_q(h, s,a).  kappa depends on (s,a) only
    # through the reachable set; for fully-connected MDPs it is a single scalar.
    reach = P > 0  # (S,A,S)
    full = reach.all(axis=2)
    if full.all():
        kappa = kappa_q(h, unc.p)
        R = unc.radius
        return Ph - R * kappa
    # mixed: per-(s,a) reachable sets (slower path)
    out = Ph.copy()
    R = unc.radius
    for s in range(S):
        for a in range(A):
            if not full[s, a]:
                out[s, a] = Ph[s, a] - (R if np.isscalar(R) else R[s, a]) * kappa_q(
                    h, unc.p, reach[s, a])
    return out


def robust_optimal_span_rvi(r: np.ndarray, P: np.ndarray, unc: Uncertainty,
                            iters: int = 20000, tol: float = 1e-11, ref: int = 0
                            ) -> tuple[float, np.ndarray, np.ndarray]:
    """Optimal robust average reward g*, bias h*, greedy policy (RRVI).

    Span-relative value iteration on T(h)(s)=max_a{r(s,a)+sigma(h)} is a
    non-expansion in Sp(.); anchoring to a reference state keeps h bounded and
    g_k = T(h_k)(ref) -> g* under unichain+compact (Assumption 3.1).
    """
    S, A = r.shape
    h = np.zeros(S)
    g = 0.0
    for _ in range(iters):
        Th = (r + _sigma_matrix(P, h, unc)).max(axis=1)
        g = Th[ref]
        hn = Th - g  # subtract reference value -> bounded bias
        if span(hn - h) < tol and np.max(np.abs(hn - h)) < tol:
            h = hn
            break
        h = hn
    Q = r + _sigma_matrix(P, h, unc)
    pi = Q.argmax(axis=1)
    return float(g), h, pi


def robust_policy_gain(pi: np.ndarray, r: np.ndarray, P: np.ndarray,
                       unc: Uncertainty, iters: int = 20000, tol: float = 1e-11,
                       ref: int = 0) -> tuple[float, np.ndarray]:
    """Robust average reward g_P^pi of a FIXED policy + its bias (RRVI eval).

    Solves h = r_pi - g*1 + sigma_{pi}(h) via span-RVE (non-expansion).
    """
    S, A = r.shape
    rpi = r[np.arange(S), pi]
    h = np.zeros(S)
    g = 0.0
    for _ in range(iters):
        sh = _sigma_policy(pi, P, h, unc)
        Th = rpi + sh
        g = Th[ref]
        hn = Th - g
        if span(hn - h) < tol and np.max(np.abs(hn - h)) < tol:
            h = hn
            break
        h = hn
    return float(g), h


def _sigma_policy(pi: np.ndarray, P: np.ndarray, h: np.ndarray,
                  unc: Uncertainty) -> np.ndarray:
    S = len(pi)
    out = np.empty(S)
    for s in range(S):
        a = pi[s]
        Prow = P[s, a]
        R = unc.radius if np.isscalar(unc.radius) else unc.radius[s, a]
        reachable = Prow > 0
        if unc.kind == "lp" and not reachable.all():
            out[s] = unc.sigma(Prow, h, R, reachable)
        else:
            out[s] = unc.sigma(Prow, h, R)
    return out


def robust_bias_span(pi: np.ndarray, r: np.ndarray, P: np.ndarray,
                     unc: Uncertainty) -> tuple[float, np.ndarray]:
    """Robust optimal bias span H = max_P Sp(h_P^{pi*}): here computed under the
    worst-case kernel realised by RRVI evaluation of the greedy policy."""
    _, h = robust_policy_gain(pi, r, P, unc)
    return span(h), h


# ---------------- MDP generation --------------------------------------------

def garnet_mdp(S: int, A: int, density: float, rng: np.random.Generator,
               n_recurrent: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Garnet G(S,A) MDP (Archibald et al. 1995) with unichain guarantee.

    Returns P (S,A,S) and r (S,A) in [0,1].  Each (s,a) has ~density*S nonzero
    next-states.  A single recurrent class (states 0..n_rec-1) is enforced so
    every deterministic policy yields a unichain (Assumption 3.1).  Transient
    states (if any) drain into the recurrent class.
    """
    if n_recurrent is None:
        n_recurrent = S
    n_rec = min(n_recurrent, S)
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            if s < n_rec:
                k = max(2, int(round(density * n_rec)))
                targets = rng.choice(n_rec, size=k, replace=False)
                probs = rng.dirichlet(np.ones(k))
                P[s, a, targets] = probs
            else:
                # transient: drain to recurrent class (unichain guarantee)
                P[s, a, rng.integers(0, n_rec)] = 1.0
    r = rng.uniform(0, 1, (S, A))
    # sanity: every row sums to 1
    assert np.allclose(P.sum(axis=2), 1.0)
    return P, r
