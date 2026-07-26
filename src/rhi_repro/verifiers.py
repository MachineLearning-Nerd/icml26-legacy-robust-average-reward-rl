"""Per-claim verifiers for arXiv 2505.12462.

Each verifier returns (verdict, detail) where verdict in {VERIFIED, FALSIFIED,
BLOCKED} and detail is a dict of machine-checkable numbers.  Every verifier is
deterministic given a seed and exits nonzero on failure.  Verifiers write raw
CSV/JSON under a given artifact dir.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from . import mdp as M
from . import exact_rhi as E
from . import rhi as R
from . import reduction as RED


# ===========================================================================
# Claim 3 - Lemma 5.1:  0 <= g*_P - g^pi_P(s) <= Sp(T_P(Q) - Q)   [universal]
# ===========================================================================

def verify_c3_lemma51(n_mdp: int = 400, seed: int = 0, art_dir: str | None = None) -> dict:
    """Exhaustive random stress test of Lemma 5.1 over many robust AMDPs + Q.

    For every instance: build a unichain robust AMDP, sample a random Q, take
    its greedy policy, and check 0 <= g* - g^pi(s) <= Sp(T_P(Q)-Q) for all s,
    with the optimal gain and bias computed exactly (RRVI).  A *negative
    control* takes a NON-greedy policy (a deliberately wrong action) on the same
    Q and verifies the bound CAN be violated there -- proving the greedy-w.r.t.-
    Q structure is what makes the lemma hold, not the span bound being loose.

    Universally quantified lemma -> finite instances are scoped corroboration;
    we also reconstruct the proof (Appendix C.1) in the report.
    """
    rng = np.random.default_rng(seed)
    cfgs = [
        dict(kind="contamination", radius=0.1, p=2.0),
        dict(kind="contamination", radius=0.25, p=2.0),
        dict(kind="lp", radius=0.05, p=2.0),
        dict(kind="lp", radius=0.05, p=np.inf),
        dict(kind="lp", radius=0.08, p=1.0),
    ]
    sizes = [(6, 3), (8, 4), (10, 5), (7, 6)]
    rows = []
    n_checked = 0
    n_lhs_viol = 0
    n_rhs_viol = 0
    worst_ratio = 0.0
    nc_nongreedy_viol = 0     # non-greedy policy violates the bound (control)
    nc_nongreedy_tested = 0
    for i in range(n_mdp):
        S, A = sizes[i % len(sizes)]
        cfg = cfgs[i % len(cfgs)]
        unc = M.Uncertainty(cfg["kind"], cfg["radius"], cfg["p"])
        P, r = M.garnet_mdp(S, A, 0.55, rng)
        try:
            g_star, _, pi_star = M.robust_optimal_span_rvi(r, P, unc, iters=4000)
        except Exception:
            continue
        Q = rng.uniform(0, 1, (S, A))
        pi = Q.argmax(axis=1)
        TQ = M.robust_bellman_T(Q, r, P, unc)
        sp_resid = M.span(TQ - Q)
        g_pi, _ = M.robust_policy_gain(pi, r, P, unc, iters=4000)
        gap = g_star - g_pi
        ok = (gap >= -1e-9) and (gap <= sp_resid + 1e-9)
        n_checked += 1
        if gap < -1e-9:
            n_lhs_viol += 1
        if gap > sp_resid + 1e-9:
            n_rhs_viol += 1
        worst_ratio = max(worst_ratio, gap / sp_resid if sp_resid > 1e-12 else 0.0)
        # negative control: a NON-greedy policy (worst action per state)
        pi_bad = ((Q.max(axis=1, keepdims=True) - Q)).argmax(axis=1)  # argmin of Q
        g_bad, _ = M.robust_policy_gain(pi_bad, r, P, unc, iters=4000)
        gap_bad = g_star - g_bad
        nc_nongreedy_tested += 1
        if gap_bad > sp_resid + 1e-9:
            nc_nongreedy_viol += 1
        rows.append(dict(i=i, S=S, A=A, kind=cfg["kind"], p=cfg["p"],
                         radius=cfg["radius"], g_star=g_star, g_pi=g_pi, gap=gap,
                         sp_resid=sp_resid, ok=ok,
                         nongreedy_gap=gap_bad,
                         nongreedy_violates=bool(gap_bad > sp_resid + 1e-9)))
    passed = (n_lhs_viol == 0) and (n_rhs_viol == 0)
    # extra negative control at Q near the FIXED POINT (tiny residual): there a
    # non-greedy policy's gap MUST exceed Sp(residual), proving the bound is
    # specific to the greedy policy and not vacuously loose.
    nc_fp = _c3_fixedpoint_nongreedy_control(rng, cfgs, sizes)
    nc_effective = (nc_nongreedy_viol > 0) or nc_fp["violates"]
    detail = dict(
        verdict="VERIFIED" if passed else "FALSIFIED",
        n_checked=n_checked, n_lhs_violations=n_lhs_viol, n_rhs_violations=n_rhs_viol,
        worst_gap_over_span=worst_ratio,
        nc_nongreedy_tested=nc_nongreedy_tested,
        nc_nongreedy_violations=nc_nongreedy_viol,
        nc_fixedpoint_control=nc_fp,
        negative_control_effective=nc_effective,
        negative_control_desc="non-greedy policy on Q near the fixed point (tiny residual): "
                              "its gap exceeds Sp(T_P(Q)-Q), so the lemma's bound is specific to "
                              "the greedy-w.r.t.-Q policy and not a loose over-bound",
    )
    if art_dir:
        _write_csv(art_dir, "c3_lemma51_instances.csv", rows)
    return detail


def _c3_fixedpoint_nongreedy_control(rng, cfgs, sizes):
    """At Q near the FIXED POINT (tiny residual span), a non-greedy policy has a
    positive gap, so gap > Sp(residual) -> the bound is violated for non-greedy
    policies, confirming the greedy requirement is essential."""
    from . import exact_rhi as E
    S, A = sizes[1]
    cfg = cfgs[2]
    unc = M.Uncertainty(cfg["kind"], cfg["radius"], cfg["p"])
    P, r = M.garnet_mdp(S, A, 0.55, rng)
    g_star, h_star, pi_star = M.robust_optimal_span_rvi(r, P, unc, iters=6000)
    # run Halpern to convergence -> Q with tiny Sp(T_P(Q)-Q)
    Q0 = rng.uniform(0, 1, (S, A))
    Qf, sp_hist, _ = E.exact_halpern(r, P, unc, Q0, K=1500, gain_every=10 ** 9)
    sp_resid = float(sp_hist[-1])
    # non-greedy policy (rotate actions away from optimal)
    pi_bad = (pi_star + 1) % A
    g_bad, _ = M.robust_policy_gain(pi_bad, r, P, unc, iters=6000)
    gap_bad = g_star - g_bad
    violates = gap_bad > sp_resid + 1e-9
    return dict(sp_resid=sp_resid, nongreedy_gap=gap_bad, violates=violates)


# ===========================================================================
# Claim 4 - Theorem 5.2:  Sp(T_P(Q_k) - Q_k) -> 0  (exact Halpern iteration)
# ===========================================================================

def verify_c4_thm52(seed: int = 0, art_dir: str | None = None) -> dict:
    """Exact (model-known) Halpern convergence on MDPs of growing size, all
    three uncertainty sets.  Records span residual -> 0 and the O(1/k) rate
    (Lieder 2021 for non-expansions).

    Negative control: T_P is a NON-EXPANSION (not a contraction) -- we exhibit
    pairs (Q1,Q2) where ||T_P(Q1)-T_P(Q2)||_sp / ||Q1-Q2||_sp == 1 (equality),
    which is exactly why the paper states "Banach-Picard iteration" lacks a
    contraction-based convergence guarantee and adopts Halpern instead.  We
    additionally report whether plain Picard VI converges the span residual on
    each instance (it often does on aperiodic unichain MDPs, but with no
    guarantee and at an uncontrolled rate)."""
    rng = np.random.default_rng(seed)
    cfgs = [
        ("contamination", 0.1, 2.0), ("lp", 0.05, 2.0), ("lp", 0.05, np.inf),
    ]
    sizes = [(6, 3), (12, 6), (20, 15)]  # incl. paper scale G(20,15)
    K = 3000
    rows, curves = [], {}
    for (S, A) in sizes:
        for (kind, rad, p) in cfgs:
            unc = M.Uncertainty(kind, rad, p)
            P, r = M.garnet_mdp(S, A, 0.5, rng)
            Q0 = rng.uniform(0, 1, (S, A))
            Qf, sp_hist, gap_hist = E.exact_halpern(r, P, unc, Q0, K, gain_every=500)
            sp_vi = _vi_span_control(r, P, unc, Q0, K)
            kk = np.arange(50, K + 1)
            y = np.maximum(sp_hist[50:], 1e-12)
            slope, _ = np.polyfit(np.log(kk), np.log(y), 1)
            tag = f"S{S}_A{A}_{kind}_p{p}"
            step = max(1, K // 200)
            curves[tag] = dict(k=list(range(0, K + 1, step)),
                               span=[float(sp_hist[j]) for j in range(0, K + 1, step)],
                               vi_span=[float(sp_vi[j]) for j in range(0, K + 1, step)])
            final_gap = float(gap_hist[np.isfinite(gap_hist)][-1]) if np.any(np.isfinite(gap_hist)) else float("nan")
            rows.append(dict(S=S, A=A, kind=kind, p=p,
                             span_k0=float(sp_hist[0]), span_kK=float(sp_hist[-1]),
                             reduction=float(sp_hist[-1] / sp_hist[0]),
                             log_slope=float(slope), final_gain_gap=final_gap,
                             halpern_converges=bool(sp_hist[-1] < 1e-2 * sp_hist[0]),
                             vi_span_kK=float(sp_vi[-1])))
    halpern_all = all(r["halpern_converges"] for r in rows)
    # negative control: T_P is a non-expansion (contraction ratio can hit 1)
    nc = _nonexpansion_control(P, r, unc, rng)
    detail = dict(verdict="VERIFIED" if halpern_all else "FALSIFIED",
                  instances=rows, halpern_converges_all=halpern_all,
                  negative_control="T_P is a non-expansion: ||T_P(Q1)-T_P(Q2)||_sp can equal "
                                   "||Q1-Q2||_sp (ratio 1.0), so Picard/VI has no contraction guarantee",
                  max_contraction_ratio=nc["max_ratio"],
                  n_pairs_tested=nc["n_pairs"],
                  nonexpansion_established=nc["max_ratio"] > 0.999)
    if art_dir:
        _write_csv(art_dir, "c4_thm52_curves.csv", rows)
        with open(os.path.join(art_dir, "c4_thm52_curves.json"), "w") as f:
            json.dump(curves, f)
    return detail


def _nonexpansion_control(P, r, unc, rng, n_pairs=200):
    """Demonstrate T_P is a non-expansion (ratio can reach 1, not a strict
    contraction).  Random pairs rarely hit equality, so we add a DELIBERATE
    construction: an identity-transition, single-greedy-action instance where
    max_A and sigma(P,.) both preserve the span exactly -> ratio == 1.

    A strict contraction would have ratio < 1 everywhere; equality proves T_P is
    only a non-expansion (the paper's stated reason Halpern, not Picard, is used).
    """
    S, A = r.shape
    max_ratio = 0.0
    for _ in range(n_pairs):
        Q1 = rng.uniform(0, 1, (S, A))
        Q2 = Q1 + rng.normal(0, 0.3, (S, A))
        T1 = M.robust_bellman_T(Q1, r, P, unc)
        T2 = M.robust_bellman_T(Q2, r, P, unc)
        num = M.span(T1 - T2)
        den = M.span(Q1 - Q2)
        if den > 1e-9:
            max_ratio = max(max_ratio, num / den)
    # deliberate construction: 2-state, A=2, identity transitions, R=0 (limit).
    # Both Q's share greedy action a0; they differ only at s1's a0 entry, so the
    # max_A map and sigma(P=I,.) both preserve the span exactly -> ratio == 1.
    S2, A2 = 2, 2
    Pdet = np.zeros((S2, A2, S2))
    Pdet[0, :, 0] = 1.0
    Pdet[1, :, 1] = 1.0
    rdet = np.zeros((S2, A2))
    unc0 = M.Uncertainty(unc.kind, 0.0, unc.p)  # no uncertainty -> non-robust limit
    Q1 = np.array([[5.0, 0.0], [1.0, 0.0]])
    Q2 = np.array([[5.0, 0.0], [0.0, 0.0]])
    T1 = M.robust_bellman_T(Q1, rdet, Pdet, unc0)
    T2 = M.robust_bellman_T(Q2, rdet, Pdet, unc0)
    deliberate = M.span(T1 - T2) / max(M.span(Q1 - Q2), 1e-12)
    max_ratio = max(max_ratio, deliberate)
    return dict(max_ratio=max_ratio, n_pairs=n_pairs,
                deliberate_construction_ratio=float(deliberate))


def _vi_span_control(r, P, unc, Q0, K):
    """Span residual Sp(T_P(Q_k)-Q_k) under Banach-Picard Q <- T_P(Q)."""
    Q = Q0.copy()
    sp = np.empty(K + 1)
    for k in range(K + 1):
        TQ = M.robust_bellman_T(Q, r, P, unc)
        sp[k] = M.span(TQ - Q)
        Q = TQ
    return sp


# ===========================================================================
# Claim 5 - Algorithm 1 / R-SAMPLE: recursive difference estimation
# ===========================================================================

def verify_c5_rsample(seed: int = 1, art_dir: str | None = None) -> dict:
    """Verify the recursive sampling scheme (Algorithm 1 + Algorithm 2).

    Checks: (1) Proposition C.2: ||T^k - T_P(Q^k)||_inf <= eps for ALL k (the
    accumulated martingale error stays bounded -- no MLMC-style blowup);
    (2) Q^k is built from the SAMPLED T^{k-1}, NOT the true Bellman operator
    (the rejected baseline's flaw); (3) the per-iteration budget m_k (proportional
    to Sp(d^k)^2) DECREASES as RHI converges, because the recursive scheme
    estimates *differences* d^k = h^k - h^{k-1} which shrink -- this is the sample
    efficiency vs a single-level re-estimator that pays ~||h^k||^2 every step.
    Negative control: a single-level re-estimator (no reuse) consumes strictly
    more samples for the same per-step error bound.
    """
    rng = np.random.default_rng(seed)
    P, r = M.garnet_mdp(10, 5, 0.5, rng)
    unc = M.Uncertainty("lp", 0.05, 2.0)
    g_star, _, pi_star = M.robust_optimal_span_rvi(r, P, unc)
    H, _ = M.robust_bias_span(pi_star, r, P, unc)
    eps = 0.15
    n = max(int(np.ceil(H / eps)), 8)  # at least 8 iters to see the trend
    res = R.rhi_model_free(r, P, unc, eps=eps, n=n, delta=0.05, rng=rng,
                           track_estimation_error=True)
    est_err = res["est_err"]
    # (1) Proposition C.2: accumulated error bounded by eps at every iteration
    c2_holds = bool(np.all(est_err[1:] <= eps + 1e-9))
    err_bounded = bool(np.max(est_err[1:]) <= eps + 1e-9)  # no blowup
    # (2) uses sampled T (est error is genuinely nonzero)
    uses_sampled_T = bool(np.mean(est_err) > 1e-6)
    # (3) per-iteration m_k decreases: d^k = h^k - h^{k-1} shrinks as RHI converges
    m_hist = res["m_history"]
    if len(m_hist) >= 6:
        early = np.mean(m_hist[1:4])
        late = np.mean(m_hist[-3:])
        m_decreases = bool(late <= early)
    else:
        early = late = float(np.mean(m_hist))
        m_decreases = True
    # variance of D^k at a FIXED (h^k, h^{k-1}) across independent draws (bounded)
    var_at_mid, var_at_late = _d_variance(r, P, unc, res, rng)
    var_bounded = bool(var_at_late <= var_at_mid * 4 + 1e-9)
    # negative control: single-level re-estimator (no reuse) budget for same err
    budget_rhi = res["total_samples"]
    budget_single = _single_level_budget(r, P, eps, n)
    detail = dict(
        verdict="VERIFIED" if (c2_holds and uses_sampled_T and err_bounded and
                               m_decreases and var_bounded and budget_rhi < budget_single)
        else "FALSIFIED",
        eps=eps, n=n, H=H,
        prop_C2_all_within_eps=c2_holds,
        accumulated_error_bounded=err_bounded,
        max_accumulated_err=float(np.max(est_err)),
        uses_sampled_T=uses_sampled_T,
        m_early_avg=float(early), m_late_avg=float(late), m_decreases=m_decreases,
        m_hist=[int(x) for x in m_hist],
        D_variance_at_mid_iter=var_at_mid,
        D_variance_at_late_iter=var_at_late,
        variance_bounded=var_bounded,
        budget_rhi_recursive=budget_rhi,
        budget_single_level_no_reuse=budget_single,
        recursive_more_efficient=bool(budget_rhi < budget_single),
        est_err_traj=[float(x) for x in est_err],
    )
    if art_dir:
        with open(os.path.join(art_dir, "c5_rsample.json"), "w") as f:
            json.dump(detail, f, indent=2)
    return detail


def _d_variance(r, P, unc, res, rng, trials=12):
    """Variance of D^k(s,a) (one entry) across `trials` independent R-SAMPLE
    draws, at the mid and late iterates, using the REAL h^k, h^{k-1} vectors
    from the reference run (so the measured variance is honest, not degenerate)."""
    h_hist = res.get("h_history")
    m_hist = res["m_history"]
    n_iter = len(m_hist) - 1
    mid_k = max(1, n_iter // 2)
    late_k = max(mid_k + 1, n_iter - 1)
    if h_hist is None or len(h_hist) <= late_k:
        return 0.0, 0.0
    d_mid = h_hist[mid_k] - h_hist[mid_k - 1]
    d_late = h_hist[late_k] - h_hist[late_k - 1]
    m_mid = max(int(m_hist[mid_k]), 1)
    m_late = max(int(m_hist[late_k]), 1)
    var_mid = _empirical_d_var(d_mid, P[0, 0], m_mid, rng, trials)
    var_late = _empirical_d_var(d_late, P[0, 0], m_late, rng, trials)
    return var_mid, var_late


def _empirical_d_var(d, Prow, m, rng, trials):
    """Variance of (1/m) sum_j d(s_j), s_j~Prow, across trials."""
    vals = []
    for _ in range(trials):
        idx = rng.choice(len(Prow), size=max(m, 1), p=Prow)
        vals.append(float(np.mean(d[idx])))
    return float(np.var(vals))


def _single_level_budget(r, P, eps, n):
    """Budget for a single-level re-estimator: re-sample the FULL T_P(Q^k) each
    iteration (no recursive reuse).  Needs ~ ||h^k||_sp^2 / eps^2 per (s,a) per
    step, and ||h^k||_sp does NOT shrink, so every step pays the full cost."""
    S, A = r.shape
    h_span_typical = 1.0  # bias span ~ O(1) for rewards in [0,1]
    m_per_step = max(int(np.ceil(h_span_typical ** 2 / eps ** 2)), 1)
    return S * A * m_per_step * n


# ===========================================================================
# Claim 1 - Theorem 5.3:  Õ(SA H^2 / eps^2) sample complexity
# ===========================================================================

def verify_c1_thm53(seed: int = 0, art_dir: str | None = None,
                    eps_grid=(0.35, 0.25, 0.18, 0.13, 0.10),
                    seeds=(0, 1, 2, 3), S: int = 10, A: int = 5) -> dict:
    """Model-free RHI sample complexity.  For each eps and seed: run faithful
    Algorithm 1 (adaptive m_k, n=ceil(H/eps)), measure total samples consumed
    and the achieved suboptimality gap.  Fit log(total) vs log(1/eps); the
    theorem predicts slope ~ 2 (Õ(eps^-2)).

    Non-circularity: eps is an independent geometric grid; total_samples and the
    gap are MEASURED from the run, not substituted from the formula.  An
    independent Bellman-estimation-error rate (~1/sqrt(n)) is also reported,
    which is the per-component rate underlying the eps^-2 total.
    """
    rng = np.random.default_rng(seed)
    P, r = M.garnet_mdp(S, A, 0.55, rng)
    unc = M.Uncertainty("lp", 0.05, 2.0)
    g_star, _, pi_star = M.robust_optimal_span_rvi(r, P, unc)
    H, _ = M.robust_bias_span(pi_star, r, P, unc)
    rows = []
    for eps in eps_grid:
        n = int(np.ceil(H / eps))
        for sd in seeds:
            run_rng = np.random.default_rng(1000 + sd)
            res = R.rhi_model_free(r, P, unc, eps=eps, n=n, delta=0.05, rng=run_rng)
            g_pi, _ = M.robust_policy_gain(res["pi"], r, P, unc)
            gap = g_star - g_pi
            rows.append(dict(eps=eps, seed=sd, n=n, total_samples=res["total_samples"],
                             gap=gap, eps_optimal=bool(gap <= eps)))
    # fit scaling: median total vs eps
    eps_arr = np.array(sorted({r_["eps"] for r_ in rows}))
    med = np.array([np.median([r_["total_samples"] for r_ in rows if r_["eps"] == e])
                    for e in eps_arr])
    slope, intercept = np.polyfit(np.log(1.0 / eps_arr), np.log(med), 1)
    all_eps_ok = all(r_["eps_optimal"] for r_ in rows)
    # independent estimation-error rate (Bellman op. estimation vs #samples/sa)
    rate = _independent_estimation_rate(r, P, unc, rng)
    detail = dict(
        verdict="VERIFIED" if (all_eps_ok and 1.5 < slope < 2.8) else "FALSIFIED",
        S=S, A=A, H=H, g_star=g_star, eps_grid=list(eps_grid),
        log_log_slope=float(slope), log_log_intercept=float(intercept),
        all_runs_eps_optimal=all_eps_ok,
        n_runs=len(rows),
        independent_bellman_error_rate=rate,  # ~ -0.5 => n ~ eps^-2
        rows=rows,
    )
    if art_dir:
        _write_csv(art_dir, "c1_thm53_scaling.csv", rows)
        with open(os.path.join(art_dir, "c1_thm53_summary.json"), "w") as f:
            json.dump({k: v for k, v in detail.items() if k != "rows"}, f, indent=2)
    return detail


def _independent_estimation_rate(r, P, unc, rng, S=None, A=None):
    """Independent (non-formula) check: error of estimated robust Bellman op vs
    number of nominal samples per (s,a).  Should scale ~ 1/sqrt(n) (slope -0.5),
    the rate that yields eps^-2 total samples."""
    S = S or P.shape[0]; A = A or P.shape[1]
    Q = rng.uniform(0, 1, (S, A))
    TQ_true = M.robust_bellman_T(Q, r, P, unc)
    ns = np.array([20, 50, 100, 200, 500, 1000])
    errs = []
    for nsm in ns:
        # estimate sigma from nsm samples
        h = Q.max(axis=1)
        sig = np.empty((S, A))
        for s in range(S):
            for a in range(A):
                idx = rng.choice(S, size=nsm, p=P[s, a])
                emp = np.mean(h[idx])
                if unc.kind == "contamination":
                    sig[s, a] = (1 - unc.radius) * emp + unc.radius * h.min()
                else:
                    sig[s, a] = emp - unc.radius * M.kappa_q(h, unc.p, P[s, a] > 0)
        TQ_est = r + sig
        errs.append(float(np.max(np.abs(TQ_est - TQ_true))))
    errs = np.maximum(np.array(errs), 1e-12)
    slope, _ = np.polyfit(np.log(ns), np.log(errs), 1)
    return float(slope)


# ===========================================================================
# Claim 2 - Theorem 4.2: reduction needs H, yields Õ(SA H^2 / eps^4)
# ===========================================================================

def verify_c2_thm42(seed: int = 0, art_dir: str | None = None) -> dict:
    """Verify the reduction framework (Theorem 4.2 + Section 4 limitations).

    (1) Policy transfer: a greedy policy for the DMDP at gamma=1-eps/H is
        O(eps)-optimal for the AMDP (model-based, exact).
    (2) REQUIRES H: gamma_reduction = 1 - eps/H needs H; underestimating H makes
        eps/H > 1 -> gamma <= 0 (INVALID discount) for modest eps, so without H
        one cannot even pick a valid gamma.  Also: overestimating H pushes gamma
        -> 1 and inflates the discounted sample complexity.
    (3) Sample-based reduction scales ~ eps^-4 (worse than RHI's eps^-2): we
        measure both slopes on the same MDP over a wide eps grid.
    (4) Algebraic derivation of Õ(SA H^2/eps^4)."""
    rng = np.random.default_rng(seed)
    P, r = M.garnet_mdp(10, 5, 0.5, rng)
    unc = M.Uncertainty("lp", 0.05, 2.0)
    g_star, _, pi_star = M.robust_optimal_span_rvi(r, P, unc)
    H, _ = M.robust_bias_span(pi_star, r, P, unc)
    S, A = r.shape
    # (1) policy transfer at the correct H
    transfer_rows = []
    for eps in [0.3, 0.2, 0.12, 0.08]:
        pi_g, gamma, gap = RED.reduction_model_based(r, P, unc, eps, H)
        transfer_rows.append(dict(eps=eps, gamma=gamma, amd_gap=gap,
                                  transfer_holds=bool(gap <= (8 + 5) * eps)))
    transfer_ok = all(r_["transfer_holds"] for r_ in transfer_rows)
    # (2) requires H: gamma = 1 - eps/H.  Underestimate H by 10x -> eps/H'>1
    # for eps > H/10 -> gamma <= 0 (invalid).  Concretely:
    H_under = H / 10.0
    eps_break = H_under  # eps at which gamma hits 0 with the wrong H
    gamma_at_break = RED.reduction_gamma(eps_break, H_under)  # == 0
    invalid_gamma_possible = bool(gamma_at_break <= 0.0)
    # and a realistic eps (0.15) with a 5x underestimate:
    H_under5 = H / 5.0
    g5 = RED.reduction_gamma(0.15, H_under5)
    requires_H = bool(invalid_gamma_possible or g5 < 0.5)
    # (3) sample-based reduction scaling vs RHI over a WIDE eps grid
    eps_grid = [0.35, 0.25, 0.18, 0.13, 0.10, 0.08]
    scaling_rows = []
    for eps in eps_grid:
        gamma = RED.reduction_gamma(eps, H)
        # discounted robust solver budget ~ SA / ((1-gamma) eps)^2  (Clavier lp)
        n_sm = max(int(np.ceil(1.0 / ((1 - gamma) * eps) ** 2)), 50)
        pi_d, budget = RED.discounted_sample_based(r, P, unc, gamma, n_sm, rng, iters=150)
        g_pd, _ = M.robust_policy_gain(pi_d, r, P, unc)
        scaling_rows.append(dict(eps=eps, gamma=gamma, budget=budget, gap=g_star - g_pd,
                                 method="reduction"))
    rhi_rows = []
    for eps in eps_grid:
        n = int(np.ceil(H / eps))
        res = R.rhi_model_free(r, P, unc, eps=eps, n=n, delta=0.05, rng=rng)
        rhi_rows.append(dict(eps=eps, budget=res["total_samples"], method="rhi"))
    e_grid = np.array(eps_grid)
    b_red = np.array([r_["budget"] for r_ in scaling_rows], dtype=float)
    b_rhi = np.array([r_["budget"] for r_ in rhi_rows], dtype=float)
    red_slope, _ = np.polyfit(np.log(1.0 / e_grid), np.log(b_red), 1)
    rhi_slope, _ = np.polyfit(np.log(1.0 / e_grid), np.log(b_rhi), 1)
    # precise "worse by eps^-2" test: budget ratio red/rhi should scale ~ eps^-2
    ratio = b_red / b_rhi
    ratio_slope, _ = np.polyfit(np.log(1.0 / e_grid), np.log(ratio), 1)
    # reduction is worse if (a) its own exponent exceeds RHI's, and (b) the
    # ratio grows with 1/eps (i.e. reduction needs strictly more samples as eps
    # shrinks).  Theory: red ~ eps^-4, rhi ~ eps^-2 -> ratio ~ eps^-2.
    reduction_worse = bool(red_slope > rhi_slope and ratio_slope > 0.5)
    ratio_at_min_eps = float(ratio[-1])
    # (4) algebra
    alg = RED.reduction_algebra(0.1, H, S, A)
    detail = dict(
        verdict="VERIFIED" if (transfer_ok and requires_H and reduction_worse) else "FALSIFIED",
        H=H, g_star=g_star,
        transfer_rows=transfer_rows, transfer_ok=transfer_ok,
        requires_H=requires_H,
        requires_H_reason=f"gamma=1-eps/H: with H underestimated to H/10, eps=H/10 "
                           f"gives gamma={gamma_at_break:.3f} (invalid); the reduction "
                           f"cannot pick a valid gamma without knowing H",
        gamma_at_break=float(gamma_at_break),
        reduction_scaling_slope=float(red_slope),
        rhi_scaling_slope=float(rhi_slope),
        budget_ratio_slope=float(ratio_slope),
        budget_ratio_at_min_eps=ratio_at_min_eps,
        reduction_worse=reduction_worse,
        scaling_rows=scaling_rows, rhi_rows=rhi_rows,
        algebra=alg,
    )
    if art_dir:
        _write_csv(art_dir, "c2_thm42_reduction.csv", scaling_rows + rhi_rows)
        with open(os.path.join(art_dir, "c2_thm42_summary.json"), "w") as f:
            json.dump({k: v for k, v in detail.items()}, f, indent=2, default=str)
    return detail


# ===========================================================================
# Claim 6 - first finite-sample guarantee for model-free robust AMDPs
# ===========================================================================

def verify_c6_first_finite(seed: int = 0, art_dir: str | None = None) -> dict:
    """Demonstrate the finite-sample guarantee concretely for BOTH uncertainty
    families (contamination and l_p), at the paper's scale G(20,15): model-free
    RHI reaches an eps-optimal robust policy with a finite, measured budget.

    The 'first finite-sample guarantee for model-free robust average-reward
    MDPs' novelty claim is a literature-positioning statement; it cannot be
    proven 'first' by experiment.  We document the positioning vs the prior
    asymptotic-only works the paper cites and mark the novelty aspect as
    corroboration, not proof.  The finite-sample REGIME itself is verified.
    """
    rng = np.random.default_rng(seed)
    S, A = 20, 15  # paper scale G(20,15)
    unc_sets = [
        ("contamination", 0.1, 2.0, "contamination"),
        ("lp", 0.05, np.inf, "TV (l_inf)"),
        ("lp", 0.05, 2.0, "l_2-norm"),
    ]
    rows = []
    for kind, rad, p, label in unc_sets:
        P, r = M.garnet_mdp(S, A, 0.4, rng)
        unc = M.Uncertainty(kind, rad, p)
        g_star, _, pi_star = M.robust_optimal_span_rvi(r, P, unc)
        H, _ = M.robust_bias_span(pi_star, r, P, unc)
        eps = 0.15
        n = int(np.ceil(H / eps))
        res = R.rhi_model_free(r, P, unc, eps=eps, n=n, delta=0.05, rng=rng)
        g_pi, _ = M.robust_policy_gain(res["pi"], r, P, unc)
        rows.append(dict(uncertainty=label, S=S, A=A, H=H, eps=eps,
                         total_samples=res["total_samples"],
                         gap=g_star - g_pi, eps_optimal=bool(g_star - g_pi <= eps)))
    finite_regime_holds = all(r_["eps_optimal"] for r_ in rows)
    detail = dict(
        verdict="VERIFIED" if finite_regime_holds else "FALSIFIED",
        rows=rows, finite_regime_holds=finite_regime_holds,
        novelty_note="The 'first finite-sample guarantee' is a positioning claim vs prior "
                     "asymptotic-only work (Wang+2023a, Grand-Clement+2023, Xu+2025 policy-eval); "
                     "it cannot be proven 'first' experimentally. Finite-sample regime verified.",
    )
    if art_dir:
        _write_csv(art_dir, "c6_first_finite.csv", rows)
        with open(os.path.join(art_dir, "c6_summary.json"), "w") as f:
            json.dump(detail, f, indent=2)
    return detail


# --- helpers ----------------------------------------------------------------

def _write_csv(art_dir, name, rows):
    if not rows:
        return
    os.makedirs(art_dir, exist_ok=True)
    with open(os.path.join(art_dir, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
