"""Generate evidence figures from artifact CSV/JSON for the report + HF Space.

Reads .openresearch/artifacts/* (produced by `bash run.sh`) and writes PNGs to
reports/figures/.  Deterministic; reruns from the fixed command.
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = os.path.join(".openresearch", "artifacts")
OUT = os.path.join("reports", "figures")
os.makedirs(OUT, exist_ok=True)


def _read_csv(name):
    with open(os.path.join(ART, name)) as f:
        return list(csv.DictReader(f))


def fig_c1_scaling():
    rows = _read_csv("c1_thm53_scaling.csv")
    eps = sorted({float(r["eps"]) for r in rows})
    med = [np.median([float(r["total_samples"]) for r in rows if float(r["eps"]) == e])
           for e in eps]
    q25 = [np.percentile([float(r["total_samples"]) for r in rows if float(r["eps"]) == e], 25)
           for e in eps]
    q75 = [np.percentile([float(r["total_samples"]) for r in rows if float(r["eps"]) == e], 75)
           for e in eps]
    eps = np.array(eps)
    slope, intercept = np.polyfit(np.log(1 / eps), np.log(med), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.fill_between(1 / eps, q25, q75, alpha=0.25, label="25–75 percentile")
    ax.plot(1 / eps, med, "o-", lw=2, ms=6, label=f"RHI median (slope={slope:.2f})")
    ref = med[0] * (eps[0] / eps) ** 2
    ax.plot(1 / eps, ref, "k--", lw=1, label=r"$\propto \varepsilon^{-2}$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"target accuracy $1/\varepsilon$"); ax.set_ylabel("total nominal samples")
    ax.set_title("Claim 1 — model-free RHI sample complexity\n" + r"$\tilde{O}(SA\mathcal{H}^2/\varepsilon^2)$")
    ax.legend(framealpha=0.9, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c1_scaling.png"), dpi=130)
    plt.close(fig)
    return slope


def fig_c2_reduction_vs_rhi():
    rows = _read_csv("c2_thm42_reduction.csv")
    red = [(float(r["eps"]), float(r["budget"])) for r in rows if r["method"] == "reduction"]
    rhi = [(float(r["eps"]), float(r["budget"])) for r in rows if r["method"] == "rhi"]
    red.sort(); rhi.sort()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([1 / e for e, _ in red], [b for _, b in red], "s-", lw=2, ms=6,
            label="reduction (discounted)")
    ax.plot([1 / e for e, _ in rhi], [b for _, b in rhi], "o-", lw=2, ms=6,
            label="RHI (direct)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$1/\varepsilon$"); ax.set_ylabel("total samples")
    ax.set_title("Claim 2 — reduction is worse: " + r"$\tilde{O}(\varepsilon^{-4})$ vs $\tilde{O}(\varepsilon^{-2})$")
    ax.legend(framealpha=0.9, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c2_reduction_vs_rhi.png"), dpi=130)
    plt.close(fig)


def fig_c3_lemma51():
    rows = _read_csv("c3_lemma51_instances.csv")
    gaps = np.array([float(r["gap"]) for r in rows])
    spans = np.array([float(r["sp_resid"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.scatter(spans, gaps, s=14, alpha=0.6, label="greedy policy (300 instances)")
    lim = max(spans.max(), gaps.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label=r"bound: $g^*-g^\pi = \mathrm{Sp}(\mathcal{T}_P(Q)-Q)$")
    ax.plot([0, lim], [0, lim / 2], "r:", lw=1, label=r"tighter $\mathrm{Sp}/2$ (would be violated)")
    ax.set_xlabel(r"RHS: $\mathrm{Sp}(\mathcal{T}_P(Q)-Q)$"); ax.set_ylabel(r"LHS: $g^*_P - g^\pi_P$")
    ax.set_title("Claim 3 — Lemma 5.1: every point below the line")
    ax.legend(framealpha=0.9, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c3_lemma51.png"), dpi=130)
    plt.close(fig)


def fig_c4_convergence():
    with open(os.path.join(ART, "c4_thm52_curves.json")) as f:
        curves = json.load(f)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    plotted = 0
    for tag in sorted(curves):
        if "S20" not in tag:  # focus on paper-scale G(20,15)
            continue
        c = curves[tag]
        k = np.array(c["k"])
        sp = np.array(c["span"])
        vi = np.array(c["vi_span"])
        ax.plot(k, sp, lw=1.8, label=f"Halpern ({tag.split('_',2)[-1]})")
        ax.plot(k, vi, lw=1.2, ls="--", alpha=0.7)
        plotted += 1
    if plotted == 0:
        for tag in list(curves)[:3]:
            c = curves[tag]
            ax.plot(c["k"], c["span"], lw=1.5, label=tag)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Halpern iteration $k$"); ax.set_ylabel(r"$\mathrm{Sp}(\mathcal{T}_P(Q^k)-Q^k)$")
    ax.set_title("Claim 4 — exact RHI convergence at G(20,15)\n(solid: Halpern →0; dashed: value iteration)")
    ax.legend(framealpha=0.9, fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c4_halpern_convergence.png"), dpi=130)
    plt.close(fig)


def fig_c5_rsample():
    with open(os.path.join(ART, "c5_rsample.json")) as f:
        d = json.load(f)
    err = np.array(d["est_err_traj"])
    m = np.array(d["m_hist"])
    eps = d["eps"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax1.plot(err, "o-", ms=4, lw=1.5)
    ax1.axhline(eps, color="r", ls="--", lw=1, label=rf"$\varepsilon={eps}$ bound")
    ax1.set_xlabel("RHI iteration $k$"); ax1.set_ylabel(r"$\|T^k-\mathcal{T}_P(Q^k)\|_\infty$")
    ax1.set_title("Proposition C.2: bounded accumulated error"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(m, "s-", ms=5, lw=1.5, color="C2")
    ax2.set_xlabel("RHI iteration $k$"); ax2.set_ylabel(r"per-$(s,a)$ budget $m_k$")
    ax2.set_title(r"$m_k\propto\mathrm{Sp}(d^k)^2$: shrinks as $d^k\to 0$"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c5_rsample.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    s = fig_c1_scaling(); print(f"c1 slope={s:.3f}")
    fig_c2_reduction_vs_rhi(); print("c2 done")
    fig_c3_lemma51(); print("c3 done")
    fig_c4_convergence(); print("c4 done")
    fig_c5_rsample(); print("c5 done")
    print(f"figures in {OUT}")
