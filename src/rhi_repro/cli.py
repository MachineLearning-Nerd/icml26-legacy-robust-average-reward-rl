"""CLI: runs all six claim verifiers and writes EVAL.md + raw artifacts.

Usage:  python -m rhi_repro.cli   (the fixed run command is `bash run.sh`)
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

from . import verifiers as V

ART = os.path.join(".openresearch", "artifacts")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    os.makedirs(ART, exist_ok=True)
    print("=" * 72, flush=True)
    print("Reproduction: A Finite-Sample Analysis of Distributionally Robust", flush=True)
    print("Average-Reward RL (arXiv 2505.12462)", flush=True)
    print("=" * 72, flush=True)
    print(f"host={socket.gethostname()} py={platform.python_version()} "
          f"np={np.__version__} sha={_git_sha()[:10]}", flush=True)

    claims = [
        ("c1_thm53_sample_complexity", V.verify_c1_thm53, dict(art_dir=ART)),
        ("c2_thm42_reduction", V.verify_c2_thm42, dict(art_dir=ART)),
        ("c3_lemma51", V.verify_c3_lemma51, dict(n_mdp=300, art_dir=ART)),
        ("c4_thm52_halpern", V.verify_c4_thm52, dict(art_dir=ART)),
        ("c5_rsample", V.verify_c5_rsample, dict(art_dir=ART)),
        ("c6_first_finite", V.verify_c6_first_finite, dict(art_dir=ART)),
    ]
    summary = {}
    for name, fn, kw in claims:
        print(f"\n--- {name} ---", flush=True)
        ts = time.time()
        try:
            res = fn(**kw)
            res["_runtime_s"] = round(time.time() - ts, 2)
            res["_claim"] = name
            summary[name] = res
            v = res.get("verdict", "?")
            print(f"  verdict: {v}   ({res['_runtime_s']}s)", flush=True)
            for k in res:
                if k in ("verdict",) or k.startswith("_"):
                    continue
                val = res[k]
                if isinstance(val, (int, float, bool, str)):
                    print(f"    {k} = {val}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary[name] = dict(verdict="BLOCKED", error=str(e),
                                 _runtime_s=round(time.time() - ts, 2))

    total = round(time.time() - t0, 1)
    _write_eval(summary, total)
    _write_manifest(summary, total)
    # exit nonzero if any non-VERIFIED (so the run "fails" the gate)
    bad = [n for n, r in summary.items() if r.get("verdict") not in ("VERIFIED", "FALSIFIED")]
    print(f"\nTotal runtime: {total}s", flush=True)
    if bad:
        print(f"BLOCKED claims: {bad}", flush=True)
    print("Wrote EVAL.md and artifacts under .openresearch/artifacts/", flush=True)


def _write_eval(summary, total):
    path = os.path.join("EVAL.md")
    lines = ["# EVAL.md — reproduction of arXiv 2505.12462", "",
             f"- Total runtime: {total}s", f"- Git SHA: {_git_sha()[:10]}",
             f"- Date: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", "",
             "| Claim | Verdict | Key result | Runtime |", "|---|---|---|---|"]
    for name, res in summary.items():
        v = res.get("verdict", "?")
        key = _key_result(name, res)
        lines.append(f"| {name} | **{v}** | {key} | {res.get('_runtime_s','?')}s |")
    lines += ["", "## Raw artifacts", "",
              "All raw CSV/JSON outputs are under `.openresearch/artifacts/`."]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _key_result(name, res):
    try:
        if name.startswith("c1"):
            return (f"slope={res.get('log_log_slope'):.2f} (expect ~2), "
                    f"all eps-optimal={res.get('all_runs_eps_optimal')}")
        if name.startswith("c2"):
            return (f"transfer_ok={res.get('transfer_ok')}, requires_H={res.get('requires_H')}, "
                    f"red_slope={res.get('reduction_scaling_slope'):.2f} > "
                    f"rhi_slope={res.get('rhi_scaling_slope'):.2f}")
        if name.startswith("c3"):
            return (f"{res.get('n_checked')} instances, "
                    f"violations={res.get('n_rhs_violations')}, "
                    f"NC effective={res.get('negative_control_effective')}")
        if name.startswith("c4"):
            conv = [r["halpern_converges"] for r in res.get("instances", [])]
            return (f"halpern converges {sum(conv)}/{len(conv)}; "
                    f"non-expansion ratio={res.get('max_contraction_ratio'):.2f}")
        if name.startswith("c5"):
            return (f"Prop C.2 holds={res.get('prop_C2_all_within_eps')}, "
                    f"uses_sampled_T={res.get('uses_sampled_T')}, "
                    f"recursive<single={res.get('recursive_more_efficient')}")
        if name.startswith("c6"):
            return f"finite regime holds={res.get('finite_regime_holds')}"
    except Exception:
        pass
    return str(res.get("error", ""))[:80]


def _write_manifest(summary, total):
    man = dict(git_sha=_git_sha(), total_runtime_s=total,
               host=socket.gethostname(), python=platform.python_version(),
               numpy=np.__version__, verdicts={n: r.get("verdict") for n, r in summary.items()})
    with open(os.path.join(ART, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)


if __name__ == "__main__":
    main()
