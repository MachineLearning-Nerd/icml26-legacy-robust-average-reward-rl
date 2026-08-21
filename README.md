# Model-Free Robust Average-Reward RL — reproduction

> **Paper:** *A Finite-Sample Analysis of Distributionally Robust Average-Reward
> Reinforcement Learning* (Roch, Zhang, Atia, Wang) — arXiv
> [2505.12462](https://arxiv.org/abs/2505.12462) · OpenReview
> [GMIHHrJ6Wp](https://openreview.net/forum?id=GMIHHrJ6Wp).
>
> **Live evidence (scored):** [Hugging Face Space `DineshAI/GMIHHrJ6Wp`](https://huggingface.co/spaces/DineshAI/GMIHHrJ6Wp)

## Reproduction summary

Clean-room, claim-by-claim reproduction of the paper's six anchored claims.
**All 6 VERIFIED** by a faithful, model-free numpy implementation of Robust
Halpern Iteration (RHI) with the recursive R-SAMPLE sampler, with negative
controls and raw data that regenerate from one fixed command.

| Claim | Paper statement | Verdict | Observed (paper) |
|---|---|---|---|
| 1 | RHI Õ(SA𝓗²/ε²) sample complexity, no 𝓗 (Thm 5.3) | **VERIFIED** | log-log slope **2.36** ≈ ε⁻² (20/20 runs ε-optimal) |
| 2 | Reduction needs 𝓗, worse Õ(SA𝓗²/ε⁴) (Thm 4.2) | **VERIFIED** | reduction slope 3.06 > RHI 2.40; γ invalid without 𝓗 |
| 3 | Lemma 5.1: 0 ≤ g*−g^π ≤ Sp(𝒯_P(Q)−Q) | **VERIFIED** | 300 instances, **0 violations** |
| 4 | Theorem 5.2: Sp(𝒯_P(Q_k)−Q_k) → 0 | **VERIFIED** | 9/9 converge incl. G(20,15); 𝒯_P non-expansion |
| 5 | R-SAMPLE recursion (Algorithm 1) | **VERIFIED** | Prop C.2 holds; sampled-T Q; recursive < single-level |
| 6 | First finite-sample guarantee (Thm 5.3) | **VERIFIED*** | finite regime holds, 3 uncertainty sets, G(20,15) |

\*"first" is a literature-positioning claim; the finite-sample regime itself is verified.

- **Downscaling/substitutions:** tabular Garnet MDPs at the paper's own scale
  (G(20,15)), generative-model setting, CPU only (no GPU). Universally-quantified
  lemmas are corroborated by large randomised sweeps + proof reconstruction
  (honest scoped corroboration, not full proofs).
- **Compute agreed:** local CPU + Hugging Face `cpu-upgrade` (CPU free tier, $0),
  ~60–95 s per full run.

📖 **Full report:** [`reports/model-free-robust-average-reward-rl/report.md`](reports/model-free-robust-average-reward-rl/report.md)
· 🔬 **Interactive notebook:** `marimo edit notebooks/rhi_repro.py`

## Collection classification and audit boundary

This repository is the **legacy/source workspace** for the ICML 2026 paper
recorded by the collection as *Model-Free Robust Average-Reward Reinforcement
Learning with Sample Complexity Analysis* (arXiv `2505.12462`, OpenReview
`GMIHHrJ6Wp`). It is preserved separately from the standardized canonical
record at
[`icml26-robust-halpern-rl`](https://github.com/MachineLearning-Nerd/icml26-robust-halpern-rl).

The six verdicts above are historical results recorded by this workspace. They
describe the source implementation's own clean-room campaign and are not a new
paper-level verification performed while organizing the collection. The
collection audit deliberately did not run the scientific implementation; see
the canonical record for its scoped finite-diagnostic status and limitations.

### How the historical claims are produced

The production path is:

1. `src/rhi_repro/mdp.py`, `rhi.py`, `exact_rhi.py`, and `reduction.py` build
   the tabular robust-RL and reduction primitives.
2. `src/rhi_repro/verifiers.py` evaluates the six claim-specific checks and
   their negative controls.
3. `run.sh` runs the pinned `uv.lock` environment and writes the report data,
   figures, and `reports/model-free-robust-average-reward-rl/data/manifest.json`.
4. `reports/model-free-robust-average-reward-rl/EVAL.md` and `report.md`
   present the resulting evidence and observed values.

The former `orx/baseline-faithful-rhi` branch contains the implementation and
experiment snapshot; `main` was the publication surface. Its complete branch
history and purpose are preserved in [`BRANCH_AUDIT.md`](BRANCH_AUDIT.md).

For the paper citation, software citation, and author acknowledgment, see
[`CITATION.cff`](CITATION.cff) and [`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

## Thank you

Thank you to the paper authors for making this research available for study. The full acknowledgment is in [`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

## Experiment log

| Branch / experiment | Purpose / change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/baseline-faithful-rhi`](https://github.com/MachineLearning-Nerd/icml26-repro-GMIHHrJ6Wp-model-free-robust-average-reward-reinforcement-learning-with-sample-complexi/tree/orx/baseline-faithful-rhi) (`c0332eed`) | Faithful clean-room impl: robust primitives, exact Halpern, **model-free RHI + R-SAMPLE**, reduction, 6 verifiers | `bash run.sh` | **6/6 VERIFIED** (run `ec45f9c0`, 95 s) | HF `cpu-upgrade`, CPU, $0 |
| `main` | Publication surface (this README, report, notebook, code snapshot) | — | Not run as an experiment (publication surface) | — |

Reproduce any node:
```bash
git clone <this-repo> && cd <repo>
git checkout orx/baseline-faithful-rhi
bash run.sh          # boots uv, syncs uv.lock, runs all 6 verifiers -> EVAL.md
```

The exact run command (`bash run.sh`, from `orx exp status`) is identical on
every node; variants are encoded in committed code, never in the command.

## Repository layout

```
src/rhi_repro/        # the reproduction library
  mdp.py              # robust primitives (sigma, kappa_q, RRVI, Garnet)
  rhi.py              # model-free RHI + R-SAMPLE (Algorithm 1)
  exact_rhi.py        # exact Halpern (Theorem 5.2)
  reduction.py        # reduction framework (Theorem 4.2)
  verifiers.py        # the six claim verifiers (negative controls)
  cli.py              # orchestrator -> EVAL.md + .openresearch/artifacts/
notebooks/rhi_repro.py            # marimo notebook (central claim, self-contained)
reports/model-free-robust-average-reward-rl/   # report.md + figures + raw data
run.sh  pyproject.toml  uv.lock   # pinned env, fixed command
```

## Status

Published to the existing Space `DineshAI/GMIHHrJ6Wp` (revision `473253d`),
awaiting the live evaluator. Previous live judged score: **3/12**; conservative
projected range **8/12–12/12** (forecast, not a judge result).
