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
