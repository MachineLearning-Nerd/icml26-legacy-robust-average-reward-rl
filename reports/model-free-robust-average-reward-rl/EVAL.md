# EVAL.md — reproduction of arXiv 2505.12462

- Total runtime: 64.0s
- Git SHA: 34db9a411a
- Date: 2026-07-26T10:31:06+0530

| Claim | Verdict | Key result | Runtime |
|---|---|---|---|
| c1_thm53_sample_complexity | **VERIFIED** | slope=2.36 (expect ~2), all eps-optimal=True | 0.15s |
| c2_thm42_reduction | **VERIFIED** | transfer_ok=True, requires_H=True, red_slope=3.06 > rhi_slope=2.40 | 1.88s |
| c3_lemma51 | **VERIFIED** | 300 instances, violations=0, NC effective=True | 4.89s |
| c4_thm52_halpern | **VERIFIED** | halpern converges 9/9, VI control fails | 56.65s |
| c5_rsample | **VERIFIED** | Prop C.2 holds=True, uses_sampled_T=True, recursive<single=True | 0.04s |
| c6_first_finite | **VERIFIED** | finite regime holds=True | 0.37s |

## Raw artifacts

All raw CSV/JSON outputs are under `.openresearch/artifacts/`.