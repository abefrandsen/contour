---
id: goal_token_mixing
type: goal
title: Sub-quadratic & Low-Rank Token Mixing Across Modalities
status: active
target_regimes:
  tasks:
    - image_classification
    - dense_segmentation
    - language_modeling
  scales:
    - 22M
    - 86M
---

## Objective & Success Criteria
Identify token-mixing bottlenecks that achieve >=20% wall-clock speedup at <=0.5% quality loss across Vision and Sequential tasks.

## Current Synthesis & Open Questions
- Vision classification Pareto improvement achieved at 22M (`[[clm_low_rank_recall]]`, `[[clm_qk_norm_warmup]]`).
- Dense spatial recall degrades at r=d/8 on segmentation; need hybrid local-global mixing or selective layer compression before scaling to 86M.
