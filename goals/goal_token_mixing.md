---
title: Sub-quadratic & Low-Rank Token Mixing Across Modalities
status: active
---

## Objective & Success Criteria
Identify token-mixing bottlenecks that achieve >=20% wall-clock speedup at <=0.5% quality loss across Vision and Sequential tasks.

## Current Synthesis
- Vision classification Pareto improvement achieved at 22M (`[[clm_low_rank_recall]]`, `[[clm_qk_norm_warmup]]`).
- Dense spatial recall degrades at r=d/8 on segmentation; need hybrid local-global mixing or selective layer compression before scaling up.

## Open Questions
- Can a local window residual recover small-object ADE20K mIoU under r=d/8 middle-block compression?
- Does QK-RMSNorm + 2x LR still hold without shortening warmup (leave-one-out warmup_epochs=5)?
