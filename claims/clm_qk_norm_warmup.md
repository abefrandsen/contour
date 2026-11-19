---
title: QK-RMSNorm allows 2x higher peak learning rate and 50% shorter warmup in ViT
status: supported
verification: agent_audited
verified_hash: 6ab1a5a2
audit_note: Causal necessity ablation 5a913f62 and synthesis 88fb406c verified; warmup_epochs=5
  solo ablation still open.
goals:
- '[[goal_token_mixing]]'
supports:
- '[[run_20261115_b72e881a]]'
- '[[run_20261116_5a913f62]]'
- '[[run_20261116_88fb406c]]'
contradicts: []
qualifies: []
---

## Hypothesis & Mechanism
Bounding query-key dot products via RMSNorm prevents early attention logit explosion, allowing shorter warmup and higher peak learning rates.

## Evidence Synthesis
- **Holds:** ViT-S (22M) on ImageNet-1k (`[[run_20261115_b72e881a]]`, +0.60% top-1) and combined with r=d/8 low-rank projections (`[[run_20261116_88fb406c]]`, 80.25% top-1 in 14.3 GPU hours).
- **Breaks / Causal Necessity:** Stripping `qk_norm` while keeping 2x LR and 0.5x warmup (`[[run_20261116_5a913f62]]`) triggers an epoch-8 gradient spike and -5.6% top-1 collapse.
- **Untested / Caveats:** Leave-one-out ablation stripping `warmup_epochs=5` alone; ViT-B/L scales (>= 86M).
