---
title: Middle-block attention rank can be compressed to d/8 for global pooling, but
  degrades dense spatial recall
status: supported
verification: human_verified
verified_hash: 07fd4b16
audit_note: Clean single-diff control (255694de vs 0012a98b); segmentation failure
  scoped.
goals:
- '[[goal_token_mixing]]'
supports:
- '[[run_20261116_255694de]]'
- '[[run_20261116_88fb406c]]'
contradicts: []
qualifies:
- '[[run_20261112_44b1c78e]]'
- '[[run_20261114_f09a331d]]'
- '[[run_20261115_a8f3910c]]'
- '[[run_20261116_7dc3039d]]'
- '[[run_20261116_d0e1d0e9]]'
- '[[ana_20261117_e4b2a901]]'
---

## Hypothesis & Mechanism
Early blocks act as local feature extractors and final blocks aggregate class tokens, whereas middle blocks perform redundant global routing on a low-dimensional manifold.

## Evidence Synthesis
- **Holds:** De-confounded pure ImageNet-1k run (`[[run_20261116_255694de]]`) cuts GPU hours by 21.7% (+25% throughput) at -0.12% top-1 (within threshold tau=0.20). Combined with QK-RMSNorm + 2x LR (`[[run_20261116_88fb406c]]`), it reaches 80.25% top-1 (+0.67% over parent).
- **Breaks:** Collapses at r=d/16 (`[[run_20261112_44b1c78e]]`, -2.8% top-1) and drops -2.9 mIoU on ADE20K segmentation (`[[run_20261116_7dc3039d]]` vs `[[run_20261116_d0e1d0e9]]`); object-size failure-mode analysis (`[[ana_20261117_e4b2a901]]`) shows the drop is concentrated on small objects (-6.4 mIoU) while large objects stay nearly flat (-0.4 mIoU). Acts as an implicit regularizer on CIFAR-100 (`[[run_20261114_f09a331d]]`). Earlier WebVision mix run (`[[run_20261115_a8f3910c]]`) was confounded by training dataset shift.
- **Untested / Caveats:** ViT-B/L scales (>= 86M) and autoregressive sequence recall.
