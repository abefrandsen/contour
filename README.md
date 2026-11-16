# Contour — Topographical & Epistemic ML Research Framework

**Contour** maps your empirical ML research as a 3-layer epistemic graph designed for both human researchers (via Git + Obsidian) and AI agents:

- **Layer 1 (Empirical Ground Truth — `Theodolite` & `fieldlogs/`):** [tools/theodolite.py](tools/theodolite.py) logs standardized survey coordinates (`run["meta"]`) + freeform model configs (`run["cfg"]`) into **Aim** (or JSON field bundles on isolated cluster nodes). `./contour sync` distills runs into Markdown cards under `fieldlogs/log_*.md`.
- **Layer 2 (Epistemic Topography — `cairns/` & `expeditions/`):** Obsidian-compatible Markdown files under `expeditions/expd_*.md` (macro research campaigns) and `cairns/crn_*.md` (falsifiable claims & principles) linked to fieldlogs via typed edges (`supports`, `contradicts`, `qualifies`).
- **Layer 3 (Navigation & Ideation — `./contour` CLI):** [tools/contour.py](tools/contour.py) provides instant graph retrieval and surfaces candidates across the **6 First-Principles Scientific Moves**, paired with [.agent/skills/contour-research/SKILL.md](.agent/skills/contour-research/SKILL.md).

---

## CLI Navigation (`./contour`)

```bash
# 1. View the topographical DAG (Expeditions -> Cairns -> FieldLogs)
./contour map

# 2. Plot the Pareto ridgeline on an evaluation benchmark
./contour ridge --eval imagenet1k_val

# 3. Take bearings: run the 6 Scientific Move detectors for next experiments
./contour bearings

# 4. Zoom in on any Expedition, Cairn, or FieldLog
./contour inspect crn_low_rank_recall
./contour inspect log_20261115_a8f3

# 5. Check for terrain drift (stale cairns or unlinked/invalidated fieldlogs)
./contour drift

# 6. Scaffold a new Expedition, Cairn, or manual FieldLog
./contour new cairn spectral_reg --title "Low-rank QKV projections act as implicit spectral regularizers on small datasets"
```

---

## Logging a Training Run (`tools/theodolite.py`)

```python
from tools.theodolite import Theodolite

with Theodolite(
    exp_name="vit_s_qknorm_lowrank_combo",
    task="image_classification",
    dataset_train=["imagenet1k_train_v1", "webvision_filtered_v2"],
    dataset_eval=["imagenet1k_val", "imagenet_v2"],
    arch_family="vit",
    arch_scale="22M",
    intervention="Combine middle-block rank d/8 + QK-RMSNorm",
    parent_runs=["log_20261115_a8f3", "log_20261115_b72e"],
    motivated_by=["crn_low_rank_recall", "crn_qk_norm_warmup"],
    raw_config=cfg_dict,
) as run:
    run.track(80.5, name="top1", context={"subset": "imagenet1k_val"})
    run.log_checkpoint("best", "/data/checkpoints/combo/best.pt")
```

Then distill finished runs into `fieldlogs/*.md` and auto-flag parent cairns for synthesis:

```bash
./contour sync
```
