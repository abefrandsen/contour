# Contour — Lightweight Epistemic ML Research Framework

**Contour** maintains an easily navigable research state built on top of many training runs, designed for both human researchers (Markdown + Git + Obsidian) and AI agents:

1. **Empirical Ground Truth (`fieldlogs/log_*.md` & `bundles/*.json`):** Any training platform (or `from contour import Run`) emits a portable `contour.field_bundle/v1` JSON bundle. `./contour sync` validates it, archives the bundle, distills a Markdown card in `fieldlogs/`, and **automatically computes `vs_parent`** (metric deltas, `single_logical_diff` confound check, and optional `noise.yaml` noise-floor SNR).
2. **Research Claims & Campaigns (`cairns/crn_*.md` & `expeditions/expd_*.md`):** Obsidian-compatible Markdown files tracking overarching research campaigns (`expeditions/`) and falsifiable claims (`cairns/`) linked directly to fieldlogs via `supports`, `contradicts`, and `qualifies`.
3. **Navigation & Experiment Generation (`./contour` CLI):** Instant graph retrieval, on-the-fly multi-run comparisons (`./contour compare`), Pareto frontiers (`./contour ridge`), and **6 Scientific Move detectors (`./contour bearings`)** that mine your runs to propose the highest-leverage next experiments.

---

## CLI Navigation (`./contour`)

```bash
# 1. View the research DAG (Expeditions -> Cairns -> FieldLogs)
./contour map

# 2. Plot the Pareto frontier (quality vs GPU hours) on an eval benchmark
./contour ridge --eval imagenet1k_val

# 3. Generate next experiment candidates across the 6 Scientific Moves
./contour bearings

# 4. Zoom in on any Expedition, Cairn, or FieldLog (shows embedded vs_parent diff)
./contour inspect crn_low_rank_recall
./contour inspect log_20261115_a8f3

# 5. Compare arbitrary control vs treatment runs on the fly (single- or multi-seed)
./contour compare log_20261110_0012 --vs log_20261115_b72e

# 6. Link a FieldLog to a Cairn
./contour link log_20261115_b72e crn_qk_norm_warmup --relation supports

# 7. Check for hygiene & confounded comparisons
./contour drift

# 8. Optional: Seed the noise-floor registry (noise.yaml) from repeated baseline runs
./contour noise seed image_classification/imagenet1k_val/22M --from log_base_s0 log_base_s1 log_base_s2

# 9. Scaffold a new Expedition, Cairn, or manual FieldLog
./contour new cairn spectral_reg --title "Low-rank QKV projections act as implicit spectral regularizers on small datasets"
```

---

## Logging a Training Run (`from contour import Run`)

```python
from contour import Run

with Run(
    exp_name="vit_s_qknorm_lowrank_combo",
    task="image_classification",
    dataset_train=["imagenet1k_train_v1"],
    dataset_eval=["imagenet1k_val", "imagenet_v2"],
    arch_family="vit",
    arch_scale="22M",
    tier="proxy",  # "probe" | "proxy" | "full"
    interventions=["attn_qkv_rank=r48", "blocks=4-9", "qk_rmsnorm", "lr_peak=2x"],
    parent_runs=["log_20261110_0012"],
    motivated_by=["crn_low_rank_recall", "crn_qk_norm_warmup"],
    raw_config=cfg_dict,
) as run:
    run.track(80.5, name="top1", context={"subset": "imagenet1k_val"})
    run.log_checkpoint("best", "/data/checkpoints/combo/best.pt")
```

Then ingest finished runs into `fieldlogs/*.md`, auto-compute `vs_parent` diffs, and flag motivated cairns for synthesis:

```bash
./contour sync
```

---

## Core Concepts

| Concept | Meaning | Lives in |
|---|---|---|
| **Expedition** | A research campaign with a goal (*North Star*) and open questions. | `expeditions/expd_*.md` |
| **Cairn** | A falsifiable claim or hypothesis (`active`, `supported`, `falsified`) linked to `fieldlogs/` via `supports`, `contradicts`, and `qualifies`. | `cairns/crn_*.md` |
| **FieldLog** | The machine-written record of one run (`status`, `tier: probe\|proxy\|full`, `metrics`, `interventions`, `vs_parent`), paired with human/agent notes in the Markdown body. | `fieldlogs/log_*.md` |
| **Field Bundle** | The portable JSON contract (`contour.field_bundle/v1`) emitted by `contour.Run` or any training platform. | `bundles/*.json` |
| **`vs_parent` & Evidence Grade (`A`/`B`/`C`)** | Auto-computed comparison on every child FieldLog (and via `./contour compare`). **A** = single logical diff (`single_logical_diff: true`), iso-compute, and clears noise floor (`SNR >= 2`). **B** = controlled single diff without a noise-floor check or within noise. **C** = confounded (multiple facets changed, e.g. dataset + architecture), `probe` tier, or non-usable. | `fieldlogs/log_*.md` frontmatter |
| **Noise Floor (`noise.yaml`)** | Optional registry storing seed-to-seed standard deviation per `(task/eval/scale)` regime so single-seed runs can be judged against baseline variance (`clears_noise`). | `noise.yaml` |
