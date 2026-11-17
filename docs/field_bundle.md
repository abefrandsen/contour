# Field Bundle v1 (`contour.field_bundle/v1`)

A **Field Bundle** is the JSON contract between any training/evaluation platform and Contour.
Any platform (a `contour.Run`-instrumented script, an Aim/W&B exporter, a SLURM epilogue, or a notebook) emits one JSON file per run, and `contour sync` validates it, archives a normalized copy under `bundles/`, distills it into a `fieldlogs/log_*.md` card, and auto-computes the `vs_parent` diff when a parent run is present.

## Required Fields

```json
{
  "schema": "contour.field_bundle/v1",
  "run_id": "0012a98b11c0...",
  "status": "final",
  "created_at": "2026-11-10T12:00:00Z",
  "finished_at": "2026-11-10T16:30:00Z",
  "facets": {
    "task": "image_classification",
    "arch_family": "vit",
    "arch_scale": "22M",
    "tier": "proxy",
    "seed": 0
  },
  "data": {
    "train": ["imagenet1k_train_v1"],
    "eval": ["imagenet1k_val", "imagenet_v2"]
  },
  "metrics": {
    "imagenet1k_val/top1": 79.7
  }
}
```

### Rules
- `status`: `"running"`, `"final"`, or `"failed"`. Terminal runs (`final` / `failed`) require `finished_at` and are immutable once archived unless `contour sync --force` is passed. `final` runs must report at least one finite scalar under `metrics`.
- `run_id`: 6–128 chars of `[A-Za-z0-9_.:-]`, globally unique (e.g. a UUID).
- `facets.tier`: `"probe"`, `"proxy"` (default), or `"full"` (legacy aliases `"recce"`, `"approach"`, `"summit"` are normalized automatically).
- `interventions`: list of atomic lowercase tags matching `^[a-z0-9][a-z0-9_.\-=+]*$` (e.g. `["qk_rmsnorm", "lr_peak=2x", "warmup=0.5x"]`).
- `telemetry`: optional internal scalars (e.g. `{"attn/entropy_mid": 2.4}`).
- `compute`: optional `gpu_hours`, `num_gpus`, `wall_seconds`, `flops`, `params`, `tokens_or_samples`.
- `code`: optional `repo`, `branch`, `commit`, `dirty`, `dirty_hash`, `dirty_patch`, `untracked`.
- `launch`: optional `cmd`, `host`, `job_id`, `node`.
- `lineage`: optional `parents` (run_ids or `log_*` ids) and `motivated_by` (`crn_*` or `expd_*`).
- `artifacts`: optional map of tag → URI string.

## Field Ownership

Once ingested into `fieldlogs/log_*.md`:
- **Machine-owned** keys (`status`, `tier`, `metrics`, `telemetry`, `compute`, `interventions`, `parent_runs`, `vs_parent`, etc.) are updated from the bundle and parent lineage.
- **Human-owned** keys (`title`, `validity`, `validity_note`, `tldr`, `tags`, ...) and the entire Markdown body are preserved across re-syncs.
