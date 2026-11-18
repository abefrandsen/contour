# Contour — Survey, Map & Explore Your ML Research Landscape

![Contour Banner](assets/pixel_art/banner_pixel_1920x1080_diffuse.png)

**Contour** maintains a clean, low-friction empirical ML research state in plain Markdown and Git across three node types: **Goals** (`goals/goal_*.md`), **Claims** (`claims/clm_*.md`), and **Runs** (`runs/run_*.md`).

1. **Research State Integrity (Verification & Hygiene):**
   - **Single-File Runs (`runs/run_*.md`):** `from contour import Run` writes machine-owned `config`, `compute`, `code`, `metrics`, and hash-checked `vs_parent` directly to Markdown while preserving human/agent notes.
   - **Three-Tier Verification Funnel:**
     1. **Mechanical (`contour check`):** Deterministic rules (`C001`–`C009`) enforcing metric conventions, config/facet diff accountability, staleness (`inputs_hash` & `verified_hash`), and claim rigor.
     2. **Agentic (*Critic* workflow + `contour verify --agent`):** Adversarial AI audit checking for hidden confounds, scope over-generalization, and alternative explanations.
     3. **Selective Human (`contour review` + `contour verify --human`):** High-signal inbox surfacing only claims awaiting sign-off, stale verified claims, and dirty-code frontier runs.
2. **State-Conditioned Idea Generation (`contour ideate`):**
   - Progressive introspection (`status`, `runs --tree`, `compare`, `frontier`, `show`) plus 6 deterministic Scientific Moves that surface confounds to isolate, tensions to probe, verified or compounding marginal wins to combine, multi-tag ablations, untested regimes, and frontier bottlenecks.

---

## Installation

```bash
pip install -e /path/to/contour
```

Inside **any ML project repository**:

```bash
cd ~/projects/my-ml-repo

# Creates ./contour/ (contour.yaml, goals/, claims/, runs/)
contour init

# Optional:
#   contour init .              # Initialize at repo root instead of ./contour/
#   contour init --with-skill   # Install .agent/skills/contour-research/SKILL.md
```

Because `contour` automatically discovers `contour.yaml` in the current directory, any ancestor, or a `contour/` (`.contour/`, `research/`) subdirectory, you can run `contour` commands and `from contour import Run` from anywhere inside your repo.

---

## Quickstart: Day-to-Day Research Loop

### 1. Frame a Goal and a Falsifiable Claim
```bash
contour init --with-skill
contour new goal token_mixing --title "Sub-quadratic & Low-Rank Token Mixing"
contour new claim qk_norm_warmup --title "QK-RMSNorm allows 2x peak LR and 50% shorter warmup in ViT"
```

### 2. Log Single-Seed Training Runs Directly to `runs/`
Configure your metric vocabulary and single-seed `min_delta` thresholds in `contour.yaml`:

```yaml
contour: 1
iso_tolerance: {params: 0.02, gpu_hours: 0.10}
ignored_config_keys: [seed, num_workers, log_interval, eval_interval, checkpoint_dir]
metrics:
  imagenet1k_val/top1: {direction: max, kind: quality, primary: true, min_delta: 0.20}
  val/loss:            {direction: min, kind: quality, min_delta: 0.005}
  sys/throughput_imgs_sec: {direction: max, kind: cost, min_delta: 50.0}
```

Wrap your training script with `contour.Run`:

```python
from contour import Run

with Run(
    title="ViT-S/16 Full-Rank Baseline",
    task="image_classification",
    arch_family="vit",
    arch_scale="22M",
    dataset_train=["imagenet1k_train_v1"],
    config={"lr": 0.001, "warmup_epochs": 10, "qk_norm": False},
    motivated_by=["goal_token_mixing"],
) as base_run:
    base_run.track(79.7, name="top1", context={"subset": "imagenet1k_val"})

with Run(
    title="ViT-S/16 QK-RMSNorm + 2x LR",
    task="image_classification",
    arch_family="vit",
    arch_scale="22M",
    dataset_train=["imagenet1k_train_v1"],
    interventions=["lr=0.002", "warmup_epochs=5", "qk_norm=true", "qk_rmsnorm"],
    parent_runs=[base_run.id],
    motivated_by=["clm_qk_norm_warmup"],
    config={"lr": 0.002, "warmup_epochs": 5, "qk_norm": True},
) as treat_run:
    treat_run.track(80.3, name="top1", context={"subset": "imagenet1k_val"})
```

On exit, `Run` writes `runs/run_*.md` and computes `vs_parent` (`control_level`, `verdict`, `config_diff`, `unaccounted`, `unrealized`, and per-metric `signal`).

### 3. Synthesize, Check & Verify
```bash
contour runs --tree
contour link <run_id> clm_qk_norm_warmup
contour check --fix --strict
contour verify clm_qk_norm_warmup --agent --note "Clean single-diff control; scoped to ViT-22M classification."
contour review
```

### 4. Generate Next Experiments
```bash
contour frontier --task image_classification
contour ideate --goal goal_token_mixing
```

---

## CLI Reference (`contour`)

```bash
# Setup & nodes
contour init [path] [--with-skill]
contour new <goal|claim|run> <slug> --title "..."

# Evidence linking
contour link <run_id> <claim_id> [--as supports|contradicts|qualifies]

# Principle 1: Verification & Hygiene
contour check [--strict] [--fix]
contour verify <claim_id> (--agent | --human) [--note "..."]
contour review

# Principle 2a: Introspection
contour status
contour show <node_id>
contour runs [--tree] [--goal G] [--claim C] [--task T] [--eval E] [--arch A] [--intervention S] [--control L] [--verdict V]
contour compare <ctrl_id> --vs <treat_id>
contour frontier [--task T] [--metric M]

# Principle 2b: Ideation
contour ideate [--goal G]
```
*(Pass `--json` to any command for structured output.)*
