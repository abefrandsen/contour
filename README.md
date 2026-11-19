# Contour — Survey, Map & Explore Your ML Research Landscape

![Contour Banner](assets/pixel_art/banner_pixel_1920x1080_diffuse.png)

**Contour** maintains a clean, minimal empirical ML research state $S$ in plain Markdown and Git across four directories—**Goals** (`goals/`), **Claims** (`claims/`), **Runs** (`runs/` + `runs/configs/`), and **Analyses** (`analyses/`)—governed by explicit **Invariants** ($\mathcal{I}$) and **Operators** ($T, \mathcal{O}$). See [docs/architecture.md](docs/architecture.md) for the full specification.

1. **Clean, Append-Only Evidence Layer (`runs/` + `analyses/`):**
   - **Lean Training Runs (`runs/run_*.md`):** `from contour import Run` writes concise ~20-line Markdown records with tracked sidecar configs (`runs/configs/<id>.yaml`), compute, source-only git provenance (excluding the Contour state directory itself), artifact pointers, and metrics—computing structured diffs (`config_diff`, `config_blocks`, `code_diff`, `compute_diff`, advisory `flags`) dynamically in memory.
   - **Post-Hoc Evals & Ad-Hoc Analyses (`analyses/ana_*.md`):** `from contour import Analysis, Eval` logs reproducible evaluations and diagnostic probes applied to existing runs/checkpoints (ad-hoc scripts in gitignored `analyses/scripts/` default to `exploratory` for `qualifies`; promoting/tracking a script in Git upgrades it to `clean` for `supports`).
2. **Three-Tier Invariants:**
   - **Tier 1 — Structural (`ERROR`):** `schema-error` and `tainted-evidence` (quarantining a flawed run with `bugged: true` automatically propagates taint downstream).
   - **Tier 2 — Epistemic & Quiescence (`WARN`):** `overclaimed-status` (clean `supports`, no unresolved `contradicts`; `qualifies` carries no verification weight), `stale-verification` (content-hashes Claim frontmatter, **Claim Markdown body prose**, and linked evidence/subjects), plus work queues (`unsynthesized-evidence`, `todo-placeholder`, `stuck-job`, `undeclared-metric`, `missing-artifact`).
   - **Tier 3 — Semantic (Critic + Human):** Adversarial audit (`contour verify --agent`) and selective human sign-off surfaced in `contour brief` (`contour verify --human`).
3. **Focused Daily Observables:**
   - **`contour brief`:** Complete morning cockpit (state counts, Goal $\to$ Claim status tree, live `running` jobs, Human Review Queue, `## Open Questions`, ideation highlights).
   - **`contour report`:** Ephemeral, verified truth bundles with Integrity Banner, `--strict` gating, preliminary `--runs`/`--analyses` mode, Provenance Appendix, and `--out <dir>` portable figure export.
   - **`contour ideate`:** Dedicated next-experiment moves driven directly by `config_diff`, tensions, open questions, and Pareto frontiers.

---

## Installation

```bash
pip install -e /path/to/contour
```

Inside **any ML project repository**:

```bash
cd ~/projects/my-ml-repo

# Creates ./contour/ (contour.yaml, goals/, claims/, runs/configs/, analyses/ + .gitignore)
contour init

# Optional:
#   contour init .              # Initialize at repo root instead of ./contour/
#   contour init --with-skill   # Install .agent/skills/contour-research/SKILL.md
```

Because `contour` automatically discovers `contour.yaml` in the current directory, any ancestor, or a `contour/` (`.contour/`, `research/`) subdirectory, you can run `contour` commands and `from contour import Run, Analysis, Eval` from anywhere inside your repo.

---

## Quickstart: Day-to-Day Research Loop

### 1. Frame a Goal and a Falsifiable Claim
```bash
contour init --with-skill
contour new goal token_mixing --title "Sub-quadratic & Low-Rank Token Mixing"
contour new claim qk_norm_warmup --title "QK-RMSNorm allows 2x peak LR and 50% shorter warmup in ViT"
```

### 2. Log Training Runs & Ad-Hoc Analyses
Configure your benchmark metric vocabulary and thresholds (`min_delta` absolute and/or `min_rel_delta` relative fraction) in `contour.yaml`:

```yaml
contour: 1
ignored_config_keys: [seed, num_workers, log_interval, eval_interval, checkpoint_dir]
metrics:
  imagenet1k_val/top1:     {direction: max, kind: quality, primary: true, min_delta: 0.20}
  val/loss:                {direction: min, kind: quality, min_delta: 0.005}
  sys/throughput_imgs_sec: {direction: max, kind: cost, min_rel_delta: 0.05}
```

Log training runs with `Run` (writes lean `runs/run_*.md` + tracked `runs/configs/<id>.yaml`):

```python
from contour import Run

with Run(
    title="ViT-S/16 QK-RMSNorm + 2x LR",
    parent=base_run.id,
    motivated_by=["clm_qk_norm_warmup"],
    config={"optim": {"lr": 0.002, "warmup_epochs": 5}, "model": {"qk_norm": True}},
) as treat_run:
    treat_run.track(80.3, name="top1", context={"subset": "imagenet1k_val"})
    treat_run.add_artifact("checkpoints/vit_s_qknorm.pt", kind="checkpoint")
```

Log post-hoc evaluations (`Eval`) or ad-hoc diagnostic probes (`Analysis`) without mutating training runs:

```python
from contour import Analysis

with Analysis(
    title="ADE20K object-size failure-mode breakdown",
    subjects=[seg_base.id, seg_lowrank.id],
    motivated_by=["clm_low_rank_recall"],
    script=__file__,
) as ana:
    ana.track(-6.4, name="small_obj_miou_delta", context={"subset": "diag"})
    ana.add_artifact("analyses/artifacts/size_breakdown.png", kind="plot")
```

### 3. Brief, Synthesize, Audit & Share
```bash
contour brief
contour ls runs --tree
contour ls analyses
contour link <run_or_ana_id> clm_qk_norm_warmup --as supports
contour check --strict
contour verify clm_qk_norm_warmup --agent --note "Clean control; scoped to ImageNet-1k classification."
contour report --goal goal_token_mixing --out /tmp/team_update
```

---

## CLI Reference (`contour`)

```bash
# Setup & transitions
contour init [path] [--with-skill]
contour new <goal|claim|run|analysis> <slug> --title "..." [--kind eval|analysis]
contour link <run_or_ana_id> <claim_id> [--as supports|contradicts|qualifies]
contour verify <claim_id> (--agent | --human) [--note "..."]

# Observables
contour brief [--goal G]
contour ls [runs|analyses] [--tree] [--goal G] [--claim C] [--eval E] [--kind K] [--subject R] [--status S] [--control L] [--verdict V]
contour show <node_id>
contour compare <ctrl_id> --vs <treat_id>
contour frontier [--metric M]
contour check [--strict]
contour ideate [--goal G]
contour report [--goal G] [--claim C ...] [--runs R ...] [--analyses A ...] [--out DIR] [--strict]
```
*(Pass `--json` to any command for structured output.)*
