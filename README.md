# Contour — Survey and Map Your ML Research Landscape

![Contour Banner](assets/banner.jpg)

**Contour** maintains an easily navigable research state built on top of many training runs, designed for both human researchers and AI agents using plain Markdown and Git.

Install Contour once and use it as a plug-in research layer across any number of independent ML repositories:

1. **Empirical Ground Truth (`fieldlogs/log_*.md` & `bundles/*.json`):** Any training platform (or `from contour import Run`) emits a portable `contour.field_bundle/v1` JSON bundle. `contour sync` validates it, archives the bundle, distills a Markdown card in `fieldlogs/`, and **automatically computes `vs_parent`** (metric deltas, `single_logical_diff` confound check, and optional `noise.yaml` noise-floor SNR).
2. **Research Claims & Campaigns (`cairns/crn_*.md` & `expeditions/expd_*.md`):** Plain Markdown files tracking overarching research campaigns (`expeditions/`) and falsifiable claims (`cairns/`) linked directly to fieldlogs via `supports`, `contradicts`, and `qualifies`.
3. **Navigation & Experiment Generation (`contour` CLI):** Instant graph retrieval, on-the-fly multi-run comparisons (`contour compare`), Pareto frontiers (`contour ridge`), and **6 Scientific Move detectors (`contour bearings`)** that mine your runs to propose the highest-leverage next experiments.

---

## Installation (Use Across Any Research Repo)

Install Contour in editable mode (or link `contour` onto `$PATH` and Python's user site-packages):

```bash
pip install -e /path/to/contour
```

Then, inside **any ML project repository**:

```bash
cd ~/projects/my-ml-repo

# Creates a self-contained ./contour/ directory (contour.yaml, expeditions/, cairns/, fieldlogs/, bundles/)
contour init

# Optional flags:
#   contour init .                  # Initialize at the repo root instead of ./contour/
#   contour init --with-skill       # Also write .agent/skills/contour-research/SKILL.md into the repo
#   contour init --eject-logger     # Write a standalone stdlib-only contour_run.py for remote cluster jobs
```

Because `contour` automatically discovers `contour.yaml` in the current directory, any parent directory, or a `contour/` (`.contour/`, `research/`) subdirectory, you can run `contour` commands and `from contour import Run` from anywhere inside `~/projects/my-ml-repo/`.

---

## Quickstart Tutorial: A Day-to-Day Research Loop

### 1. Initialize & Frame the Research Question
Start by initializing Contour in your project repo, then create an **Expedition** (your overarching goal) and a **Cairn** (a falsifiable hypothesis you want to test):

```bash
contour init
contour new expedition token_mixing \
  --title "Sub-quadratic & Low-Rank Token Mixing"
contour new cairn qk_norm_warmup \
  --title "QK-RMSNorm allows 2x higher peak learning rate and 50% shorter warmup in ViT"
```

Open `contour/cairns/crn_qk_norm_warmup.md` in any editor to jot down your intuition and link it to `[[expd_token_mixing]]`.

### 2. Run Experiments & Sync Ground Truth
Wrap your training script with `contour.Run` (or emit a [Field Bundle JSON](docs/field_bundle.md) from any job runner):

```python
from contour import Run

# 1. Baseline run
with Run(
    exp_name="vit_s_baseline",
    task="image_classification",
    dataset_train=["imagenet1k_train"],
    dataset_eval=["imagenet1k_val"],
    arch_family="vit",
    arch_scale="22M",
    motivated_by=["expd_token_mixing"],
) as base_run:
    base_run.track(79.7, name="top1", context={"subset": "imagenet1k_val"})

# 2. Treatment run branching from the baseline
with Run(
    exp_name="vit_s_qknorm_2xlr",
    task="image_classification",
    dataset_train=["imagenet1k_train"],
    dataset_eval=["imagenet1k_val"],
    arch_family="vit",
    arch_scale="22M",
    interventions=["qk_rmsnorm", "lr_peak=2x", "warmup=0.5x"],
    parent_runs=[base_run.run_id],
    motivated_by=["crn_qk_norm_warmup"],
) as treat_run:
    treat_run.track(80.3, name="top1", context={"subset": "imagenet1k_val"})
```

Ingest finished runs into `fieldlogs/*.md`:

```bash
contour sync
```

`sync` validates and archives the raw JSON under `bundles/`, creates the Markdown cards in `fieldlogs/`, flags `crn_qk_norm_warmup` with `needs_synthesis: true`, and **automatically computes `vs_parent`** on the treatment run.

### 3. Inspect Diffs, Check Confounds & Synthesize
Check what needs attention and inspect the new run:

```bash
contour drift
contour inspect <treatment_log_id>
```

The `inspect` view shows the auto-computed `vs_parent` block—confirming whether only the declared `interventions` changed (`single_logical_diff: true`) or whether a confounding facet (like `dataset_train` or `arch_scale`) also shifted. You can also compare arbitrary sets of runs (including multi-seed arms) on the fly:

```bash
contour compare <control_log_id> --vs <treatment_log_id>
```

Once you've reviewed the result, link the run to the Cairn and update the Cairn's notes on where the claim holds vs. breaks:

```bash
contour link <treatment_log_id> crn_qk_norm_warmup --relation supports
```

*(Optional)* If you have 2+ baseline seed repeats, register the regime's seed-to-seed noise floor once so single-seed deltas are automatically scored by signal-to-noise ratio (`SNR >= 2` $\to$ `Grade A`):

```bash
contour noise seed image_classification/imagenet1k_val/22M --from <seed0_id> <seed1_id> <seed2_id>
```

### 4. Generate the Next Wave of Experiments
As runs accumulate, use `ridge` and `bearings` (or hand the repo to an AI agent with `--json`) to decide what to run next:

```bash
# View the quality vs. compute Pareto frontier
contour ridge --eval imagenet1k_val

# Run the 6 Scientific Move detectors to propose next experiments
contour bearings
```

`contour bearings` automatically scans your graph for:
- **Move 1 (Confound Isolation):** Multi-diff runs that need a clean single-diff control run.
- **Move 2 (Dialectics):** Cairns with contradictions or boundary qualifications (`qualifies`).
- **Move 3 (Synthesis):** Sibling Pareto-improving interventions (`c1.interventions | c2.interventions`) that haven't been combined yet.
- **Move 4 (Occam's Razor):** Bundled multi-tag runs (like `["qk_rmsnorm", "lr_peak=2x", "warmup=0.5x"]`) that need leave-one-out ablations to test which tags are actually necessary.
- **Move 5 (Deduction):** Untested tasks, eval benchmarks, or model scales for supported cairns.
- **Move 6 (Abduction):** Bottleneck and anomaly notes from the top Pareto-frontier runs.

---

## CLI Reference (`contour`)

```bash
contour init [path] [--with-skill] [--eject-logger]       # Initialize Contour (defaults to ./contour/)
contour sync [--from-json <paths...>] [--force]           # Ingest Field Bundles into fieldlogs/
contour map                                               # Overview of Expeditions -> Cairns -> FieldLogs
contour inspect <node_id>                                 # Zoom in on an Expedition, Cairn, or FieldLog
contour compare <ctrl_ids...> --vs <treat_ids...>         # On-the-fly control vs. treatment comparison
contour link <log_id> <cairn_id> --relation <rel>         # Link a FieldLog under supports/contradicts/qualifies
contour ridge [--eval <ds>] [--task <t>] [--metric <m>]   # Plot the quality vs. GPU-hour Pareto frontier
contour bearings [--expedition <expd_id>]                 # Propose next experiments across the 6 Scientific Moves
contour drift [--strict]                                  # Check graph hygiene and confounded comparisons
contour noise list                                        # View registered noise floors (noise.yaml)
contour noise seed <regime> --from <log_ids...>           # Estimate noise floor from baseline seed repeats
contour new <expedition|cairn|fieldlog> <slug> --title "..."
```

Pass `--json` to any command for structured output when working with AI agents or scripts.

---

## Core Concepts

| Concept | Meaning | Lives in |
|---|---|---|
| **Expedition** | A research campaign with a goal (*North Star*) and open questions. | `expeditions/expd_*.md` |
| **Cairn** | A falsifiable claim or hypothesis (`active`, `supported`, `falsified`) linked to `fieldlogs/` via `supports`, `contradicts`, and `qualifies`. | `cairns/crn_*.md` |
| **FieldLog** | The machine-written record of one run (`status`, `tier: probe\|proxy\|full`, `metrics`, `interventions`, `vs_parent`), paired with human/agent notes in the Markdown body. | `fieldlogs/log_*.md` |
| **Field Bundle** | The portable JSON contract (`contour.field_bundle/v1`) emitted by `contour.Run` or any training platform. | `bundles/*.json` |
| **`vs_parent` & Evidence Grade (`A`/`B`/`C`)** | Auto-computed comparison on every child FieldLog (and via `contour compare`). **A** = single logical diff (`single_logical_diff: true`), iso-compute, and clears noise floor (`SNR >= 2`). **B** = controlled single diff without a noise-floor check or within noise. **C** = confounded (multiple facets changed, e.g. dataset + architecture), `probe` tier, or non-usable. | `fieldlogs/log_*.md` frontmatter |
| **Noise Floor (`noise.yaml`)** | Optional registry storing seed-to-seed standard deviation per `(task/eval/scale)` regime so single-seed runs can be judged against baseline variance (`clears_noise`). | `noise.yaml` |
