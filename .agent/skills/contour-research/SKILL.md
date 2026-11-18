---
name: contour-research
description: Navigate, verify, synthesize, and ideate over the Contour ML research state (Goals, Claims, and Runs). Use whenever logging/inspecting ML runs, auditing claims, checking for confounds, or generating next experiments.
---

# Contour: Epistemic ML Research Copilot Skill

**Contour** maintains an empirical ML research state in plain Markdown + Git across three node types:
- **Goals (`goals/goal_*.md`):** Top-level research objectives and target regimes.
- **Claims (`claims/clm_*.md`):** Falsifiable hypotheses/findings with separate belief (`status`) and rigor (`verification`) states, linked to runs via `supports`, `contradicts`, and `qualifies`.
- **Runs (`runs/run_*.md`):** Single canonical record per run (`from contour import Run` writes directly here) with machine-owned `config`, `compute`, `code`, `metrics`, and hash-checked `vs_parent`.

> **Root Discovery:** `contour` locates `contour.yaml` in the current directory, any ancestor, or a `contour/` (`.contour/`, `research/`) subdirectory.

---

## Core Rule: Progressive Context Loading

**Never glob-read all `.md` files at once.** Survey compactly with `--json` or tables first, then `show` only the 2–5 relevant nodes:

```bash
contour status --json
contour runs --tree
contour check --json
contour frontier --json
contour show <node_id>
```

---

## Workflow 1: Synthesize New Runs into Claims

Run this whenever a training run finishes (`from contour import Run` or `contour new run`):

1. **Inspect the run & fresh diff:**
   - Run `contour show <run_id>` (or `contour compare <ctrl_id> --vs <treat_id>`).
   - Check `control_level` (`clean | exploratory | confounded`), `verdict`, `unaccounted`, `unrealized`, and per-metric `signal`.
   - Write a concrete summary in the run's `## Key Takeaway (TL;DR)` (replacing `- TODO:`).
2. **Salvage confounds if needed (never discard useful runs silently):**
   - If a matching-facet baseline already exists, set `baseline: '[[run_...]]'` on the treatment run and run `contour check --fix`.
   - If a key in `unaccounted` is a benign engineering setting (e.g. `num_workers`), register it under `ignored_config_keys` in `contour.yaml`.
   - Otherwise, link it under `qualifies` and let `contour ideate` Move 1 propose the bridge control.
3. **Link & synthesize onto the Claim:**
   - `contour link <run_id> <claim_id>` (uses the advisory relation; pass `--as supports|contradicts|qualifies` when the claim predicts a cost reduction or failure boundary).
   - **Required synthesis prose:** Update the Claim's `## Evidence Synthesis` to explicitly cite `[[run_id]]` with its concrete delta and control level, and update `scope` (`holds_in`, `breaks_in`, `untested`).
4. **Run the mechanical gate:**
   - Run `contour check --fix` then `contour check --json`.
   - **Hygiene issues (`C001`–`C006`, `C009`):** Fix immediately in the same turn (missing citations, `- TODO:` placeholders, unlinked motivating runs, undeclared metrics).
   - **Epistemic issues (`C007` over-claimed status):** Never fabricate evidence. Move non-qualifying runs from `supports` to `qualifies`, calibrate `status` back to `active` if needed, and propose a bridge baseline or Move 1 control.

---

## Workflow 2: Adversarial Claim Audit (*The Critic*)

Run this after updating a Claim or whenever auditing the research state (`contour verify <claim_id> --agent`).

1. **Pass the mechanical gate first:** Ensure `contour check --json` has zero ERRORs and zero epistemic issues (`C007`) on `<claim_id>` and its `supports` runs.
2. **Critique the Claim skeptically across 4 dimensions:**
   - **Config & code integrity:** Do `config_diff` and `interventions` genuinely reflect the claimed mechanism, with no hyperparameter tuning asymmetry or uncommitted `code.dirty` changes?
   - **Scope calibration:** Do the Claim's `title` and `scope.holds_in` stay strictly within the tested `(task, eval, arch_scale)` without over-generalizing?
   - **Direction & failure boundaries:** Does each linked run's relation (`supports` / `contradicts` / `qualifies`) match what the Claim actually predicts? Are all secondary metric regressions and failure boundaries reflected in `scope.breaks_in`?
   - **Alternative explanations:** Could a simpler confound (implicit regularization on small datasets, compute/step difference, data shift) explain the gain?
3. **Record the audit verdict:**
   - If flaws are found: revise `title`, `scope`, or `status` honestly first.
   - Once solid, run:
     ```bash
     contour verify <claim_id> --agent --note "<concise audit summary>"
     ```
   - This promotes `verification` to `agent_audited` with a `verified_hash` lock, queuing it in `contour review` for selective human sign-off (`--human`).

---

## Workflow 3: State-Conditioned Experiment Ideation

Run this when deciding what experiments to run next:

1. Run `contour ideate --json` and `contour frontier --json` (optionally filtered by `--goal` or `--task`).
2. Review the 6 deterministic moves:
   - **Move 1 (Isolate confounds):** Salvage or de-confound positive `confounded` runs.
   - **Move 2 (Resolve tensions):** Probe failure boundaries (`contradicts` / `qualifies` / `tradeoff`).
   - **Move 3 (Combine wins):** Pair `clean` siblings (`win`, `efficiency_win`, or compounding `marginal` wins whose projected $\Delta_1 + \Delta_2 \ge \tau$).
   - **Move 4 (Strip complexity):** Leave-one-out `probe`/`proxy` ablations on multi-intervention wins.
   - **Move 5 (Expand regimes):** Cover untested `(task, arch_scale)` cells from `goal.target_regimes` and `scope.untested`.
   - **Move 6 (Diagnose frontier):** Form abductive hypotheses from top frontier runs' observations and diagnostic metrics.
3. **Prescribe concrete, ranked experiments:** Rank proposals by **expected information gain per GPU-hour** (using `estimated_gpu_hours`). For each recommended experiment, specify:
   - **Baseline run** (`parent_runs` / `baseline`)
   - **Exact intervention / config diff** (following `key=value` grammar)
   - **Tier** (`probe` for cheap falsification, `proxy`/`full` for confirmation)
   - **Falsification criterion** (what metric outcome supports vs. contradicts the Claim).
