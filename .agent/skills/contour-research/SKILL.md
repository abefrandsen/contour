---
name: contour-research
description: Navigate, brief, verify, analyze, report, and ideate over the Contour ML research state (Goals, Claims, Runs, and Analyses). Use whenever briefing on open threads, logging/inspecting ML runs, running ad-hoc failure-mode analyses, auditing claims, or sharing evidence-backed results.
---

# Contour: Epistemic ML Research Copilot Skill

**Contour** models an empirical ML research project as a versioned **Research State** $S$ in plain Markdown + Git, governed by **Invariants** ($\mathcal{I}$) and **Operators** ($T, \mathcal{O}$) (see `docs/architecture.md`):
- **Goals (`goals/goal_*.md`):** Minimal YAML (`title`, `status`) + `## Objective & Success Criteria`, `## Current Synthesis`, and `## Open Questions` bullets.
- **Claims (`claims/clm_*.md`):** Separate belief (`status`: `active | supported | falsified | archived`) and rigor (`verification`: `draft | agent_audited | human_verified`), linked to evidence via `supports`, `contradicts`, and `qualifies` (`qualifies` carries no weight in verification), with holds/breaks/caveats in `## Evidence Synthesis` (covered by `verified_hash`).
- **Runs (`runs/run_*.md` + `runs/configs/*.yaml`):** Append-only training records (`from contour import Run`) with tracked sidecar configs, `parent`, `compute`, source-only `code` provenance (excludes the Contour dir), `artifacts`, `metrics`, and dynamically computed diffs.
- **Analyses (`analyses/ana_*.md`):** Post-hoc evals (`kind: eval`) and ad-hoc diagnostic probes (`kind: analysis`) applied to existing `subjects` (`from contour import Analysis, Eval`), with scripts and outputs stored in gitignored `analyses/scripts/` and `analyses/artifacts/` by default (tracked via `script`, `script_sha256`, `script_tracked`, and `launch_cmd`). Untracked scripts default to `exploratory` (fine in `qualifies`); promote/track a script in Git for it to count as `clean` in `supports`.

Node **type** and **ID** are derived from directory + filename stem (no `id:`/`type:` needed in YAML). Flawed evidence is quarantined via `bugged: true`, never deleted.

> **Root Discovery:** `contour` locates `contour.yaml` in the current directory, any ancestor, or a `contour/` (`.contour/`, `research/`) subdirectory.

---

## Core Rule: Progressive Context Loading

**Never glob-read all `.md` files at once.** Survey compactly with `--json` or tables first, then `show` only the 2–5 relevant nodes:

```bash
contour brief --json
contour ls --json
contour ls runs --tree
contour check --json
contour show <node_id>
```

---

## Workflow 1: Morning Startup Briefing

Run at the start of a session or whenever asked *"what's open?"*:

1. Run `contour brief --json` (optionally `--goal <goal_id>`).
2. Present a concise, skeptical digest covering:
   - **Live / unsynthesized work:** `running` jobs (with log artifact paths), `stuck-job`, `unsynthesized-evidence`, and `todo-placeholder`.
   - **Human review queue:** Claims awaiting `--human` sign-off, `stale-verification` claims, and dirty frontier runs.
   - **Goals, Claims & Open Questions:** Compact status tree, `## Open Questions` bullets, and highest-leverage `ideate` moves.
3. Lead honestly with confounds, regressions, and open contradictions rather than smoothing over noise.

---

## Workflow 2: Synthesize Completed Runs into Claims

Run whenever a training run finishes (`from contour import Run` or `contour new run`):

1. **Inspect the run & transparent diff:**
   - Run `contour show <run_id>` (or `contour compare <ctrl_id> --vs <treat_id>`).
   - Always surface raw metric deltas (including secondary regressions), `control_level` (`clean` = valid & reproducible | `exploratory` = dirty/untracked | `invalid` = quarantined/unfinished), `verdict`, `config_diff`/`config_blocks`, `code_diff`, `compute_diff`, and advisory `flags` (`code_changed`, `eval_drift:*`, `multi_block_config_diff`, etc.) before proposing any Claim update.
   - Write a concrete summary in the run's `## Key Takeaway (TL;DR)` (replacing `- TODO:`).
2. **Link & synthesize onto the Claim:**
   - Evidence-relation semantics:
     - **`supports`:** Clean evidence directly confirming the claim's prediction (including causal-necessity ablations or neutral runs for invariance claims).
     - **`contradicts`:** Clean evidence refuting the claim as currently stated.
     - **`qualifies`:** Informational context bounding, scoping, or explaining the claim (failure boundaries, tradeoffs, exploratory/dirty runs, ad-hoc diagnostic analyses). Carries no weight in verification.
   - Run `contour link <run_id> <claim_id> [--as supports|contradicts|qualifies]`.
   - Update the Claim's `## Evidence Synthesis` (`- **Holds:**`, `- **Breaks:**`, `- **Untested / Caveats:**`) citing `[[run_id]]`.
3. **Verify invariants (`contour check --json`):**
   - Fix hygiene items (`schema-error`, `tainted-evidence`, `unsynthesized-evidence`, `todo-placeholder`, `undeclared-metric`) immediately.
   - For `overclaimed-status`: never leave `status: supported` with non-clean `supports` or unresolved `contradicts` (downgrade to `active`/`falsified`, or narrow the Claim and move boundary evidence to `qualifies`).
   - To quarantine a flawed run: set `bugged: true` (do not delete), unlink it from Claims, and mark tainted downstream Analyses `bugged: true` (or re-point valid children's `parent:`).

---

## Workflow 3: Ad-Hoc Failure-Mode & Diagnostic Analysis

Run when investigating *why* a model underperformed, probing checkpoints, checking failure modes, or running post-hoc evals:

1. **Orient & locate checkpoints:**
   - Run `contour compare <ctrl_id> --vs <treat_id> --json` and `contour show` on the subject runs to read checkpoint/log paths from `artifacts`.
2. **Explore in scratch, crystallize real findings:**
   - Iterate freely in scratch/REPL while hunting for the root cause—do not clutter `analyses/` with throwaway debug runs.
   - Once you have a **reproducible finding or a valuable ruled-out hypothesis**, save a repeatable script in `analyses/scripts/` (gitignored by default; logged as `exploratory`)—or commit it in a tracked scripts folder if it needs to be `clean` load-bearing evidence in `supports`—wrapped in `from contour import Analysis` (or `Eval` for standardized benchmarks):
     - Pass `subjects=[...]`, `motivated_by=[...]`, and `script=__file__` (auto-captures `script_sha256`, `script_tracked`, and `launch_cmd`).
     - Save plots/CSV/JSON outputs under `analyses/artifacts/<ana_id>/` via `ana.add_artifact(...)`.
     - Ad-hoc `kind: analysis` diagnostic scalars do not require `contour.yaml` registration; `kind: eval` benchmarks do.
3. **Fold insight back into the research graph:**
   - Replace `- TODO:` in `analyses/ana_*.md` with a concise TL;DR and findings table.
   - Link via `contour link <ana_id> <claim_id> --as qualifies|supports|contradicts` (analyses default to `qualifies` unless `--as` is explicit).
   - Update the Claim's `## Evidence Synthesis` citing `[[ana_id]]`, or open a follow-up Claim motivated by `[[ana_id]]`.
   - Run `contour check --json`.

---

## Workflow 4: Adversarial Claim Audit (*The Critic*, Tier-3 Invariants)

Run after updating a Claim or whenever auditing the research state (`contour verify <claim_id> --agent`):

1. **Pass mechanical gates first:** Ensure `contour check --json` has zero Tier-1 `ERROR`s (`schema-error`, `tainted-evidence`) and zero Tier-2 epistemic issues (`overclaimed-status`) on `<claim_id>` and its linked evidence.
2. **Critique skeptically across 4 dimensions:**
   - **Diff & causal fairness:** Inspect `config_diff`/`config_blocks`, `code_diff`, `compute_diff`, `launch_cmd`, and advisory `flags` (`code_changed`, `eval_drift:*`, `multi_block_config_diff`). Distinguish fair structural swaps (e.g., changing encoders) from accidental confounds (e.g., unintended dataset or hyperparameter drift in `config`).
   - **Direction & polarity alignment:** Because `overclaimed-status` checks cleanliness rather than metric polarity, verify every `supports`/`contradicts`/`qualifies` link matches what the Claim predicts (including standalone `[no parent]` runs and causal-necessity ablations).
   - **Honest scope & caveats:** Verify the Claim's `title` and `## Evidence Synthesis` (`Holds` / `Breaks` / `Untested / Caveats`) stay within tested conditions and note secondary metric regressions.
   - **Alternative explanations & append-only discipline:** Challenge gains against simpler explanations (seed noise, implicit regularization, extra compute/steps).
3. **Lock the audit:**
   - Revise `title`, synthesis, links, or `status` honestly if flaws appear.
   - Run `contour verify <claim_id> --agent --note "<concise audit summary>"` to content-hash frontmatter, body prose, and evidence into `verified_hash` and queue the Claim in `contour brief` for selective human sign-off (`--human`).

---

## Workflow 5: State-Conditioned Experiment Ideation

Run when deciding what experiments to run next:

1. Run `contour ideate --json` and `contour frontier --json` (optionally `--goal <goal_id>`).
2. Review the 6 deterministic moves:
   - **Move 1:** Confirm positive `exploratory` (dirty) or flagged runs.
   - **Move 2:** Probe failure boundaries (`contradicts` / `qualifies` / `tradeoff`).
   - **Move 3:** Combine clean sibling wins/marginals on the same `parent` with disjoint `config_diff` keys (including compounding marginals crossing $\tau$).
   - **Move 4:** Ablate multi-unit winning `config_diff`s.
   - **Move 5:** Address Goal `## Open Questions` and unsupported active claims.
   - **Move 6:** Diagnose top frontier runs using observations, diagnostic metrics, and attached `analyses/`.
3. Rank concrete experiment proposals by expected information gain per GPU-hour, specifying `parent`, config/code change, invariance smoke test if editing code, and falsification criterion.

---

## Workflow 6: Sharing Evidence-Grounded Results (`contour report`)

Run whenever asked to write up results for colleagues (Slack update, research memo, PR summary, slides):

1. **Generate the verified truth bundle first:**
   ```bash
   contour report --goal <goal_id> --json           # or --claim <clm_...>
   contour report --goal <goal_id> --out <dir>      # portable report.md + copied local plots/tables
   contour report --runs <r1> <r2> --json           # preliminary comparison (no Claim yet)
   ```
   Pass `--strict` for decision-critical updates to fail loudly if any load-bearing item (`supports` / `contradicts`) is `draft`, `stale-verification`, dirty, exploratory, or preliminary.
2. **Draft strictly from the truth bundle:**
   - Preserve exact numbers, verification badges (`human_verified` / `agent_audited` / `draft` / `PRELIMINARY`), and all caveats from `## Evidence Synthesis`.
   - Never present `draft`, `exploratory`, or `PRELIMINARY` findings as settled.
