# Contour Evolution — Action Plan

> **Living document** capturing decided design changes and open brainstorm threads as we evolve Contour.

## North-Star Goals

1. **Principled, minimal structure** for human-led, AI-agent-assisted ML research.
2. **Clean, reproducible evidence base** covering both model training runs and downstream evaluations/analyses produced by reproducible scripts.
3. **Effortless navigation** for both humans and AI agents over what has been learned, why things broke, and what open questions remain.

---

## Decided Workstreams

### Workstream 1: Expanded Evidence Layer (`runs/` + `analyses/`)

Split empirical evidence—all framed as *outputs produced by reproducible scripts/code*—into two top-level evidence folders:

- **`runs/run_<YYYYMMDD>_<8hex>.md`** — Model training runs (`train`), retaining lineage, structured diffs against baseline (computed dynamically), and compute tracking.
- **`analyses/ana_<YYYYMMDD>_<8hex>.md`** — Post-hoc evaluations (`kind: eval`) and ad-hoc diagnostic/mechanistic/comparative analyses (`kind: analysis`) applied to existing runs, checkpoints, or datasets.

#### 1.1 Directory & Git Conventions
- `contour init` creates:
  - `analyses/` (tracked in Git — holds `ana_*.md` records)
  - `analyses/scripts/` (**gitignored by default** — dedicated home for ad-hoc analysis scripts)
  - `analyses/artifacts/` (**gitignored by default** — plots, CSV/JSON tables, raw analysis outputs)
- Seed/update `.gitignore` with `analyses/scripts/` and `analyses/artifacts/` so agent-written ad-hoc scripts and plots do **not** dirty `git status` (`code.dirty` continues to accurately reflect core ML repo cleanliness).
- Reusable or permanent scripts can always be promoted/committed into the main repo later.

#### 1.2 Schema Additions & Node Structure
- **Addition to `runs/`:**
  - Add lightweight `artifacts:` field (e.g., checkpoint paths/URIs, output directory, external links) so downstream evals and agent-led analyses can locate checkpoints without reverse-engineering `launch_cmd`.
- **New `analyses/` node (`ana_<YYYYMMDD>_<8hex>.md`):**
  - **Shared provenance envelope:** `id`, `type: analysis`, `kind: eval | analysis`, `title`, `status` (`running | final | failed`), `validity` (`ok | bugged | superseded`), `created_at`, `finished_at`, `motivated_by`, `code` (`commit`, `dirty`, `branch`), `launch_cmd`, `config`, optional `compute`.
  - **Script provenance:** `script` (path to script) and `script_sha256` (content hash at execution time, ensuring local reproducibility even when scripts live in gitignored `analyses/scripts/`).
  - **Subjects & outputs:**
    - `subjects`: list of run wikilinks (`['[[run_...]]', ...]`, or `[]` for dataset-only analyses).
    - `checkpoint`: optional checkpoint identifier/mapping (defaults to final checkpoint).
    - `metrics`: optional scalar dict.
    - `artifacts`: optional list of output pointers (`path`/`uri`, `kind`, `note`).
  - **Body:** `## Key Takeaway (TL;DR)` and `## Findings & Observations` (concise Markdown tables, interpretations, mechanistic hypotheses).
- **Python API (`src/contour/`):**
  - Add `from contour import Analysis` (and optional thin `Eval` wrapper with `kind="eval"`), mirroring `Run`'s rank-0 context-manager API (`ana.track(...)`, `ana.add_artifact(...)`, auto-capturing git info, `script`, `script_sha256`, `launch_cmd`, and preserving author-owned keys/body on re-write).
  - Add `run.add_artifact(...)` to `Run`.

#### 1.3 Epistemic & Mechanical Verification (`check.py`)
- **Analysis cleanliness (`control_level`):**
  - `invalid`: `status != final`, `validity != ok`, or **any `subject` run is missing or not `final/ok`** (automatic downstream taint propagation when an upstream run is later marked `bugged`).
  - `exploratory`: `code.dirty == true` or missing `code.commit`.
  - `clean`: `final/ok`, all `subjects` are `final/ok`, and clean repo commit (`code.dirty == false`).
- **Claim linking & `C007`:**
  - Claims can link `[[ana_...]]` under `supports`, `contradicts`, and `qualifies` identically to `[[run_...]]`.
  - A `clean` analysis satisfies `C007` in `supports` and `contradicts`.
- **Staleness & taint propagation (`verified_hash` / `C008`):**
  - `compute_verified_hash` hashes linked analyses' provenance/outputs **and** their `subjects`' snapshots—so mutating or invalidating an upstream training run automatically invalidates dependent claims' verification.
- **Metric hygiene (`C009`):**
  - Enforced strictly on `runs/` and `analyses/` with `kind: eval`.
  - Relaxed for `kind: analysis` so ad-hoc diagnostic scalars do not require upfront registration in `contour.yaml` (if a key *is* declared in `contour.yaml`, its spec still applies).
- **Artifact hygiene:**
  - Missing local relative artifact paths emit a non-blocking hygiene **`WARN`** (not an `ERROR`, so gitignored local outputs on another machine never break `contour verify` or CI); remote URIs (`s3://`, `gs://`, `https://`, etc.) are passed through unchecked. (By contrast, tracked `config_path` files missing on disk remain a blocking `C001` `ERROR`.)
- **Unlinked motivation (`C003`) & TODOs (`C004`):**
  - Apply consistently to finalized `analyses/` motivated by claims.

#### 1.4 Dynamic Graph & CLI Composition (Zero Mutation of `runs/`)
- **`ContourGraph` (`graph.py`):**
  - Load `analyses/*.md` into `g.analyses` and `g.all_nodes`; index attached evals/analyses per run (`g.run_analyses(run_id)`).
- **CLI (`cli.py`):**
  - `contour new analysis <slug> --title "..." [--kind eval|analysis]`
  - `contour analyses [--kind eval|analysis] [--subject R] [--claim C] [--goal G] [--control L]`
  - `contour link <run_or_ana_id> <claim_id> [--as supports|contradicts|qualifies]`
  - `contour show <run_id>` displays attached `eval` and `analysis` records; `contour show <ana_id>` displays full analysis provenance, subjects, metrics, and artifacts.
  - `contour compare <run_A> --vs <run_B>` composes inline run metrics plus shared `kind: eval` benchmarks, flagging an eval confound if `config` or `code.commit` drifted between the two evals.
  - `contour frontier --metric <eval_metric>` joins a run's training `gpu_hours` with its latest clean `kind: eval` score.
  - `contour status` includes analysis counts.

#### 1.5 Agent Skill & Workflow (`SKILL.md`)
- Add **Workflow: Ad-Hoc Failure-Mode & Diagnostic Analysis**:
  1. Orient on target runs via `contour compare` / `contour show` and read checkpoint paths from `artifacts`.
  2. Explore freely in scratch/REPL while hunting for signal.
  3. Crystallize reproducible findings—and meaningful ruled-out hypotheses—into a script under `analyses/scripts/` using `Analysis(...)`, saving plots/tables under `analyses/artifacts/<ana_id>/`.
  4. Synthesize back into the graph (`contour link`, update Claim `## Evidence Synthesis` and `scope`, optionally propose a follow-up Claim) and run `contour check`.

#### 1.6 Implementation Checklist (Workstream 1)
- [x] `src/contour/graph.py`: add `analyses` node directory, `ContourNode` analysis helpers, `usable_analyses()`, and subject→analysis reverse indexing.
- [x] `src/contour/run_logger.py`: add `artifacts` support to `Run`; implement `Analysis` (and `Eval`) logger with `script` + `script_sha256` capture and `ingest_analysis_record`.
- [x] `src/contour/__init__.py`: export `Analysis` (and `Eval`).
- [x] `src/contour/compare.py`: add analysis cleanliness helper (`compute_analysis_control_level`) and compose shared `kind: eval` metrics + eval-drift detection in `compare_runs`.
- [x] `src/contour/check.py`: extend `C001`–`C009`, `C007` (allow clean analysis in `supports`/`contradicts`), `compute_verified_hash` (include analyses + subjects), artifact existence check, and `compute_frontier` (join `kind: eval` metrics).
- [x] `src/contour/cli.py`: update `init` (dirs + `.gitignore`), `new`, `link`, `show`, `status`, `compare`, `frontier`, and add `analyses` command.
- [x] `.agent/skills/contour-research/SKILL.md` & `skills/contour-research/SKILL.md`: document `analyses/` and the ad-hoc analysis workflow.
- [x] `tests/`: add unit & integration tests covering `Analysis` logging, taint propagation, `C007`/`C008`/`C009`, `compare`/`frontier` eval composition, and CLI flows.

---

### Workstream 2: Daily Research Workflows (`contour brief` & `contour report`)

Support the bookend human-plus-agent workflows—starting the day oriented on live threads and ending by sharing evidence-grounded results with colleagues—without adding new node types.

#### 2.1 Morning Startup Briefing (`contour brief [--goal G] [--json]`)
- **Single startup command** assembling a high-signal snapshot of live work:
  1. **Active & unsynthesized work:** currently `running` runs/analyses (with `artifacts` log pointers), stuck runs (`C005`), and newly finalized runs/analyses not yet linked or missing TL;DRs (`C003`/`C004`).
  2. **Human review inbox:** claims awaiting `--human` sign-off, stale verified claims (`C008`), and dirty-code frontier runs.
  3. **Open threads & questions:** explicit `open_questions` from active Goals, active claims with tensions (`contradicts` / `qualifies`), `scope.untested`, and top `ideate` moves ranked by expected info gain per GPU-hour.
  4. **Health summary:** mechanical `contour check` issue count & error status.
- **Lightweight `open_questions:` on `Goal`:** add an optional `open_questions:` list in `goal_*.md` frontmatter (keeps the ontology minimal while giving human-posed questions a deterministic home surfaced by `contour brief`).
- **Live-run ergonomics:**
  - Support `--status <running|final|failed>` filtering in `contour runs` (and `contour analyses`).
  - Record initial `artifacts` (e.g., log directory / TB / W&B link) at `Run` startup so agents inspecting ongoing runs know immediately where live logs live.

#### 2.2 Evidence-Grounded Colleague Write-Ups (`contour report`)
- **`contour report [--goal G] [--claim C ...] [--runs R ...] [--analyses A ...] [--out <dir>] [--strict] [--json]`:**
  - Produces an **ephemeral, verified truth bundle** (never a tracked node type, avoiding stale-report rot) that an AI agent can tailor into a Slack update, research memo, PR summary, or slide bullets without provenance laundering or over-claiming.
  - **Packet contents:**
    - **Integrity Banner** at the top summarizing verification coverage (`human_verified` / `agent_audited` / `draft`), staleness (`C008`), dirty/probe runs, and any open `check` warnings—plus `--strict` mode to fail loudly when sharing decision-critical results if anything cited is unverified, stale, or dirty.
    - For Goals/Claims: each Claim's exact `status`, `effective_verification`, `audit_note`, verified baseline-vs-treatment deltas, `config_diff`/`data_diff`, `compute_diff`, linked `analyses/` summaries & artifacts, and mandatory **`scope.holds_in` / `scope.breaks_in` / `scope.untested`** boundaries.
    - Lightweight **`--runs` / `--analyses` fallback** for quick early comparisons before a Claim exists (automatically stamped `[PRELIMINARY — NO VERIFIED CLAIM]`).
    - Compact **Provenance Appendix** (run/analysis IDs, git commits, `verified_hash`es, timestamp).
  - **Portable artifact export (`--out <dir>`):** writes `report.md` and copies any referenced local plots/tables from gitignored `analyses/artifacts/` into `<dir>/assets/` with rewritten relative links, producing a self-contained folder ready to share or upload.
- **Skill update (`SKILL.md`):**
  - Add **Workflow: Morning Briefing** (use `contour brief --json`, surface confounds/tensions honestly rather than smoothing over noise).
  - Add **Workflow: Sharing Results (`contour report`)** (always generate the grounded truth bundle first; preserve verification badges and `scope` boundaries in any colleague-facing write-up; never present `draft`, `exploratory`, or preliminary findings as settled).

#### 2.3 Implementation Checklist (Workstream 2)
- [x] `src/contour/cli.py` & `src/contour/check.py`: implement `contour brief` (aggregating active/unsynthesized runs & analyses, `review` inbox, `open_questions`, tensions, top `ideate` moves, and check summary).
- [x] `src/contour/cli.py`: add optional `open_questions: []` to `goal` template (`contour new goal`), add `--status` filter to `contour runs` / `contour analyses`.
- [x] `src/contour/report.py` & `src/contour/cli.py`: implement `contour report` generating grounded Markdown + `--json` truth bundles, Integrity Banner, `--strict` gating, `--runs`/`--analyses` preliminary mode, and `--out <dir>` portable artifact bundling.
- [x] `.agent/skills/contour-research/SKILL.md` & `skills/contour-research/SKILL.md`: add Morning Briefing and Colleague Write-Up workflows (plus explicit devil's-advocate reporting guidance for completed runs).
- [x] `tests/`: add tests for `contour brief`, `contour report` (Integrity Banner, `--strict`, preliminary `--runs`/`--analyses`, `--out` asset bundling), `open_questions`, and `--status` filtering.

---

### Workstream 3: Simplified Run Schema & Transparent Diff Model

Simplify run metadata and comparisons down to what a machine can compute reliably with zero busywork—surfacing diffs transparently so the Tier-2 Critic and human own causal judgment (handling both 1-knob ablations and wholesale architectural swaps like changing encoders).

#### 3.1 Simplified Run Schema
- **Drop brittle facets:** Remove mandatory `arch_family` and `arch_scale` from `runs/` (and drop hardcoded `scales` assumptions from `Goal.target_regimes`, CLI tables, and `ideate` Move 5). Architecture details live naturally in `config` (and `compute.params` if desired).
- **Retain universal, lightweight run fields:**
  - `task` (optional grouping string, e.g., for frontiers/tables)
  - `dataset_train` (list of split names—representing unions of splits—or optional weighted mapping; sampling differences also live naturally in `config`)
  - `tier` (`probe | proxy | full`)
  - `config` (arbitrary nested dict)
  - `compute` (recorded automatically for free: `gpu_hours`, `num_gpus`, optional `params`)
  - `code` (`commit`, `dirty`, `branch`), `launch_cmd`, `metrics`, `artifacts`
  - `parent_runs` / `baseline`
  - `interventions`: lightweight, human-readable tags (e.g., `["siglip_encoder"]`, `["qk_rmsnorm", "2x_lr"]`)—**no forced `key=value` re-typing of `config_diff`**.

#### 3.2 Transparent Structured Diff Operator (`compare.py`)
Replace strict `key=value` `unaccounted`/`unrealized` policing with five objective, automatically computed diffs:
1. **`data_diff`:** Added, removed, or reweighted splits in `dataset_train`.
2. **`config_diff`:** Dot-path flattened diff over nested `config` dicts (`optim.lr: 0.001 -> 0.002`), minus `ignored_config_keys`, with clean subtree grouping/summarization when an entire component block (e.g., `encoder.*`) changes.
3. **`code_diff`:** Commit change indicator (`same_commit` vs. `commit_a -> commit_b`) and `dirty` status.
4. **`compute_diff`:** `gpu_hours` and `params` comparisons—recorded and surfaced mechanically for free without silently mangling quality verdicts unless an explicit `cost` metric regresses (or strict `iso_tolerance` is opted into).
5. **`metric_deltas`:** Shared metrics with `delta` and `signal` (`clear | marginal | neutral | regression`) against `min_delta` ($\tau$).

#### 3.3 Hard Mechanical Checks + Advisory Flags (vs. Mechanical Confound Policing)
- **Mechanical `control_level` (hard facts only):**
  - `invalid`: either run has `status != final` or `validity != ok`.
  - `exploratory`: either run has `code.dirty == true` or `tier == probe`.
  - `clean`: both runs are `final/ok`, clean commits, non-probe.
- **Advisory `flags` (surfaced for Critic & human review, not hard blockers):**
  - e.g., `dataset_changed`, `task_changed`, `code_changed`, `compute_increased`, `multi_block_config_diff`.
  - Lets an encoder swap pass cleanly when fair, while immediately highlighting unexpected drift (e.g., `dataset_changed`) during synthesis and `verify --agent`.
- **Keep `inputs_hash` and `verified_hash` staleness locks** intact.
- **Keep `ideate.py` aligned:** Remove `arch_family`/`arch_scale` and `key=value` `unaccounted`/`unrealized` coupling; adapt Move 1 to surface positive runs that are `exploratory` (dirty/probe) or carry high-risk advisory flags (e.g., `dataset_changed`); work cleanly with lightweight `interventions` tags and honor `--goal` scoping across moves.

#### 3.4 Implementation Checklist (Workstream 3)
- [x] `src/contour/graph.py` & `src/contour/run_logger.py`: remove required `arch_family`/`arch_scale` facets; keep `task` optional; support list or weighted-dict `dataset_train`; preserve `compute` auto-recording.
- [x] `src/contour/compare.py`: implement dot-path nested `config_diff` (with subtree summary), `data_diff`, `code_diff`, `compute_diff`, advisory `flags`, and simplified `control_level` (`invalid | exploratory | clean`); keep `inputs_hash` & `min_delta` signals.
- [x] `src/contour/check.py` & `src/contour/ideate.py`: align `C007` and `ideate` moves with simplified `control_level`, lightweight `interventions`, optional `target_regimes`, and `--goal` scoping.
- [x] `src/contour/cli.py`: update `show`, `runs`, `compare`, `frontier`, `new` templates (drop `ARCH` column or replace with concise run title/tags).
- [x] `.agent/skills/contour-research/SKILL.md` & `skills/contour-research/SKILL.md` & sample nodes (`goals/`, `claims/`, `runs/`): update to match simplified schema and Critic advisory-flag review.
- [x] `tests/`: update and expand tests for nested config diffs, subtree swaps, `data_diff`, advisory flags, and simplified `control_level`.

---

### Workstream 4: Lean, Human-Readable Markdown & Sidecar Configs

Keep `runs/*.md` (and `analyses/*.md`) concise, source-of-truth documents that are pleasant to skim in GitHub, Obsidian, or an editor—even when real ML training configs are 100+ lines.

#### 4.1 Drop Persisted Derived `vs_parent` Cache
- Stop writing derived `vs_parent` blobs (~40 lines of nested YAML) into `runs/*.md` frontmatter.
- Continue computing `vs_parent` and `inputs_hash` dynamically in memory (`get_fresh_vs_parent`) for `show`, `runs`, `compare`, `check`, `ideate`, `report`, and `verified_hash` (`C008`).
- Eliminates `C006` stale-cache warnings and noisy `check --fix` Git churn when `contour.yaml` thresholds change, while keeping `runs/*.md` pure source-of-truth.

#### 4.2 Sidecar Configs (`runs/configs/<run_id>.yaml`)
- Store logged `Run` configs in tracked sidecar YAML files at `runs/configs/<run_id>.yaml`, referenced from frontmatter via `config_path: runs/configs/<run_id>.yaml`. For `analyses/`, keep small configs inline in frontmatter (≤10 leaf keys) and write a sidecar `analyses/configs/<ana_id>.yaml` only when large.
- `ContourNode.config` transparently resolves `config_path` (relative to Contour root) while remaining backwards-compatible with inline `config:` dicts (useful for quick manual runs/tests).
- `contour check` (`C001`) verifies that any referenced `config_path` exists and parses cleanly as a YAML mapping.
- `contour init` creates `runs/configs/` (tracked in Git so `compare` and reproducibility always work across checkouts).

#### 4.3 Compact Frontmatter Hygiene
- Omit empty/null/default fields (`baseline: null`, `intervention_note: null`, empty `artifacts: []`, etc.) when writing run/analysis Markdown files.
- Keep `runs/run_*.md` frontmatter down to ~15–20 crisp lines so `## Key Takeaway (TL;DR)` and `## Observations & Anomalies` are immediately visible without scrolling.

#### 4.4 Implementation Checklist (Workstream 4)
- [x] `src/contour/graph.py`: load `config` transparently from `config_path` (with fallback/merge for inline `config:`), recording load errors if missing/malformed.
- [x] `src/contour/run_logger.py`: write logged `Run` (and large/non-empty `Analysis`) configs to `runs/configs/<id>.yaml` (or `analyses/configs/<id>.yaml`), set `config_path`, stop writing derived `vs_parent` to frontmatter, and omit null/empty optional fields.
- [x] `src/contour/check.py` & `src/contour/cli.py`: remove `C006` / stale `vs_parent` fixing (keep `inputs_hash` in memory for `C008` `verified_hash`), validate `config_path` in `C001`, create `runs/configs/` in `contour init`.
- [x] `tests/` & sample nodes (`runs/`): update tests and sample run files for sidecar configs and lean frontmatter.

---

### Workstream 5: Claim ↔ Evidence Semantics

Align Claim linking and `C007` verification with the expanded evidence layer (`runs/` + `analyses/`) and simplified comparison model—keeping mechanical checks honest and minimal while leaving semantic polarity to the Tier-2 Critic and human.

#### 5.1 Crisp Definitions for the 3 Evidence Relations
Retain exactly three relations on Claims (`supports`, `contradicts`, `qualifies`):
- **`supports`:** Clean evidence (`run` or `analysis`) directly confirming the claim's prediction (including causal-necessity ablations or neutral runs for invariance claims, explained in `## Evidence Synthesis`).
- **`contradicts`:** Clean evidence directly refuting the claim *as currently stated inside `scope.holds_in`*.
- **`qualifies`:** Evidence that scopes, bounds, explains, or tentatively informs the claim without proving/refuting it inside `holds_in` (failure boundaries in `breaks_in`, tradeoffs, exploratory/dirty/probe runs, diagnostic analyses).

#### 5.2 Cleaner Mechanical `C007` Rules
- **`status: supported` requires:**
  1. Non-empty `supports`,
  2. Every item in `supports` is usable (`final/ok`) and `clean` (non-dirty, non-probe; for `analyses/`, all `subjects` are `final/ok`), and
  3. **`contradicts` is empty** — unresolved contradictions must be addressed (set `status: active`/`falsified`, or narrow the claim's title/`scope.holds_in`, record the failure in `scope.breaks_in`, and move the run to `qualifies`).
- **`status: falsified` requires:** at least one `clean` item in `contradicts`.
- **No mechanical verdict-polarity policing in `C007`:** Removes the `verdict in DECISIVE_VERDICTS` requirement so legitimate null/invariance claims (`clean/neutral`) and standalone absolute-threshold runs (`[no baseline]`) aren't falsely blocked. Direction/polarity alignment stays explicitly owned by the Tier-2 Critic (`verify --agent`) and human sign-off (`verify --human`), with `contour show <claim>` prominently displaying each linked item's `[control/verdict]` and advisory flags.

#### 5.3 `contour link` Behavior
- For `runs/` with a baseline: optional `--as` keeps sensible advisory default (`clean + win/efficiency_win -> supports`, `clean + regression -> contradicts`, else `qualifies`), overridable via `--as`.
- For `analyses/` (no directional verdict): default to `qualifies` unless `--as supports|contradicts|qualifies` is explicitly passed (preventing accidental rubber-stamping into `supports`).

#### 5.4 Implementation Checklist (Workstream 5)
- [x] `src/contour/check.py`: update `C007` to check cleanliness + non-empty `supports` + empty `contradicts` for `supported`, and clean `contradicts` for `falsified` (across both `runs/` and `analyses/`).
- [x] `src/contour/cli.py`: update `contour link` to accept both run and analysis IDs, require/default `qualifies` for analyses unless `--as` is passed, and display `[no baseline]` / advisory flags in `contour show <claim>`.
- [x] `.agent/skills/contour-research/SKILL.md`: document the sharpened `supports`/`contradicts`/`qualifies` definitions and Critic polarity check.
- [x] `tests/`: add tests for `C007` (clean neutral invariance claim, unresolved contradiction blocking `supported`, standalone clean run, clean analysis in `supports`, and `contour link` with analyses).

---

## Recommended Implementation Order

To avoid touching `graph.py`, `compare.py`, `check.py`, and `tests/` twice, execute in three clean passes:

1. **Pass 1 — Core Schema, Sidecar Configs & Transparent Diff (Workstreams 3 + 4):**
   - Simplify `Run` schema (drop `arch_family`/`arch_scale`, flexible `dataset_train`, lightweight `interventions`), add tracked sidecar configs (`runs/configs/`), drop persisted `vs_parent` (`C006`), implement nested dot-path `config_diff` + `data_diff` + `code_diff` + `compute_diff` + advisory `flags` + simplified `control_level`.
2. **Pass 2 — Expanded Evidence Layer & Claim Semantics (Workstreams 1 + 5):**
   - Add `analyses/` (`ana_*.md`, gitignored `analyses/scripts/` & `analyses/artifacts/`), `Analysis`/`Eval` Python API + `script_sha256`, `artifacts` on `runs/`, taint propagation, eval composition in `compare`/`frontier`/`show`, and updated `C007`/`C009`/`contour link` rules.
3. **Pass 3 — Daily Workflows, Docs & Sample State (Workstream 2):**
   - Implement `contour brief`, `open_questions` on `Goal`, `--status` filtering, `contour report` (Integrity Banner, `--strict`, preliminary `--runs`/`--analyses`, `--out <dir>` portable bundling), update `SKILL.md` / `README.md` / sample nodes, and run full test suite.

---

## Explicitly Deferred (Keep Minimal Until Real Usage Demands It)

- **Clean training divergences:** Handle via normal `final` runs (logging last/collapsed metrics + takeaway) or an `Analysis` of the training curve; revisit only if a distinct `diverged` status becomes genuinely necessary.
- **Multi-seed & sweep groupings:** Handle via multi-subject `Analysis` scripts (`subjects: [seed_0, seed_1, seed_2]`) computing mean ± std rather than adding a new Sweep/Group abstraction.
- **Pre-flight CLI check (`contour preflight`):** Easy to add later on top of the Pass 1 diff engine if config drift at launch time becomes an actual pain point.
