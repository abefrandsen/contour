---
name: contour-research
description: Navigate, survey, and reason over the Contour ML Epistemic Graph (Expeditions, Cairns, and FieldLogs). Use this skill whenever ingesting new ML runs, comparing runs, updating cairns, checking for confounded diffs, or generating next experimental bearings.
---

# Contour: Epistemic ML Research Copilot Skill

**Contour** maintains a clean, low-friction Markdown + Git research state over many ML training runs:
1. **Empirical Ground Truth (`fieldlogs/log_*.md`, `bundles/*.json`, optional `noise.yaml`):** Any training platform (or `from contour import Run`) emits a `contour.field_bundle/v1` JSON bundle; `./contour sync` validates, archives, and distills it into `fieldlogs/` with an auto-computed `vs_parent` diff (`single_logical_diff`, `confounding_facets`, metric deltas, and `A`/`B`/`C` quality badge).
2. **Research Claims (`expeditions/expd_*.md`, `cairns/crn_*.md`):** Cairns hold falsifiable hypotheses and link directly to `fieldlogs/` via `supports`, `contradicts`, and `qualifies`.
3. **Experiment Generation (`./contour bearings`):** Mines the run graph to surface confounded runs needing isolation, contradictions, orthogonal Pareto wins to combine, multi-tag runs to ablate, untested regime gaps, and Pareto frontier bottlenecks.

---

## Core Rule: Progressive Context Loading

**Never read all `.md` files at once.** Use `./contour` (with `--json` when structured output helps) to survey the graph, then `inspect` only the 2–5 relevant nodes:

```bash
./contour map --json
./contour bearings --json
./contour ridge --eval imagenet1k_val --json
./contour drift --json
./contour inspect <node_id>
```

---

## Protocol 1: Ingesting, Comparing, and Synthesizing Runs

1. **Sync runs:** `./contour sync` (or `./contour sync --from-json <bundle_or_dir>`).
   - Every child run automatically gets a `vs_parent` block comparing it against its first resolved parent run.
2. **Inspect or compare runs:**
   - `./contour inspect <log_id>` shows the run's metrics and its `vs_parent` diff.
   - `./contour compare <ctrl_ids...> --vs <treat_ids...>` computes an on-the-fly comparison between any control and treatment run(s) (single-seed or multi-seed).
   - Always check `single_logical_diff` and `confounding_facets` before attributing a metric gain to an intervention.
   - Optionally seed baseline variance once per regime via `./contour noise seed <task/eval/scale> --from <baseline_logs...>` to unlock `SNR` and Grade `A` badges on single-seed runs.
3. **Link runs & update Cairn prose:**
   - `./contour link <log_id> <cairn_id> --relation <supports|contradicts|qualifies>` places the run on the cairn and clears `needs_synthesis`.
   - Update the Cairn's Markdown body (`## Where This Holds vs. Breaks`) to reflect what holds, what fails, and what remains untested.

---

## Protocol 2: Generating Next Experiments (`./contour bearings`)

Run `./contour bearings` to surface concrete next experiments across the **6 Scientific Moves**:
- **Move 0 (Drift & Hygiene):** Fix stale cairns (`needs_synthesis: true`), unlinked motivating runs (`H003`), or confounded comparisons (`H012`).
- **Move 1 (Confound Isolation):** When a run changed multiple facets vs its parent (e.g., intervention + `dataset_train`), run the clean single-diff control experiment first.
- **Move 2 (Dialectics):** Investigate `contradicts` and `qualifies` boundary regimes on active cairns.
- **Move 3 (Synthesis):** Combine sibling Pareto-improving interventions (`c1.interventions | c2.interventions`) that branched from the same parent.
- **Move 4 (Occam's Razor):** Run leave-one-out ablations on multi-tag runs (`len(interventions) > 1`) to strip unnecessary complexity.
- **Move 5 (Deduction):** Expand verified cairns into untested tasks, evals, or model scales.
- **Move 6 (Abduction):** Inspect the top Pareto-frontier run on each benchmark (`./contour ridge`) and target its primary anomaly/bottleneck.
