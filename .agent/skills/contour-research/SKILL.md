---
name: contour-research
description: Navigate, survey, and reason over the Contour 3-layer ML Epistemic Graph (Expeditions, Cairns, and Theodolite-logged FieldLogs). Use this skill whenever ingesting new ML runs, updating research cairns/claims, inspecting Pareto ridgelines, or generating next experimental bearings.
---

# Contour: Epistemic ML Research Copilot Skill

**Contour** organizes empirical ML research as a 3-layer topographical & epistemic graph:
1. **Layer 1 (Empirical Ground Truth — `Theodolite` & `fieldlogs/`):** Logged via `tools/theodolite.py` into Aim (`.aim`) and distilled via `./contour sync` into `fieldlogs/log_*.md` cards.
2. **Layer 2 (Epistemic Topography — `cairns/` & `expeditions/`):** Falsifiable hypotheses/principles (`cairns/crn_*.md`) and macro research campaigns (`expeditions/expd_*.md`) linked to fieldlogs via typed edges (`supports`, `contradicts`, `qualifies`).
3. **Layer 3 (Navigation & Ideation — `./contour` CLI):** Fast deterministic graph retrieval paired with the 6 Scientific Moves below.

---

## Core Rule: Progressive Context Loading

**Never read all `.md` files in `fieldlogs/` or `cairns/` at once.** Always use `./contour` first to survey the graph, then `inspect` only the 2–5 relevant nodes:

```bash
# 1. Print the macro DAG of Expeditions -> Cairns -> FieldLog counts
./contour map

# 2. Check for terrain drift (stale cairns or unlinked/invalidated fieldlogs)
./contour drift

# 3. Compare verified fieldlogs & plot the Pareto ridgeline on a benchmark
./contour ridge --eval imagenet1k_val

# 4. Run the 6 Scientific Moves to surface next experimental bearings
./contour bearings

# 5. Expand context on a specific Expedition, Cairn, or FieldLog
./contour inspect <node_id>
```

---

## Protocol 1: Surveying Runs & Maintaining Cairns (Terrain Drift)

When new experiments complete (or when asked to sync and synthesize):

1. **Sync Layer 1 $\rightarrow$ Layer 2:**
   * Run `./contour sync` (or `./contour sync --from-json <bundle.json>` for cluster field bundles).
   * Run `./contour drift` to list newly created fieldlogs, unlinked runs, or cairns flagged with `needs_synthesis: true`.
2. **Evaluate Evidence Rigorously (Anti-Sycophancy Check):**
   * Inspect each new fieldlog and its `parent_runs` (`./contour inspect <log_id>`).
   * Before adding a fieldlog to a Cairn's `supports` list, check for **confounders**:
     * Did `compute_gpu_hours` or `dataset_train` change alongside the intervention?
     * Could a gain on a small dataset be an implicit regularization artifact rather than the hypothesized mechanism?
     * If a run shows mixed results or boundary conditions, place it under `qualifies` or `contradicts` rather than forcing it into `supports`.
3. **Update the Cairn Prose:**
   * Keep YAML frontmatter strictly to graph links (`supports`, `contradicts`, `qualifies`, `status`, `needs_synthesis: false`).
   * Update `## Where This Holds vs. Breaks (Scope & Regimes)` in concise Markdown bullets. Never delete historical anomalies—explain them or keep them visible.

---

## Protocol 2: Taking Bearings (The 6 Scientific Moves)

When asked to brainstorm, prioritize, or propose next experiments, run `./contour bearings` and evaluate opportunities across the **6 First-Principles Moves**:

1. **Move 0 — Drift & Hygiene (`./contour drift`):** Fix any invalidated evidence or synthesize unlinked fieldlogs before proposing new climbs.
2. **Move 1 — Dialectics (`contradicts` / `qualifies`):** Isolate the hidden confounding variable between a supporting fieldlog and a contradicting/anomalous fieldlog. Propose a **crucial experiment** that varies *only* that confounder.
3. **Move 2 — Deduction (*Terra Incognita* / Regime Expansion $R \rightarrow R'$):** Take a `working_assumption` or `well_supported` cairn verified in only one regime (e.g., 22M classification) and design the **minimal transfer/scaling test** (e.g., dense prediction, sequence task, or 100M scale).
4. **Move 3 — Synthesis (Composition $I_1 + I_2$):** Combine two orthogonal Pareto-winning interventions that branched from the same baseline on the same eval benchmark to test whether gains stack additively or interfere.
5. **Move 4 — Occam's Razor (Subtractive Ablation $I \setminus i_k$):** When a winning fieldlog bundled multiple changes (`A + B`), propose a simplification run removing one component to verify whether all parts are earning their complexity.
6. **Move 5 — Abduction (Ridgeline Bottleneck Diagnosis $\text{Errors}(O) \rightarrow C_{\text{new}}$):** Inspect the current Pareto-ridgeline run on the target benchmark (`./contour ridge`), read its `## Observations & Anomalies` (and Aim curves if available), diagnose what bottlenecks it, and stack a new Cairn (`./contour new cairn <slug> --title "..."`).

### Ranking Proposals
Always rank proposed experiments by **Expected Value of Information (VoI) per GPU Hour**:
$$\text{Priority} \propto \frac{\text{Epistemic Uncertainty Resolved} \times \text{Centrality to Active Expedition}}{\text{Estimated GPU Hours}}$$
Prefer cheap, decisive falsification tests before expensive full-scale ascents.
