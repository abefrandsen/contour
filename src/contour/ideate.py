"""`contour ideate`: state-conditioned experiment generation across 6 Scientific Moves (§6.2)."""

from __future__ import annotations

from typing import Any

from .check import compute_frontier, frontier_metrics
from .compare import compute_diff_accountability, get_fresh_vs_parent
from .graph import FACET_KEYS, ContourGraph, ContourNode
from .mdio import strip_wiki


def _facets_tuple(run: ContourNode) -> tuple[Any, ...]:
    fm = run.frontmatter
    ds = fm.get("dataset_train") or []
    ds_norm = tuple(sorted(str(x) for x in ([ds] if isinstance(ds, str) else ds)))
    return (
        fm.get("task"),
        fm.get("arch_family"),
        fm.get("arch_scale"),
        ds_norm,
        fm.get("tier"),
    )


def _primary_beneficial_delta(
    g: ContourGraph, vs: dict[str, Any]
) -> tuple[str | None, float | None, float | None]:
    """Return (metric_key, beneficial_delta, min_delta) for the first primary metric in deltas."""
    deltas = vs.get("deltas") or {}
    for k, d in deltas.items():
        if g.config.is_primary_metric(k):
            direction = g.config.metric_direction(k)
            raw = float(d["delta"])
            beneficial = raw if direction == "max" else -raw
            return k, round(beneficial, 6), g.config.min_delta(k)
    return None, None, None


def _has_positive_primary_delta(g: ContourGraph, vs: dict[str, Any]) -> bool:
    deltas = vs.get("deltas") or {}
    for k, d in deltas.items():
        if g.config.is_primary_metric(k):
            direction = g.config.metric_direction(k)
            raw = float(d["delta"])
            beneficial = raw if direction == "max" else -raw
            if beneficial > 0:
                return True
    return False


def generate_ideation(
    g: ContourGraph,
    goal_id: str | None = None,
) -> dict[str, Any]:
    """Run all 6 Scientific Move detectors over usable runs and fresh `vs_parent` (§6.2)."""
    usable = g.usable_runs()
    if goal_id:
        gid = strip_wiki(goal_id)
        target_claims = g.goal_claims(gid)
        target_goals = [g.goals[gid]] if gid in g.goals else []
    else:
        target_claims = list(g.claims.values())
        target_goals = list(g.goals.values())

    fresh_vs: dict[str, dict[str, Any] | None] = {
        r.id: get_fresh_vs_parent(g, r) for r in usable
    }

    # --- Move 1: Isolate confounds -----------------------------------------
    clean_pairs = {
        (r.effective_baseline, frozenset(r.interventions))
        for r in usable
        if r.effective_baseline and (fresh_vs.get(r.id) or {}).get("control_level") == "clean"
    }
    m1_isolate: list[dict[str, Any]] = []
    for r in usable:
        vs = fresh_vs.get(r.id)
        if not vs or vs.get("control_level") != "confounded":
            continue
        if not _has_positive_primary_delta(g, vs):
            continue
        base_id = r.effective_baseline
        if base_id and (base_id, frozenset(r.interventions)) in clean_pairs:
            continue

        # Look for an existing matching-facet run whose interventions are a strict subset
        # and that leaves zero unaccounted/unrealized diffs when used as baseline
        r_facets = _facets_tuple(r)
        r_intervs = set(r.interventions)
        bridge_candidates: list[str] = []
        for cand in usable:
            if cand.id == r.id or _facets_tuple(cand) != r_facets:
                continue
            if not (set(cand.interventions) < r_intervs):
                continue
            diff = compute_diff_accountability(
                cand.frontmatter, r.frontmatter, g.config.ignored_config_keys
            )
            if not diff["unaccounted"] and not diff["unrealized"]:
                bridge_candidates.append(cand.id)
        mkey, bdelta, _ = _primary_beneficial_delta(g, vs)
        if bridge_candidates:
            suggestion = (
                f"Existing facet-matched baseline found ({bridge_candidates[0]}). "
                f"Set `baseline: '[[{bridge_candidates[0]}]]'` on {r.id} and run `contour check --fix`."
            )
        else:
            suggestion = (
                f"Run a single-diff bridge control matching {r.id}'s facets "
                f"without its intervention(s) (unaccounted={vs.get('unaccounted')}, unrealized={vs.get('unrealized')})."
            )
        m1_isolate.append({
            "run": r.id,
            "baseline": base_id,
            "primary_metric": mkey,
            "primary_delta": bdelta,
            "unaccounted": vs.get("unaccounted") or [],
            "unrealized": vs.get("unrealized") or [],
            "bridge_candidates": bridge_candidates,
            "estimated_gpu_hours": r.gpu_hours,
            "suggestion": suggestion,
        })

    # --- Move 2: Resolve tensions ------------------------------------------
    m2_tensions: list[dict[str, Any]] = []
    for c in sorted(target_claims, key=lambda x: x.id):
        con, qual = c.refs("contradicts"), c.refs("qualifies")
        if not (con or qual):
            continue
        linked = [g.runs[rid] for rid in (con + qual) if rid in g.runs]
        costs = [r.gpu_hours for r in linked if r.gpu_hours is not None]
        est_cost = round(sum(costs) / len(costs), 3) if costs else None
        scope = c.frontmatter.get("scope") or {}
        m2_tensions.append({
            "kind": "claim_tension",
            "claim": c.id,
            "status": c.status,
            "title": c.title,
            "contradicts": con,
            "qualifies": qual,
            "breaks_in": scope.get("breaks_in") or [],
            "estimated_gpu_hours": est_cost,
        })
    for r in usable:
        vs = fresh_vs.get(r.id)
        if vs and vs.get("verdict") == "tradeoff":
            m2_tensions.append({
                "kind": "run_tradeoff",
                "run": r.id,
                "baseline": r.effective_baseline,
                "interventions": r.interventions,
                "estimated_gpu_hours": r.gpu_hours,
                "deltas": {
                    k: d for k, d in (vs.get("deltas") or {}).items()
                    if d.get("signal") in ("clear", "regression")
                },
            })

    # --- Move 3: Combine wins (including compounding marginal wins) --------
    def _cumulative_interventions(run: ContourNode, stop_at: str) -> frozenset[str] | None:
        tags: set[str] = set(run.interventions)
        cur = run
        visited: set[str] = {run.id}
        while cur.effective_baseline and cur.effective_baseline != stop_at:
            bid = cur.effective_baseline
            if bid in visited or bid not in g.runs:
                return None
            visited.add(bid)
            cur = g.runs[bid]
            tags.update(cur.interventions)
        return frozenset(tags) if cur.effective_baseline == stop_at else None

    def _kv_map(intervs: list[str]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for iv in intervs:
            if "=" in iv:
                k, v = iv.split("=", 1)
                out.setdefault(k.strip(), set()).add(v.strip())
        return out

    all_interv_sets = {frozenset(r.interventions) for r in usable}
    by_base: dict[str, list[ContourNode]] = {}
    for r in usable:
        vs = fresh_vs.get(r.id)
        if (
            vs is not None
            and r.effective_baseline
            and vs.get("control_level") == "clean"
            and vs.get("verdict") in ("win", "efficiency_win", "marginal")
            and r.interventions
        ):
            by_base.setdefault(r.effective_baseline, []).append(r)

    m3_combine: list[dict[str, Any]] = []
    for base_id, siblings in sorted(by_base.items()):
        seen_since_base: set[frozenset[str]] = {
            frozenset(r.interventions) for r in usable if r.effective_baseline == base_id
        }
        for r in usable:
            if r.effective_baseline and r.effective_baseline != base_id:
                cum = _cumulative_interventions(r, base_id)
                if cum is not None:
                    seen_since_base.add(cum)
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                r1, r2 = siblings[i], siblings[j]
                s1, s2 = frozenset(r1.interventions), frozenset(r2.interventions)
                combined = s1 | s2
                if s1 == s2 or combined in seen_since_base:
                    continue
                kv1, kv2 = _kv_map(r1.interventions), _kv_map(r2.interventions)
                if any(len(kv1[k] | kv2[k]) > 1 for k in (set(kv1) & set(kv2))):
                    continue
                vs1, vs2 = fresh_vs[r1.id] or {}, fresh_vs[r2.id] or {}
                d1_map = vs1.get("deltas") or {}
                d2_map = vs2.get("deltas") or {}
                shared_primary = [
                    k for k in (set(d1_map) & set(d2_map))
                    if g.config.is_primary_metric(k)
                ]
                if not shared_primary:
                    continue
                pk = sorted(shared_primary)[0]
                direction = g.config.metric_direction(pk)
                sign = 1.0 if direction == "max" else -1.0
                p1 = float(d1_map[pk]["delta"]) * sign
                p2 = float(d2_map[pk]["delta"]) * sign
                projected = round(p1 + p2, 6)
                tau = g.config.min_delta(pk)
                crosses = (projected >= tau) if tau is not None else None

                c1_hr, c2_hr = r1.gpu_hours, r2.gpu_hours
                est_cost = (
                    round((c1_hr + c2_hr) / 2.0, 3)
                    if (c1_hr is not None and c2_hr is not None)
                    else (c1_hr or c2_hr)
                )
                m3_combine.append({
                    "baseline": base_id,
                    "left": {"id": r1.id, "verdict": vs1.get("verdict"), "interventions": r1.interventions},
                    "right": {"id": r2.id, "verdict": vs2.get("verdict"), "interventions": r2.interventions},
                    "combined_interventions": sorted(combined),
                    "primary_metric": pk,
                    "projected_primary_delta": projected,
                    "min_delta": tau,
                    "crosses_threshold": crosses,
                    "estimated_gpu_hours": est_cost,
                })
    # Rank pairs crossing threshold tau first, then by projected delta descending
    m3_combine.sort(
        key=lambda x: (
            not bool(x.get("crosses_threshold")),
            -(x.get("projected_primary_delta") or 0.0),
        )
    )

    # --- Move 4: Strip complexity (Occam's Razor) --------------------------
    m4_ablate: list[dict[str, Any]] = []
    seen_sets: set[frozenset[str]] = set()
    for r in usable:
        vs = fresh_vs.get(r.id)
        if (
            not vs
            or vs.get("verdict") not in ("win", "efficiency_win")
            or len(r.interventions) <= 1
        ):
            continue
        iset = frozenset(r.interventions)
        if iset in seen_sets:
            continue
        seen_sets.add(iset)
        # Any sibling run on the same baseline with a strict non-empty subset already ablated the stripped tags
        already_stripped: set[str] = set()
        for other in usable:
            if other.id != r.id and other.effective_baseline == r.effective_baseline:
                oset = frozenset(other.interventions)
                if oset and oset < iset:
                    already_stripped.update(iset - oset)
        tags = sorted(iset)
        leave_one_out = [
            {"strip": t, "keep": [x for x in tags if x != t]}
            for t in tags
            if t not in already_stripped
            and frozenset(x for x in tags if x != t) not in all_interv_sets
        ]
        if leave_one_out:
            m4_ablate.append({
                "run": r.id,
                "verdict": vs.get("verdict"),
                "interventions": tags,
                "candidate_ablations": leave_one_out,
                "estimated_gpu_hours": r.gpu_hours,
            })

    # --- Move 5: Expand regimes --------------------------------------------
    target_tasks: list[str] = []
    target_scales: list[str] = []
    for goal in target_goals:
        tr = goal.frontmatter.get("target_regimes") or {}
        for t in tr.get("tasks") or []:
            if str(t) not in target_tasks:
                target_tasks.append(str(t))
        for s in tr.get("scales") or []:
            if str(s) not in target_scales:
                target_scales.append(str(s))

    m5_expand: list[dict[str, Any]] = []
    for c in sorted(target_claims, key=lambda x: x.id):
        if c.status not in ("active", "supported"):
            continue
        linked_runs = [g.runs[rid] for rid in (c.refs("supports") + c.refs("qualifies")) if rid in g.runs]
        covered_pairs = {
            (str(r.get("task")), str(r.get("arch_scale")))
            for r in linked_runs
            if r.get("task") and r.get("arch_scale")
        }
        covered_tasks = sorted({t for t, _ in covered_pairs})
        covered_scales = sorted({s for _, s in covered_pairs})
        missing_tasks = [t for t in target_tasks if t not in covered_tasks]
        missing_scales = [s for s in target_scales if s not in covered_scales]
        scope_untested = list((c.frontmatter.get("scope") or {}).get("untested") or [])

        if missing_tasks or missing_scales or scope_untested:
            costs = [r.gpu_hours for r in linked_runs if r.gpu_hours is not None]
            est_cost = round(sum(costs) / len(costs), 3) if costs else None
            m5_expand.append({
                "claim": c.id,
                "status": c.status,
                "covered_tasks": covered_tasks,
                "covered_scales": covered_scales,
                "missing_target_tasks": missing_tasks,
                "missing_target_scales": missing_scales,
                "scope_untested": scope_untested,
                "estimated_gpu_hours": est_cost,
            })

    # --- Move 6: Diagnose frontier -----------------------------------------
    all_tasks = sorted({str(r.get("task")) for r in usable if r.get("task")})
    m6_frontier: list[dict[str, Any]] = []
    for t in all_tasks:
        t_runs = [r for r in usable if r.get("task") == t]
        for fm_metric in frontier_metrics(g, t_runs):
            metric, direction, ranked, _ = compute_frontier(g, task=t, metric_filter=fm_metric)
            if not ranked:
                continue
            best = ranked[0]
            obs_sec = best.sections.get("Observations & Anomalies", "").strip()
            obs_lines = [ln.lstrip("- ").strip() for ln in obs_sec.splitlines() if ln.strip()]
            diagnostics = {
                k: v for k, v in best.metrics.items()
                if g.config.metric_kind(k) == "diagnostic"
            }
            m6_frontier.append({
                "task": t,
                "best_run": best.id,
                "title": best.title,
                "metric": metric,
                "direction": direction,
                "value": best.metrics.get(metric or ""),
                "estimated_gpu_hours": best.gpu_hours,
                "observations": obs_lines[:3],
                "diagnostic_metrics": diagnostics,
            })

    return {
        "move_1_isolate_confounds": m1_isolate,
        "move_2_resolve_tensions": m2_tensions,
        "move_3_combine_wins": m3_combine,
        "move_4_strip_complexity": m4_ablate,
        "move_5_expand_regimes": m5_expand,
        "move_6_diagnose_frontier": m6_frontier,
    }
