"""`contour ideate`: state-conditioned experiment generation driven directly by diffs, tensions, and open questions."""

from __future__ import annotations

from typing import Any

from .check import compute_frontier, frontier_metrics
from .compare import get_fresh_vs_parent
from .graph import EDGE_KEYS, ContourGraph, ContourNode
from .mdio import strip_wiki


def _has_high_risk_flag(flags: list[str]) -> bool:
    return any(fl == "code_dirty" or fl.startswith("eval_drift:") for fl in flags)


def _primary_beneficial_delta(
    g: ContourGraph, vs: dict[str, Any]
) -> tuple[str | None, float | None, float | None]:
    deltas = vs.get("deltas") or {}
    for k, d in deltas.items():
        if g.config.is_primary_metric(k):
            direction = g.config.metric_direction(k)
            raw = float(d["delta"])
            beneficial = raw if direction == "max" else -raw
            tau = g.config.min_delta(k)
            if tau is None and g.config.min_rel_delta(k) is not None:
                tau = abs(float(d["baseline"])) * float(g.config.min_rel_delta(k) or 0.0)
            return k, round(beneficial, 6), tau
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


def _runs_for_goal(g: ContourGraph, goal_id: str) -> list[ContourNode]:
    gid = strip_wiki(goal_id)
    g_claims = g.goal_claims(gid)
    claim_ids = {c.id for c in g_claims}
    allowed: set[str] = set()
    for c in g_claims:
        for edge in EDGE_KEYS:
            for eid in c.refs(edge):
                if eid in g.runs:
                    allowed.add(eid)
                elif eid in g.analyses:
                    allowed.update(g.analyses[eid].subjects)
    for r in g.runs.values():
        mot = set(r.refs("motivated_by"))
        if gid in mot or (mot & claim_ids):
            allowed.add(r.id)
    for rid in list(allowed):
        run = g.runs.get(rid)
        if run and run.parent:
            allowed.add(run.parent)
    return [r for r in g.usable_runs() if r.id in allowed]


def _diff_units(vs: dict[str, Any]) -> list[str]:
    """Return conceptual changed units: subtree block names (e.g. `encoder.*`) + standalone keys."""
    blocks = vs.get("config_blocks") or {}
    cdiff = vs.get("config_diff") or {}
    block_keys = {k for info in blocks.values() for k in info.get("keys", [])}
    units = [f"{blk}.*" for blk in sorted(blocks)]
    units.extend(sorted(k for k in cdiff if k not in block_keys))
    return units


def generate_ideation(
    g: ContourGraph,
    goal_id: str | None = None,
) -> dict[str, Any]:
    """Run all 6 deterministic ideation moves over usable runs, analyses, claims, and goals."""
    if goal_id:
        gid = strip_wiki(goal_id)
        target_claims = g.goal_claims(gid)
        target_goals = [g.goals[gid]] if gid in g.goals else []
        usable = _runs_for_goal(g, gid)
    else:
        target_claims = list(g.claims.values())
        target_goals = list(g.goals.values())
        usable = g.usable_runs()

    fresh_vs: dict[str, dict[str, Any] | None] = {
        r.id: get_fresh_vs_parent(g, r) for r in usable
    }

    # --- Move 1: Investigate exploratory / flagged positive runs ------------
    m1_flagged: list[dict[str, Any]] = []
    for r in usable:
        vs = fresh_vs.get(r.id)
        if not vs or not _has_positive_primary_delta(g, vs):
            continue
        flags = list(vs.get("flags") or [])
        ctrl = vs.get("control_level")
        if ctrl != "exploratory" and not _has_high_risk_flag(flags):
            continue
        mkey, bdelta, _ = _primary_beneficial_delta(g, vs)
        if ctrl == "exploratory":
            suggestion = f"Re-run {r.id} on a clean git commit to confirm this exploratory gain."
        else:
            suggestion = f"Re-run or audit {r.id} to resolve advisory flag(s) {flags}."
        m1_flagged.append({
            "run": r.id,
            "parent": r.parent,
            "control_level": ctrl,
            "flags": flags,
            "primary_metric": mkey,
            "primary_delta": bdelta,
            "estimated_gpu_hours": r.gpu_hours,
            "suggestion": suggestion,
        })

    # --- Move 2: Resolve tensions & tradeoffs -------------------------------
    m2_tensions: list[dict[str, Any]] = []
    for c in sorted(target_claims, key=lambda x: x.id):
        con, qual = c.refs("contradicts"), c.refs("qualifies")
        if not (con or qual):
            continue
        linked_runs = [g.runs[eid] for eid in (con + qual) if eid in g.runs]
        linked_anas = [g.analyses[eid].id for eid in (con + qual) if eid in g.analyses]
        costs = [r.gpu_hours for r in linked_runs if r.gpu_hours is not None]
        est_cost = round(sum(costs) / len(costs), 3) if costs else None
        m2_tensions.append({
            "kind": "claim_tension",
            "claim": c.id,
            "status": c.status,
            "title": c.title,
            "contradicts": con,
            "qualifies": qual,
            "analyses": linked_anas,
            "estimated_gpu_hours": est_cost,
        })
    for r in usable:
        vs = fresh_vs.get(r.id)
        if vs and vs.get("verdict") == "tradeoff":
            m2_tensions.append({
                "kind": "run_tradeoff",
                "run": r.id,
                "parent": r.parent,
                "title": r.title,
                "changed_units": _diff_units(vs),
                "estimated_gpu_hours": r.gpu_hours,
                "deltas": {
                    k: d for k, d in (vs.get("deltas") or {}).items()
                    if d.get("signal") in ("clear", "regression")
                },
            })

    # --- Move 3: Combine & compound clean sibling wins (disjoint config_diff) -
    by_parent: dict[str, list[ContourNode]] = {}
    for r in usable:
        vs = fresh_vs.get(r.id)
        if (
            vs is not None
            and r.parent
            and vs.get("control_level") == "clean"
            and not _has_high_risk_flag(vs.get("flags") or [])
            and vs.get("verdict") in ("win", "efficiency_win", "marginal")
            and vs.get("config_diff")
        ):
            by_parent.setdefault(r.parent, []).append(r)

    m3_combine: list[dict[str, Any]] = []
    for parent_id, siblings in sorted(by_parent.items()):
        # Collect already-tried combined keysets descending from parent_id
        seen_keysets: set[frozenset[str]] = set()
        for r in usable:
            vs = fresh_vs.get(r.id)
            if r.parent == parent_id and vs and vs.get("config_diff"):
                seen_keysets.add(frozenset(vs["config_diff"].keys()))
            elif r.parent and r.parent in g.runs and g.runs[r.parent].parent == parent_id:
                p_vs = fresh_vs.get(r.parent)
                if vs and p_vs:
                    seen_keysets.add(
                        frozenset((vs.get("config_diff") or {}).keys())
                        | frozenset((p_vs.get("config_diff") or {}).keys())
                    )

        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                r1, r2 = siblings[i], siblings[j]
                vs1, vs2 = fresh_vs[r1.id] or {}, fresh_vs[r2.id] or {}
                k1 = frozenset((vs1.get("config_diff") or {}).keys())
                k2 = frozenset((vs2.get("config_diff") or {}).keys())
                if not k1 or not k2 or (k1 & k2):
                    continue  # Overlapping config keys -> conflict
                combined_keys = k1 | k2
                if combined_keys in seen_keysets:
                    continue

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
                if tau is None and g.config.min_rel_delta(pk) is not None:
                    tau = abs(float(d1_map[pk]["baseline"])) * float(g.config.min_rel_delta(pk) or 0.0)
                crosses = (projected >= tau) if tau is not None else None

                c1_hr, c2_hr = r1.gpu_hours, r2.gpu_hours
                est_cost = (
                    round((c1_hr + c2_hr) / 2.0, 3)
                    if (c1_hr is not None and c2_hr is not None)
                    else (c1_hr or c2_hr)
                )
                m3_combine.append({
                    "parent": parent_id,
                    "left": {"id": r1.id, "verdict": vs1.get("verdict"), "changed_keys": sorted(k1)},
                    "right": {"id": r2.id, "verdict": vs2.get("verdict"), "changed_keys": sorted(k2)},
                    "combined_keys": sorted(combined_keys),
                    "primary_metric": pk,
                    "projected_primary_delta": projected,
                    "min_delta": tau,
                    "crosses_threshold": crosses,
                    "estimated_gpu_hours": est_cost,
                })
    m3_combine.sort(
        key=lambda x: (
            not bool(x.get("crosses_threshold")),
            -(x.get("projected_primary_delta") or 0.0),
        )
    )

    # --- Move 4: Strip complexity (ablate multi-unit winning config_diffs) --
    m4_ablate: list[dict[str, Any]] = []
    for r in usable:
        vs = fresh_vs.get(r.id)
        if (
            not vs
            or vs.get("control_level") != "clean"
            or vs.get("verdict") not in ("win", "efficiency_win")
        ):
            continue
        units = _diff_units(vs)
        if len(units) <= 1:
            continue
        # Check if sibling runs on the same parent already ablated a strict subset of units
        already_stripped: set[str] = set()
        unit_set = frozenset(units)
        for other in usable:
            if other.id != r.id and other.parent == r.parent:
                ovs = fresh_vs.get(other.id)
                if ovs:
                    oset = frozenset(_diff_units(ovs))
                    if oset and oset < unit_set:
                        already_stripped.update(unit_set - oset)
        leave_one_out = [
            {"strip": u, "keep": [x for x in units if x != u]}
            for u in units
            if u not in already_stripped
        ]
        if leave_one_out:
            m4_ablate.append({
                "run": r.id,
                "title": r.title,
                "verdict": vs.get("verdict"),
                "changed_units": units,
                "candidate_ablations": leave_one_out,
                "estimated_gpu_hours": r.gpu_hours,
            })

    # --- Move 5: Open questions & unsupported active claims -----------------
    m5_open: list[dict[str, Any]] = []
    for goal in sorted(target_goals, key=lambda x: x.id):
        if goal.status != "active":
            continue
        for q in goal.open_questions:
            m5_open.append({"kind": "open_question", "goal": goal.id, "question": q})
    for c in sorted(target_claims, key=lambda x: x.id):
        if c.status == "active" and not c.refs("supports"):
            m5_open.append({
                "kind": "unsupported_claim",
                "claim": c.id,
                "title": c.title,
            })

    # --- Move 6: Diagnose frontier -----------------------------------------
    usable_ids = {r.id for r in usable}
    m6_frontier: list[dict[str, Any]] = []
    for fm_metric in frontier_metrics(g, usable):
        metric, direction, ranked, _ = compute_frontier(g, metric_filter=fm_metric)
        ranked = [r for r in ranked if r.id in usable_ids]
        if not ranked:
            continue
        best = ranked[0]
        eff_metrics = g.effective_run_metrics(best)
        obs_sec = best.sections.get("Observations & Anomalies", "").strip()
        obs_lines = [ln.lstrip("- ").strip() for ln in obs_sec.splitlines() if ln.strip()]
        diagnostics = {
            k: v for k, v in eff_metrics.items()
            if g.config.metric_kind(k) == "diagnostic"
        }
        attached_analyses = [
            {"id": a.id, "kind": a.kind, "tldr": a.tldr}
            for a in g.run_analyses(best.id)
            if a.is_usable_evidence
        ]
        m6_frontier.append({
            "best_run": best.id,
            "title": best.title,
            "metric": metric,
            "direction": direction,
            "value": eff_metrics.get(metric or ""),
            "estimated_gpu_hours": best.gpu_hours,
            "observations": obs_lines[:3],
            "diagnostic_metrics": diagnostics,
            "attached_analyses": attached_analyses,
        })

    return {
        "move_1_investigate_flagged": m1_flagged,
        "move_2_resolve_tensions": m2_tensions,
        "move_3_combine_wins": m3_combine,
        "move_4_strip_complexity": m4_ablate,
        "move_5_open_questions": m5_open,
        "move_6_diagnose_frontier": m6_frontier,
    }
