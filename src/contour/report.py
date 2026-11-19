"""Daily startup briefings (`contour brief`) and evidence-grounded reports (`contour report`)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import shutil
from typing import Any

from .check import (
    ERROR,
    compute_verified_hash,
    effective_verification,
    evidence_control_level,
    find_check_issues,
    find_review_items,
    is_verification_stale,
)
from .compare import compare_runs, get_fresh_vs_parent
from .graph import EDGE_KEYS, ContourGraph, ContourNode, is_remote_uri
from .ideate import generate_ideation
from .mdio import strip_wiki
from .run_logger import _get_git_info


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Morning Startup Briefing (`contour brief`)
# ---------------------------------------------------------------------------


def build_brief(g: ContourGraph, *, goal_id: str | None = None) -> dict[str, Any]:
    """Assemble a single high-signal morning startup snapshot of live research state."""
    gid = strip_wiki(goal_id) if goal_id else None
    issues = find_check_issues(g)
    review = find_review_items(g)
    ideate = generate_ideation(g, goal_id=gid)

    target_goals = (
        [g.goals[gid]] if (gid and gid in g.goals) else sorted(g.goals.values(), key=lambda x: x.id)
    )
    target_claims = (
        g.goal_claims(gid) if gid else sorted(g.claims.values(), key=lambda x: x.id)
    )
    target_claim_ids = {c.id for c in target_claims}

    if gid:
        review = {
            "awaiting_signoff": [c for c in review["awaiting_signoff"] if c["id"] in target_claim_ids],
            "stale_verified": [c for c in review["stale_verified"] if c["id"] in target_claim_ids],
            "dirty_frontier": review["dirty_frontier"],
        }

    running_runs = [
        {
            "id": r.id,
            "title": r.title,
            "created_at": r.get("created_at"),
            "artifacts": r.artifacts,
        }
        for r in sorted(g.runs.values(), key=lambda x: x.id)
        if r.status == "running"
    ]
    running_analyses = [
        {
            "id": a.id,
            "kind": a.kind,
            "title": a.title,
            "subjects": a.subjects,
            "created_at": a.get("created_at"),
            "artifacts": a.artifacts,
        }
        for a in sorted(g.analyses.values(), key=lambda x: x.id)
        if a.status == "running"
    ]

    stuck = [i.as_dict() for i in issues if i.code == "C005"]
    unsynthesized = [i.as_dict() for i in issues if i.code in ("C003", "C004")]

    open_questions: list[dict[str, str]] = []
    for goal in target_goals:
        if goal.status != "active":
            continue
        for q in goal.open_questions:
            open_questions.append({"goal": goal.id, "question": q})

    claim_threads: list[dict[str, Any]] = []
    for c in target_claims:
        if c.status not in ("active", "supported"):
            continue
        con = c.refs("contradicts")
        qual = c.refs("qualifies")
        if c.status == "active" or con or qual:
            claim_threads.append({
                "id": c.id,
                "status": c.status,
                "verification": effective_verification(g, c),
                "title": c.title,
                "contradicts": con,
                "qualifies": qual,
            })

    by_code: dict[str, int] = {}
    for i in issues:
        by_code[i.code] = by_code.get(i.code, 0) + 1

    return {
        "goal_filter": gid,
        "health": {
            "total_issues": len(issues),
            "errors": sum(1 for i in issues if i.severity == ERROR),
            "warnings": sum(1 for i in issues if i.severity != ERROR),
            "by_code": dict(sorted(by_code.items())),
        },
        "active_work": {
            "running_runs": running_runs,
            "running_analyses": running_analyses,
            "stuck": stuck,
            "unsynthesized": unsynthesized,
        },
        "review": review,
        "open_threads": {
            "open_questions": open_questions,
            "claim_threads": claim_threads,
        },
        "ideate": ideate,
    }


# ---------------------------------------------------------------------------
# Evidence-Grounded Reports (`contour report`)
# ---------------------------------------------------------------------------


def _serialize_run_evidence(g: ContourGraph, run: ContourNode) -> dict[str, Any]:
    vs = get_fresh_vs_parent(g, run)
    ctrl = evidence_control_level(g, run)
    attached = [
        {"id": a.id, "kind": a.kind, "title": a.title, "tldr": a.tldr}
        for a in g.run_analyses(run.id)
        if a.is_usable_evidence
    ]
    return {
        "id": run.id,
        "type": "run",
        "title": run.title,
        "status": run.status,
        "bugged": run.bugged,
        "control_level": ctrl,
        "parent": run.parent,
        "gpu_hours": run.gpu_hours,
        "compute": run.compute,
        "code": run.code,
        "metrics": g.effective_run_metrics(run),
        "artifacts": run.artifacts,
        "tldr": run.tldr,
        "vs_parent": vs,
        "attached_analyses": attached,
    }


def _serialize_analysis_evidence(g: ContourGraph, ana: ContourNode) -> dict[str, Any]:
    ctrl = evidence_control_level(g, ana)
    return {
        "id": ana.id,
        "type": "analysis",
        "kind": ana.kind,
        "title": ana.title,
        "status": ana.status,
        "bugged": ana.bugged,
        "control_level": ctrl,
        "subjects": ana.subjects,
        "script": ana.script,
        "script_sha256": ana.script_sha256,
        "launch_cmd": ana.get("launch_cmd"),
        "code": ana.code,
        "metrics": ana.metrics,
        "artifacts": ana.artifacts,
        "tldr": ana.tldr,
    }


def build_report(
    g: ContourGraph,
    *,
    goal_id: str | None = None,
    claim_ids: list[str] | None = None,
    run_ids: list[str] | None = None,
    analysis_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build an ephemeral, evidence-grounded truth bundle for colleague write-ups."""
    preliminary = bool((run_ids or analysis_ids) and not (goal_id or claim_ids))

    included_claims: list[ContourNode] = []
    included_goals: list[ContourNode] = []
    included_runs: dict[str, ContourNode] = {}
    included_analyses: dict[str, ContourNode] = {}

    if preliminary:
        for rid in (run_ids or []):
            r_clean = strip_wiki(rid)
            if r_clean not in g.runs:
                raise ValueError(f"Unknown run: {r_clean!r}")
            included_runs[r_clean] = g.runs[r_clean]
        for aid in (analysis_ids or []):
            a_clean = strip_wiki(aid)
            if a_clean not in g.analyses:
                raise ValueError(f"Unknown analysis: {a_clean!r}")
            included_analyses[a_clean] = g.analyses[a_clean]
    else:
        if goal_id:
            gid = strip_wiki(goal_id)
            if gid not in g.goals:
                raise ValueError(f"Unknown goal: {gid!r}")
            included_goals.append(g.goals[gid])
            included_claims.extend(g.goal_claims(gid))
        if claim_ids:
            seen_c = {c.id for c in included_claims}
            for cid in claim_ids:
                c_clean = strip_wiki(cid)
                if c_clean not in g.claims:
                    raise ValueError(f"Unknown claim: {c_clean!r}")
                if c_clean not in seen_c:
                    included_claims.append(g.claims[c_clean])
                    seen_c.add(c_clean)
        if not goal_id and not claim_ids:
            included_goals = sorted(g.goals.values(), key=lambda x: x.id)
            included_claims = sorted(g.claims.values(), key=lambda x: x.id)

        for c in included_claims:
            for edge in EDGE_KEYS:
                for eid in c.refs(edge):
                    if eid in g.runs:
                        included_runs[eid] = g.runs[eid]
                    elif eid in g.analyses:
                        included_analyses[eid] = g.analyses[eid]
                        for sid in g.analyses[eid].subjects:
                            if sid in g.runs:
                                included_runs.setdefault(sid, g.runs[sid])
        for rid in (run_ids or []):
            r_clean = strip_wiki(rid)
            if r_clean not in g.runs:
                raise ValueError(f"Unknown run: {r_clean!r}")
            included_runs[r_clean] = g.runs[r_clean]
        for aid in (analysis_ids or []):
            a_clean = strip_wiki(aid)
            if a_clean not in g.analyses:
                raise ValueError(f"Unknown analysis: {a_clean!r}")
            included_analyses[a_clean] = g.analyses[a_clean]

    claims_payload: list[dict[str, Any]] = []
    verif_counts = {"human_verified": 0, "agent_audited": 0, "draft": 0}
    stale_claims: list[str] = []

    for c in included_claims:
        eff = effective_verification(g, c)
        stale = is_verification_stale(g, c)
        verif_counts[eff] = verif_counts.get(eff, 0) + 1
        if stale:
            stale_claims.append(c.id)

        edges_out: dict[str, list[dict[str, Any]]] = {}
        for edge in EDGE_KEYS:
            items: list[dict[str, Any]] = []
            for eid in c.refs(edge):
                if eid in g.runs:
                    items.append(_serialize_run_evidence(g, g.runs[eid]))
                elif eid in g.analyses:
                    items.append(_serialize_analysis_evidence(g, g.analyses[eid]))
            edges_out[edge] = items

        claims_payload.append({
            "id": c.id,
            "title": c.title,
            "status": c.status,
            "effective_verification": eff,
            "stored_verification": c.verification,
            "stale": stale,
            "verified_hash": c.get("verified_hash") or compute_verified_hash(g, c),
            "audit_note": c.get("audit_note"),
            "hypothesis": c.sections.get("Hypothesis & Mechanism", "").strip(),
            "synthesis": c.sections.get("Evidence Synthesis", "").strip(),
            "evidence": edges_out,
        })

    prelim_runs = [_serialize_run_evidence(g, r) for r in included_runs.values()] if preliminary else []
    prelim_analyses = [_serialize_analysis_evidence(g, a) for a in included_analyses.values()] if preliminary else []
    prelim_comparisons: list[dict[str, Any]] = []
    if preliminary and run_ids and len(run_ids) >= 2:
        ctrl_id = strip_wiki(run_ids[0])
        for tid in run_ids[1:]:
            prelim_comparisons.append(compare_runs(g, ctrl_id, strip_wiki(tid)))

    included_ids = (
        {gl.id for gl in included_goals}
        | {c.id for c in included_claims}
        | set(included_runs.keys())
        | set(included_analyses.keys())
    )
    all_issues = find_check_issues(g)
    scoped_issues = [
        i for i in all_issues
        if i.node in included_ids or bool(included_ids & set(i.related_nodes))
    ]

    dirty_evidence: list[str] = []
    exploratory_evidence: list[str] = []
    invalid_evidence: list[str] = []
    for r in included_runs.values():
        if bool(r.code.get("dirty")):
            dirty_evidence.append(r.id)
        lvl = evidence_control_level(g, r)
        if lvl == "exploratory":
            exploratory_evidence.append(r.id)
        elif lvl == "invalid":
            invalid_evidence.append(r.id)
    for a in included_analyses.values():
        if bool(a.code.get("dirty")):
            dirty_evidence.append(a.id)
        lvl = evidence_control_level(g, a)
        if lvl == "exploratory":
            exploratory_evidence.append(a.id)
        elif lvl == "invalid":
            invalid_evidence.append(a.id)

    warnings: list[str] = []
    if preliminary:
        warnings.append("PRELIMINARY — NO VERIFIED CLAIM: report covers raw runs/analyses only.")
    if verif_counts["draft"] > 0:
        warnings.append(f"{verif_counts['draft']} included claim(s) are unverified (`draft`).")
    if stale_claims:
        warnings.append(f"Stale verified claim(s) (C008): {', '.join(stale_claims)}.")
    if dirty_evidence:
        warnings.append(f"Uncommitted code (`code.dirty: true`) on: {', '.join(sorted(dirty_evidence))}.")
    if exploratory_evidence:
        warnings.append(f"Exploratory evidence included: {', '.join(sorted(exploratory_evidence))}.")
    if invalid_evidence:
        warnings.append(f"Invalid/quarantined evidence included: {', '.join(sorted(invalid_evidence))}.")
    blocking_issues = [i for i in scoped_issues if i.severity == ERROR or i.category == "epistemic"]
    if blocking_issues:
        warnings.append(f"{len(blocking_issues)} error/epistemic check issue(s) apply to included items.")

    strict_ok = (
        not preliminary
        and verif_counts["draft"] == 0
        and not stale_claims
        and not dirty_evidence
        and not exploratory_evidence
        and not invalid_evidence
        and not blocking_issues
    )
    mode = "preliminary" if preliminary else ("verified" if strict_ok else "mixed")

    prov_entries: list[dict[str, Any]] = []
    for c in included_claims:
        prov_entries.append({
            "id": c.id,
            "type": "claim",
            "status": c.status,
            "verification": effective_verification(g, c),
            "verified_hash": c.get("verified_hash"),
        })
    for r in sorted(included_runs.values(), key=lambda x: x.id):
        prov_entries.append({
            "id": r.id,
            "type": "run",
            "status": r.status,
            "commit": r.code.get("commit"),
            "dirty": bool(r.code.get("dirty", False)),
            "config_path": r.config_path,
            "gpu_hours": r.gpu_hours,
        })
    for a in sorted(included_analyses.values(), key=lambda x: x.id):
        prov_entries.append({
            "id": a.id,
            "type": f"analysis:{a.kind}",
            "status": a.status,
            "commit": a.code.get("commit"),
            "dirty": bool(a.code.get("dirty", False)),
            "script": a.script,
            "script_sha256": a.script_sha256,
        })

    git_info = _get_git_info(g.root)

    return {
        "generated_at": _utc_now_iso(),
        "mode": mode,
        "integrity": {
            "strict_ok": strict_ok,
            "claims_total": len(included_claims),
            "human_verified": verif_counts["human_verified"],
            "agent_audited": verif_counts["agent_audited"],
            "draft": verif_counts["draft"],
            "stale_claims": stale_claims,
            "dirty_evidence": sorted(dirty_evidence),
            "exploratory_evidence": sorted(exploratory_evidence),
            "invalid_evidence": sorted(invalid_evidence),
            "open_issues": [i.as_dict() for i in scoped_issues],
            "warnings": warnings,
        },
        "goals": [
            {
                "id": gl.id,
                "title": gl.title,
                "status": gl.status,
                "tldr": gl.tldr,
                "open_questions": gl.open_questions,
            }
            for gl in included_goals
        ],
        "claims": claims_payload,
        "preliminary_runs": prelim_runs,
        "preliminary_analyses": prelim_analyses,
        "preliminary_comparisons": prelim_comparisons,
        "provenance": {
            "repo_commit": git_info.get("commit"),
            "repo_dirty": bool(git_info.get("dirty", False)),
            "entries": prov_entries,
        },
    }


def _collect_artifact_paths(packet: dict[str, Any]) -> list[dict[str, Any]]:
    arts: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(entries: list[dict[str, Any]]) -> None:
        for a in entries:
            p = str(a.get("path") or "")
            if p and p not in seen:
                seen.add(p)
                arts.append(a)

    for c in packet.get("claims") or []:
        for edge in EDGE_KEYS:
            for item in (c.get("evidence") or {}).get(edge) or []:
                _add(item.get("artifacts") or [])
    for item in packet.get("preliminary_runs") or []:
        _add(item.get("artifacts") or [])
    for item in packet.get("preliminary_analyses") or []:
        _add(item.get("artifacts") or [])
    return arts


def render_report_markdown(
    packet: dict[str, Any],
    *,
    asset_rewrites: dict[str, str] | None = None,
) -> str:
    """Render a report truth bundle into clean, colleague-ready Markdown."""
    rewrites = asset_rewrites or {}
    integ = packet["integrity"]
    lines: list[str] = []

    if packet["mode"] == "preliminary":
        lines.append("# Contour Research Report [PRELIMINARY — NO VERIFIED CLAIM]")
    else:
        lines.append("# Contour Research Report")
    lines.append("")

    lines.append("## Integrity Banner")
    if packet["mode"] == "preliminary":
        lines.append("- **Status:** `[PRELIMINARY — NO VERIFIED CLAIM]`")
    else:
        lines.append(
            f"- **Claims:** {integ['claims_total']} total "
            f"({integ['human_verified']} `human_verified`, "
            f"{integ['agent_audited']} `agent_audited`, "
            f"{integ['draft']} `draft`)"
        )
    lines.append(
        f"- **Evidence hygiene:** {len(integ['stale_claims'])} stale | "
        f"{len(integ['dirty_evidence'])} dirty | "
        f"{len(integ['exploratory_evidence'])} exploratory | "
        f"{len(integ['open_issues'])} check issue(s)"
    )
    if integ["warnings"]:
        lines.append("- **Warnings:**")
        for w in integ["warnings"]:
            lines.append(f"  - {w}")
    lines.append("")

    for gl in packet.get("goals") or []:
        lines.append(f"## Goal: {gl['title']} (`{gl['id']}` — {gl['status']})")
        if gl.get("tldr"):
            lines.append(f"> {gl['tldr']}")
        if gl.get("open_questions"):
            lines.append("\n**Open Questions:**")
            for q in gl["open_questions"]:
                lines.append(f"- {q}")
        lines.append("")

    for c in packet.get("claims") or []:
        badge = f"`{c['status']}` | `{c['effective_verification']}`"
        if c["stale"]:
            badge += " | `STALE`"
        lines.append(f"## Claim: {c['title']} (`{c['id']}`)")
        lines.append(f"**Status:** {badge}")
        if c.get("audit_note"):
            lines.append(f"**Audit Note:** {c['audit_note']}")
        lines.append("")
        if c.get("hypothesis"):
            lines.append(f"**Hypothesis:** {c['hypothesis']}\n")
        if c.get("synthesis"):
            lines.append("**Evidence Synthesis & Caveats:**")
            lines.append(c["synthesis"])
            lines.append("")

        rows: list[str] = []
        for edge in EDGE_KEYS:
            for item in (c.get("evidence") or {}).get(edge) or []:
                if item["type"] == "run":
                    vs = item.get("vs_parent")
                    delta_parts = []
                    if vs:
                        for mk, d in (vs.get("deltas") or {}).items():
                            sig = f" [{d['signal']}]" if d.get("signal") else ""
                            delta_parts.append(f"{mk}: {d['delta']:+.4g}{sig}")
                    delta_s = "; ".join(delta_parts) if delta_parts else ", ".join(
                        f"{k}={v}" for k, v in (item.get("metrics") or {}).items()
                    )
                    flags_s = f" (flags: {', '.join(vs['flags'])})" if vs and vs.get("flags") else ""
                    rows.append(
                        f"| `{edge}` | `{item['id']}` | run (`{item['control_level']}`) | "
                        f"{delta_s}{flags_s} | {item.get('gpu_hours') or '-'} |"
                    )
                else:
                    m_s = ", ".join(f"{k}={v}" for k, v in (item.get("metrics") or {}).items()) or item.get("tldr", "")
                    rows.append(
                        f"| `{edge}` | `{item['id']}` | {item['kind']} (`{item['control_level']}`) | "
                        f"{m_s} | - |"
                    )
        if rows:
            lines.append("| Relation | ID | Type / Control | Outcome / Summary | GPU hr |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            lines.extend(rows)
            lines.append("")

    if packet["mode"] == "preliminary":
        if packet.get("preliminary_runs"):
            lines.append("## Runs")
            lines.append("| Run ID | Title | Control | GPU hr | Metrics |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for r in packet["preliminary_runs"]:
                ms = ", ".join(f"{k}={v}" for k, v in (r.get("metrics") or {}).items())
                lines.append(
                    f"| `{r['id']}` | {r['title']} | `{r['control_level']}` | {r.get('gpu_hours') or '-'} | {ms} |"
                )
            lines.append("")
        if packet.get("preliminary_comparisons"):
            lines.append("## Comparisons")
            for cmp in packet["preliminary_comparisons"]:
                flags_s = f" | flags: {', '.join(cmp['flags'])}" if cmp.get("flags") else ""
                lines.append(
                    f"### `{cmp['control']}` -> `{cmp['treatment']}` "
                    f"(`{cmp['control_level']}` / `{cmp['verdict']}`{flags_s})"
                )
                for mk, d in (cmp.get("deltas") or {}).items():
                    sig = f" [{d['signal']}]" if d.get("signal") else ""
                    lines.append(f"- `{mk}`: {d['baseline']} -> {d['value']} ({d['delta']:+.4g}){sig}")
                lines.append("")
        if packet.get("preliminary_analyses"):
            lines.append("## Analyses")
            for a in packet["preliminary_analyses"]:
                lines.append(f"- **`{a['id']}`** ({a['kind']}, `{a['control_level']}`): {a['title']} — {a['tldr']}")
            lines.append("")

    arts = _collect_artifact_paths(packet)
    if arts:
        lines.append("## Artifacts")
        for a in arts:
            orig = str(a["path"])
            target = rewrites.get(orig, orig)
            kind = f" ({a['kind']})" if a.get("kind") else ""
            note = f" — {a['note']}" if a.get("note") else ""
            if target.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".webp")):
                lines.append(f"- ![{a.get('note') or Path(target).name}]({target}){kind}{note}")
            else:
                lines.append(f"- [{Path(target).name}]({target}){kind}{note}")
        lines.append("")

    prov = packet["provenance"]
    lines.append("## Provenance Appendix")
    lines.append(
        f"- **Generated:** `{packet['generated_at']}` | "
        f"**Repo commit:** `{prov.get('repo_commit') or 'N/A'}` "
        f"(dirty={prov.get('repo_dirty')})"
    )
    if prov["entries"]:
        lines.append("")
        lines.append("| ID | Type | Status | Commit / Hash | Details |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for e in prov["entries"]:
            if e["type"] == "claim":
                c_hash = f"`{e.get('verified_hash') or '-'}`"
                detail = f"verification={e.get('verification')}"
            elif e["type"] == "run":
                c_hash = f"`{e.get('commit') or '-'}`" + (" (dirty)" if e.get("dirty") else "")
                detail = f"config={e.get('config_path') or 'inline'}, gpu_h={e.get('gpu_hours') or '-'}"
            else:
                c_hash = f"`{e.get('commit') or '-'}`" + (" (dirty)" if e.get("dirty") else "")
                detail = f"script={e.get('script') or '-'} (sha={e.get('script_sha256') or '-'})"
            lines.append(f"| `{e['id']}` | {e['type']} | {e['status']} | {c_hash} | {detail} |")
    lines.append("")

    return "\n".join(lines)


def export_report(
    g: ContourGraph,
    packet: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Export `report.md` and copy any referenced local artifacts into `<out_dir>/assets/`."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"

    rewrites: dict[str, str] = {}
    copied: list[str] = []
    missing: list[str] = []
    used_names: set[str] = set()

    for art in _collect_artifact_paths(packet):
        orig = str(art.get("path") or "")
        if not orig or is_remote_uri(orig):
            continue
        p = Path(orig)
        src: Path | None = None
        for cand in ([p] if p.is_absolute() else [g.root / p, g.root.parent / p]):
            if cand.is_file():
                src = cand
                break
        if src is None:
            missing.append(orig)
            continue
        assets_dir.mkdir(parents=True, exist_ok=True)
        name = src.name
        stem, suffix = src.stem, src.suffix
        idx = 1
        while name in used_names:
            name = f"{stem}_{idx}{suffix}"
            idx += 1
        used_names.add(name)
        dest = assets_dir / name
        shutil.copy2(src, dest)
        rewrites[orig] = f"assets/{name}"
        copied.append(str(dest))

    md_text = render_report_markdown(packet, asset_rewrites=rewrites)
    report_path = out_dir / "report.md"
    report_path.write_text(md_text, encoding="utf-8")
    return {
        "report_path": str(report_path),
        "copied_assets": copied,
        "missing_assets": missing,
    }
