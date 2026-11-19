"""Contour CLI: Research state integrity, daily workflows, introspection, and ideation."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from .check import (
    ERROR,
    compute_frontier,
    compute_verified_hash,
    effective_verification,
    evidence_control_level,
    find_check_issues,
    find_review_items,
    frontier_metrics,
    issues_blocking_verify,
)
from .compare import compare_runs, get_fresh_vs_parent, recommend_relation
from .config import CONFIG_FILE, DEFAULT_CONFIG, SUBDIR_CANDIDATES, ConfigError, find_root
from .graph import ANALYSIS_KINDS, EDGE_KEYS, ContourGraph, ContourNode
from .ideate import generate_ideation
from .mdio import as_wikilink, parse_markdown_file, strip_wiki, write_markdown_file
from .report import build_brief, build_report, export_report, render_report_markdown

GOAL_BODY_TEMPLATE = (
    "## Objective & Success Criteria\n"
    "- TODO: State the overarching research problem and quantitative success criteria.\n\n"
    "## Current Synthesis\n"
    "- \n\n"
    "## Open Questions\n"
    "- \n"
)

CLAIM_BODY_TEMPLATE = (
    "## Hypothesis & Mechanism\n"
    "- TODO: State the falsifiable hypothesis and conceptual mechanism.\n\n"
    "## Evidence Synthesis\n"
    "- **Holds:** \n"
    "- **Breaks:** \n"
    "- **Untested / Caveats:** \n"
)

_GITIGNORE_ENTRIES = (
    "analyses/scripts/",
    "analyses/artifacts/",
)


def _emit_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _bundled_skill_markdown() -> str:
    pkg_root = Path(__file__).resolve().parent.parent.parent
    for rel in (
        pkg_root / "skills" / "contour-research" / "SKILL.md",
        pkg_root / ".agent" / "skills" / "contour-research" / "SKILL.md",
    ):
        if rel.exists():
            return rel.read_text(encoding="utf-8")
    return (
        "---\n"
        "name: contour-research\n"
        "description: Navigate, brief, verify, analyze, report, and ideate over the Contour ML research state.\n"
        "---\n\n"
        "# Contour: Epistemic ML Research Copilot Skill\n\n"
        "Run `contour brief --json`, `contour status --json`, `contour runs --tree`, "
        "`contour analyses --json`, `contour check --json`, `contour report --json`, and `contour ideate --json`.\n"
    )


def _primary_delta_summary(g: ContourGraph, vs: dict[str, Any] | None) -> str:
    if not vs:
        return "-"
    deltas = vs.get("deltas") or {}
    for k, d in deltas.items():
        if g.config.is_primary_metric(k):
            return f"{float(d['delta']):+.3g} ({k.rsplit('/', 1)[-1]})"
    for k, d in deltas.items():
        if g.config.metric_kind(k) == "quality":
            return f"{float(d['delta']):+.3g} ({k.rsplit('/', 1)[-1]})"
    return "-"


def _ensure_gitignore(root: Path) -> bool:
    """Idempotently add gitignored local analysis directories to `<root>/.gitignore`."""
    gi = root / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    existing_lines = {ln.strip() for ln in existing.splitlines()}
    missing = [e for e in _GITIGNORE_ENTRIES if e not in existing_lines]
    if not missing:
        return False
    buf = existing
    if buf and not buf.endswith("\n"):
        buf += "\n"
    if not existing:
        buf += "# Local ad-hoc analysis scripts & artifacts (untracked by default)\n"
    for entry in missing:
        buf += f"{entry}\n"
    gi.write_text(buf, encoding="utf-8")
    return True


def _print_diff_details(vs: dict[str, Any], indent: str = "  ") -> None:
    if vs.get("flags"):
        print(f"{indent}flags:        {vs['flags']}")
    blocks = vs.get("config_blocks") or {}
    cdiff = vs.get("config_diff") or {}
    if blocks:
        for blk, info in blocks.items():
            swap = (
                f" ({info['baseline']} -> {info['value']})"
                if "baseline" in info and "value" in info
                else ""
            )
            print(f"{indent}config_block: {blk}.*{swap} [{info['changed_keys']} keys]")
        block_keys = {k for info in blocks.values() for k in info.get("keys", [])}
        rem = {k: v for k, v in cdiff.items() if k not in block_keys}
        if rem:
            print(f"{indent}config_diff:  {rem}")
    elif cdiff:
        print(f"{indent}config_diff:  {cdiff}")
    code_diff = vs.get("code_diff") or {}
    if not code_diff.get("same_commit", True) and (
        code_diff.get("baseline_commit") or code_diff.get("value_commit")
    ):
        print(
            f"{indent}code_diff:    {code_diff.get('baseline_commit')} -> {code_diff.get('value_commit')}"
        )
    if vs.get("compute_diff"):
        print(f"{indent}compute_diff: {vs['compute_diff']}")
    if vs.get("eval_provenance"):
        print(f"{indent}eval_sources: {vs['eval_provenance']}")
    for k, d in (vs.get("deltas") or {}).items():
        sig = f" [{d['signal']}]" if d.get("signal") else ""
        rel = f" ({d['rel_delta']*100:+.2f}%)" if "rel_delta" in d else ""
        print(f"{indent}- {k}: {d['baseline']} -> {d['value']} ({d['delta']:+.4g}){rel}{sig}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_init(target: Path | None, with_skill: bool = False, as_json: bool = False) -> None:
    if target is None:
        cwd = Path.cwd().resolve()
        if (cwd / CONFIG_FILE).exists() or cwd.name in SUBDIR_CANDIDATES:
            resolved = cwd
        else:
            resolved = cwd / "contour"
    else:
        resolved = target.resolve()

    resolved.mkdir(parents=True, exist_ok=True)
    cfg_path = resolved / CONFIG_FILE
    created_cfg = False
    if not cfg_path.exists():
        cfg_path.write_text(
            yaml.safe_dump(copy.deepcopy(DEFAULT_CONFIG), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        created_cfg = True

    for sub in (
        "goals",
        "claims",
        "runs",
        "runs/configs",
        "analyses",
        "analyses/scripts",
        "analyses/artifacts",
    ):
        (resolved / sub).mkdir(parents=True, exist_ok=True)

    updated_gitignore = _ensure_gitignore(resolved)

    skill_path: Path | None = None
    if with_skill:
        repo_base = resolved.parent if resolved.name in SUBDIR_CANDIDATES else resolved
        skill_path = repo_base / ".agent" / "skills" / "contour-research" / "SKILL.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(_bundled_skill_markdown(), encoding="utf-8")

    if as_json:
        _emit_json({
            "root": str(resolved),
            "created_config": created_cfg,
            "updated_gitignore": updated_gitignore,
            **({"skill_path": str(skill_path)} if skill_path else {}),
        })
        return
    status = "Created" if created_cfg else "Initialized"
    print(f"{status} Contour root at {resolved}")
    if skill_path:
        print(f"Installed agent skill at {skill_path}")


def cmd_new(
    g: ContourGraph,
    node_type: str,
    slug: str,
    title: str,
    *,
    kind: str = "analysis",
    as_json: bool = False,
) -> None:
    today = dt.date.today().isoformat()
    date_str = today.replace("-", "")
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")

    if node_type == "goal":
        node_id = slug if slug.startswith("goal_") else f"goal_{slug}"
        path = g.root / "goals" / f"{node_id}.md"
        fm: dict[str, Any] = {
            "title": title,
            "status": "active",
        }
        body = GOAL_BODY_TEMPLATE
    elif node_type == "claim":
        node_id = slug if slug.startswith("clm_") else f"clm_{slug}"
        path = g.root / "claims" / f"{node_id}.md"
        fm = {
            "title": title,
            "status": "active",
            "verification": "draft",
            "goals": [],
            "supports": [],
            "contradicts": [],
            "qualifies": [],
        }
        body = CLAIM_BODY_TEMPLATE
    elif node_type == "run":
        node_id = slug if slug.startswith("run_") else f"run_{date_str}_{slug}"
        path = g.root / "runs" / f"{node_id}.md"
        fm = {
            "title": title,
            "status": "final",
            "created_at": now_iso,
            "finished_at": now_iso,
            "code": {"commit": None, "dirty": False},
            "metrics": {},
        }
        body = "## Key Takeaway (TL;DR)\n- TODO: Summarize key outcome.\n\n## Observations & Anomalies\n- \n"
    elif node_type == "analysis":
        if kind not in ANALYSIS_KINDS:
            raise SystemExit(f"Invalid analysis kind: {kind!r} (expected one of {ANALYSIS_KINDS})")
        node_id = slug if slug.startswith("ana_") else f"ana_{date_str}_{slug}"
        path = g.root / "analyses" / f"{node_id}.md"
        fm = {
            "kind": kind,
            "title": title,
            "status": "final",
            "created_at": now_iso,
            "finished_at": now_iso,
            "subjects": [],
            "code": {"commit": None, "dirty": False},
            "metrics": {},
        }
        body = "## Key Takeaway (TL;DR)\n- TODO: Summarize key finding.\n\n## Findings & Observations\n- \n"
    else:
        raise SystemExit(f"Unknown node type: {node_type}")

    if path.exists():
        raise SystemExit(f"File already exists: {path}")
    write_markdown_file(path, fm, body)
    rel = str(path.relative_to(g.root))
    if as_json:
        _emit_json({"id": node_id, "path": rel})
    else:
        print(f"Created {rel}")


def cmd_link(
    g: ContourGraph,
    evidence_id: str,
    claim_id: str,
    relation: str | None = None,
    as_json: bool = False,
) -> None:
    eid = strip_wiki(evidence_id)
    cid = strip_wiki(claim_id)
    ev = g.get_evidence(eid)
    if not ev:
        raise SystemExit(f"Error: Unknown run or analysis {eid!r}")
    claim = g.claims.get(cid)
    if not claim:
        raise SystemExit(f"Error: Unknown claim {cid!r}")

    fresh_vs = get_fresh_vs_parent(g, ev) if ev.type == "run" else None
    if relation is None:
        if ev.type == "run" and fresh_vs is not None:
            relation = recommend_relation(fresh_vs["control_level"], fresh_vs["verdict"])
        else:
            relation = "qualifies"
    if relation not in EDGE_KEYS:
        raise SystemExit(f"Error: Invalid relation {relation!r} (expected one of {EDGE_KEYS})")

    fm, body = parse_markdown_file(claim.path)
    for edge in EDGE_KEYS:
        existing = [strip_wiki(x) for x in (fm.get(edge) or [])]
        if edge == relation:
            if eid not in existing:
                existing.append(eid)
        else:
            existing = [x for x in existing if x != eid]
        fm[edge] = [as_wikilink(x) for x in existing]

    write_markdown_file(claim.path, fm, body)
    ctrl = evidence_control_level(g, ev)
    if as_json:
        _emit_json({
            "claim": cid,
            "evidence": eid,
            "run": eid if ev.type == "run" else None,
            "type": ev.type,
            "relation": relation,
            "control_level": ctrl,
            "vs_parent": fresh_vs,
        })
        return
    if ev.type == "run" and fresh_vs:
        badge = f" [{fresh_vs['control_level']}/{fresh_vs['verdict']}]"
    elif ev.type == "analysis":
        badge = f" [{ev.kind} | {ctrl}]"
    else:
        badge = f" [{ctrl} | no parent]"
    print(f"Linked {eid} -> {cid} under `{relation}`{badge}")


def cmd_check(g: ContourGraph, strict: bool = False, as_json: bool = False) -> None:
    issues = find_check_issues(g)
    if as_json:
        _emit_json({
            "count": len(issues),
            "issues": [i.as_dict() for i in issues],
        })
    else:
        print("=== CONTOUR MECHANICAL CHECK ===")
        if not issues:
            print("All goals, claims, runs, and analyses pass mechanical verification!")
        else:
            for item in issues:
                print(f" - [{item.code}/{item.severity.upper()}/{item.category}] ({item.node}) {item.message}")
                print(f"     remediation: {item.remediation}")

    has_error = any(i.severity == ERROR for i in issues)
    if has_error or (strict and bool(issues)):
        raise SystemExit(1)


def cmd_verify(
    g: ContourGraph,
    claim_id: str,
    mode: str,  # "agent" | "human"
    note: str | None = None,
    as_json: bool = False,
) -> None:
    cid = strip_wiki(claim_id)
    claim = g.claims.get(cid)
    if not claim:
        raise SystemExit(f"Error: Unknown claim {cid!r}")

    blocking = issues_blocking_verify(g, cid)
    if blocking:
        if as_json:
            _emit_json({
                "verified": False,
                "claim": cid,
                "blocking_issues": [b.as_dict() for b in blocking],
            })
        else:
            print(f"Cannot verify {cid}; resolve blocking issues first:", file=sys.stderr)
            for b in blocking:
                print(f"  - [{b.code}/{b.severity.upper()}] {b.message} -> {b.remediation}", file=sys.stderr)
        raise SystemExit(1)

    target_verif = "agent_audited" if mode == "agent" else "human_verified"
    fm, body = parse_markdown_file(claim.path)
    fm["verification"] = target_verif
    if note is not None:
        fm["audit_note"] = note
    claim.frontmatter = fm
    vhash = compute_verified_hash(g, claim)
    fm["verified_hash"] = vhash
    write_markdown_file(claim.path, fm, body)

    if as_json:
        _emit_json({
            "verified": True,
            "claim": cid,
            "verification": target_verif,
            "verified_hash": vhash,
            "audit_note": fm.get("audit_note"),
        })
        return
    print(f"Verified {cid} -> {target_verif} (verified_hash={vhash})")


def cmd_review(g: ContourGraph, as_json: bool = False) -> None:
    items = find_review_items(g)
    if as_json:
        _emit_json(items)
        return

    print("=== CONTOUR HUMAN REVIEW INBOX ===")
    awaiting = items["awaiting_signoff"]
    stale = items["stale_verified"]
    dirty = items["dirty_frontier"]

    print(f"\n[1. Claims Awaiting Human Sign-Off] ({len(awaiting)})")
    for c in awaiting:
        print(f"  * {c['id']} [{c['status']}]: {c['title']}")
        print(f"    +supports={c['supports']} | -contradicts={c['contradicts']} | ~qualifies={c['qualifies']}")
        if c.get("audit_note"):
            print(f"    audit_note: {c['audit_note']}")

    print(f"\n[2. Verified Claims Gone Stale] ({len(stale)})")
    for c in stale:
        flag = " [CONTRADICTED]" if c["has_contradicts"] else ""
        print(f"  * {c['id']} [{c['status']} | was {c['stored_verification']}]{flag}: {c['title']}")

    print(f"\n[3. Frontier Runs with Uncommitted Code (`code.dirty: true`)] ({len(dirty)})")
    for r in dirty:
        print(f"  * {r['id']}: {r['metric']}={r['value']} @ {r['gpu_hours']}h (commit={r['commit']})")

    if not (awaiting or stale or dirty):
        print("\nNothing in the human review queue.")


def cmd_status(g: ContourGraph, as_json: bool = False) -> None:
    covered_claims: set[str] = set()
    goals_out: list[dict[str, Any]] = []
    for goal in g.goals.values():
        g_claims = g.goal_claims(goal.id)
        claim_list = []
        for c in g_claims:
            covered_claims.add(c.id)
            edges = g.claim_edges(c)
            claim_list.append({
                "id": c.id,
                "status": c.status,
                "verification": effective_verification(g, c),
                "stored_verification": c.verification,
                "title": c.title,
                "counts": {k: len(v) for k, v in edges.items()},
            })
        goals_out.append({
            "id": goal.id,
            "status": goal.status,
            "title": goal.title,
            "tldr": goal.tldr,
            "open_questions": goal.open_questions,
            "claims": claim_list,
        })

    unassigned = [
        {
            "id": c.id,
            "status": c.status,
            "verification": effective_verification(g, c),
            "title": c.title,
            "counts": {k: len(v) for k, v in g.claim_edges(c).items()},
        }
        for cid, c in sorted(g.claims.items())
        if cid not in covered_claims
    ]
    issues = find_check_issues(g)
    rev = find_review_items(g)
    review_count = len(rev["awaiting_signoff"]) + len(rev["stale_verified"]) + len(rev["dirty_frontier"])

    if as_json:
        _emit_json({
            "counts": {
                "goals": len(g.goals),
                "claims": len(g.claims),
                "runs": len(g.runs),
                "usable_runs": len(g.usable_runs()),
                "analyses": len(g.analyses),
                "usable_analyses": len(g.usable_analyses()),
                "check_issues": len(issues),
                "review_items": review_count,
            },
            "goals": goals_out,
            "unassigned_claims": unassigned,
        })
        return

    print("=== CONTOUR RESEARCH STATUS ===")
    print(
        f"State: {len(g.goals)} Goal(s) | {len(g.claims)} Claim(s) | "
        f"{len(g.runs)} Run(s) ({len(g.usable_runs())} usable) | "
        f"{len(g.analyses)} Analysis/Eval(s) ({len(g.usable_analyses())} usable) | "
        f"check={len(issues)} | review={review_count}\n"
    )
    for goal in goals_out:
        print(f"[GOAL] {goal['id']} ({goal['status'].upper()}): {goal['title']}")
        if goal["tldr"]:
            print(f"  Objective: {goal['tldr']}")
        if goal["open_questions"]:
            print(f"  Open questions: {len(goal['open_questions'])}")
        if not goal["claims"]:
            print("  (No linked claims)")
        for c in goal["claims"]:
            cnt = c["counts"]
            print(f"  └── [CLAIM] {c['id']} [{c['status']} | {c['verification']}]: {c['title']}")
            print(
                f"        Evidence: +{cnt['supports']} supports | "
                f"-{cnt['contradicts']} contradicts | ~{cnt['qualifies']} qualifies"
            )
        print()

    if unassigned:
        print("[UNASSIGNED CLAIMS]")
        for c in unassigned:
            print(f"  - {c['id']} [{c['status']} | {c['verification']}]: {c['title']}")
        print()


def cmd_show(g: ContourGraph, node_id: str, as_json: bool = False) -> None:
    nid = strip_wiki(node_id)
    node = g.all_nodes.get(nid)
    if not node:
        raise SystemExit(f"Node {nid!r} not found in Contour graph.")

    vs_parent = get_fresh_vs_parent(g, node) if node.type == "run" else None

    if as_json:
        fm_out = dict(node.frontmatter)
        extra: dict[str, Any] = {}
        if node.type == "run":
            fm_out["vs_parent"] = vs_parent
            extra["resolved_config"] = node.config
            extra["control_level"] = evidence_control_level(g, node)
            extra["attached_analyses"] = [
                {"id": a.id, "kind": a.kind, "title": a.title, "control_level": evidence_control_level(g, a), "tldr": a.tldr}
                for a in g.run_analyses(node.id)
            ]
        elif node.type == "analysis":
            extra["control_level"] = evidence_control_level(g, node)
        elif node.type == "claim":
            extra["effective_verification"] = effective_verification(g, node)
            extra["verified_hash_expected"] = compute_verified_hash(g, node)
        elif node.type == "goal":
            extra["claims"] = [c.id for c in g.goal_claims(node.id)]
            extra["open_questions"] = node.open_questions
        _emit_json({
            "id": node.id,
            "type": node.type,
            "title": node.title,
            "status": node.status,
            "path": str(node.path.relative_to(g.root)),
            "frontmatter": fm_out,
            **extra,
            "body": node.body,
        })
        return

    print(f"=== [{node.type.upper()}] {node.id}: {node.title} ===")
    print(f"Path:   {node.path.relative_to(g.root)}")
    print(f"Status: {node.status}")

    if node.type == "run":
        code = node.code
        bug_s = " | BUGGED" if node.bugged else ""
        print(f"Control: {evidence_control_level(g, node)}{bug_s}")
        print(f"GPU hr:  {node.gpu_hours} | Params: {node.compute.get('params')} | Commit: {code.get('commit')} | Dirty: {code.get('dirty')}")
        if node.config_path:
            print(f"Config:  {node.config_path} ({len(node.config)} top-level keys)")
        elif node.config:
            print(f"Config:  {node.config}")
        if node.artifacts:
            print("Artifacts:")
            for a in node.artifacts:
                k = f" ({a['kind']})" if a.get("kind") else ""
                n = f" — {a['note']}" if a.get("note") else ""
                print(f"  - {a['path']}{k}{n}")

        if vs_parent is not None:
            bid = strip_wiki(str(vs_parent.get("baseline", "")))
            print(
                f"\nvs_parent ({bid}) [control={vs_parent['control_level']} | "
                f"verdict={vs_parent['verdict']} | hash={vs_parent['inputs_hash']}]:"
            )
            _print_diff_details(vs_parent)
        elif node.metrics:
            print("\nMetrics:")
            for k, v in node.metrics.items():
                print(f"  - {k}: {v}")

        attached = g.run_analyses(node.id)
        if attached:
            print(f"\nAttached Evals & Analyses ({len(attached)}):")
            for a in attached:
                ac = evidence_control_level(g, a)
                m_str = ", ".join(f"{k}={v}" for k, v in a.metrics.items())
                suffix = f" ({m_str})" if m_str else f": {a.tldr}"
                print(f"  - {a.id} [{a.kind} | {ac}]{suffix}")

    elif node.type == "analysis":
        code = node.code
        bug_s = " | BUGGED" if node.bugged else ""
        print(f"Kind:     {node.kind} | Control: {evidence_control_level(g, node)}{bug_s}")
        print(f"Subjects: {node.subjects}")
        print(f"Commit:   {code.get('commit')} | Dirty: {code.get('dirty')}")
        if node.script:
            print(f"Script:   {node.script} (sha256={node.script_sha256 or 'N/A'})")
        if node.get("launch_cmd"):
            print(f"Command:  {node.get('launch_cmd')}")
        if node.metrics:
            print("\nMetrics:")
            for k, v in node.metrics.items():
                print(f"  - {k}: {v}")
        if node.artifacts:
            print("\nArtifacts:")
            for a in node.artifacts:
                k = f" ({a['kind']})" if a.get("kind") else ""
                n = f" — {a['note']}" if a.get("note") else ""
                print(f"  - {a['path']}{k}{n}")

    elif node.type == "claim":
        eff = effective_verification(g, node)
        print(f"Verification: {eff} (stored: {node.verification})")
        if node.get("audit_note"):
            print(f"Audit note:   {node.get('audit_note')}")
        for edge_type in EDGE_KEYS:
            e_ids = node.refs(edge_type)
            if e_ids:
                print(f"\n{edge_type.upper()} ({len(e_ids)}):")
                for eid in e_ids:
                    ev = g.get_evidence(eid)
                    if ev and ev.type == "run":
                        vs = get_fresh_vs_parent(g, ev)
                        if vs:
                            fl = f" flags={vs['flags']}" if vs.get("flags") else ""
                            badge = f" [{vs['control_level']}/{vs['verdict']}{fl}]"
                        else:
                            badge = f" [{evidence_control_level(g, ev)} | no parent]"
                        print(f"  - {eid}{badge}: {ev.tldr}")
                    elif ev and ev.type == "analysis":
                        ac = evidence_control_level(g, ev)
                        print(f"  - {eid} [{ev.kind} | {ac}]: {ev.tldr}")
                    else:
                        print(f"  - {eid} (missing evidence)")

    elif node.type == "goal":
        if node.open_questions:
            print(f"\nOPEN QUESTIONS ({len(node.open_questions)}):")
            for q in node.open_questions:
                print(f"  - {q}")
        g_claims = g.goal_claims(node.id)
        print(f"\nLINKED CLAIMS ({len(g_claims)}):")
        for c in g_claims:
            print(f"  - {c.id} [{c.status} | {effective_verification(g, c)}]: {c.title}")

    print("\n--- Notes ---")
    print(node.body)


def cmd_runs(
    g: ContourGraph,
    *,
    tree: bool = False,
    goal_id: str | None = None,
    claim_id: str | None = None,
    eval_scope: str | None = None,
    status: str | None = None,
    control: str | None = None,
    verdict: str | None = None,
    as_json: bool = False,
) -> None:
    runs = list(g.runs.values())

    allowed_by_claim: set[str] | None = None
    if goal_id:
        gid = strip_wiki(goal_id)
        g_claims = g.goal_claims(gid)
        allowed_by_claim = set()
        for c in g_claims:
            for edge in EDGE_KEYS:
                for eid in c.refs(edge):
                    if eid in g.runs:
                        allowed_by_claim.add(eid)
                    elif eid in g.analyses:
                        allowed_by_claim.update(g.analyses[eid].subjects)
        for r in runs:
            if gid in r.refs("motivated_by") or any(c.id in r.refs("motivated_by") for c in g_claims):
                allowed_by_claim.add(r.id)
    if claim_id:
        cid = strip_wiki(claim_id)
        cnode = g.claims.get(cid)
        c_set: set[str] = set()
        if cnode:
            for edge in EDGE_KEYS:
                for eid in cnode.refs(edge):
                    if eid in g.runs:
                        c_set.add(eid)
                    elif eid in g.analyses:
                        c_set.update(g.analyses[eid].subjects)
        for r in runs:
            if cid in r.refs("motivated_by"):
                c_set.add(r.id)
        allowed_by_claim = c_set if allowed_by_claim is None else (allowed_by_claim & c_set)

    filtered: list[tuple[ContourNode, dict[str, Any] | None]] = []
    for r in sorted(runs, key=lambda x: x.id):
        if allowed_by_claim is not None and r.id not in allowed_by_claim:
            continue
        if status and r.status != status:
            continue
        if eval_scope:
            scopes = {k.split("/", 1)[0] for k in g.effective_run_metrics(r) if "/" in k}
            if eval_scope not in scopes:
                continue
        vs = get_fresh_vs_parent(g, r)
        ctrl_lvl = vs["control_level"] if vs else evidence_control_level(g, r)
        verd = vs["verdict"] if vs else "-"
        if control and ctrl_lvl != control:
            continue
        if verdict and verd != verdict:
            continue
        filtered.append((r, vs))

    if as_json:
        _emit_json({
            "count": len(filtered),
            "runs": [
                {
                    "id": r.id,
                    "title": r.title,
                    "status": r.status,
                    "bugged": r.bugged,
                    "gpu_hours": r.gpu_hours,
                    "parent": r.parent,
                    "control_level": vs["control_level"] if vs else evidence_control_level(g, r),
                    "verdict": vs["verdict"] if vs else None,
                    "flags": vs.get("flags", []) if vs else [],
                    "primary_delta": _primary_delta_summary(g, vs),
                }
                for r, vs in filtered
            ],
        })
        return

    if not filtered:
        print("No matching runs found.")
        return

    if tree:
        print("=== CONTOUR RUN LINEAGE TREE ===")
        by_id = {r.id: (r, vs) for r, vs in filtered}
        children: dict[str, list[str]] = {}
        roots: list[str] = []
        for r, _ in filtered:
            p = r.parent
            if p and p in by_id:
                children.setdefault(p, []).append(r.id)
            else:
                roots.append(r.id)

        def _print_branch(rid: str, prefix: str = "", is_last: bool = True, is_root: bool = True) -> None:
            r, vs = by_id[rid]
            if vs:
                fl = f" !{','.join(vs['flags'])}" if vs.get("flags") else ""
                badge = f"[{vs['control_level']}/{vs['verdict']} {_primary_delta_summary(g, vs)}{fl}]"
            else:
                badge = f"[{r.status} | root]"
            conn = "" if is_root else ("└── " if is_last else "├── ")
            print(f"{prefix}{conn}{r.id} {badge} :: {r.title}")
            kids = children.get(rid, [])
            child_prefix = "" if is_root else (prefix + ("    " if is_last else "│   "))
            for idx, kid in enumerate(kids):
                _print_branch(kid, child_prefix, idx == len(kids) - 1, is_root=False)

        for root_id in roots:
            _print_branch(root_id)
        return

    print(f"=== CONTOUR RUNS ({len(filtered)}) ===")
    print(f"{'ID':<22} {'STATUS':<8} {'CONTROL':<11} {'VERDICT':<15} {'PRIMARY Δ':<20} {'TITLE'}")
    print("-" * 116)
    for r, vs in filtered:
        ctrl_s = vs["control_level"] if vs else evidence_control_level(g, r)
        verd_s = vs["verdict"] if vs else "-"
        pdelta = _primary_delta_summary(g, vs)
        print(f"{r.id:<22} {r.status:<8} {ctrl_s:<11} {verd_s:<15} {pdelta:<20} {r.title[:44]}")


def cmd_analyses(
    g: ContourGraph,
    *,
    kind: str | None = None,
    subject: str | None = None,
    claim_id: str | None = None,
    goal_id: str | None = None,
    status: str | None = None,
    control: str | None = None,
    as_json: bool = False,
) -> None:
    allowed: set[str] | None = None
    if goal_id:
        gid = strip_wiki(goal_id)
        g_claims = g.goal_claims(gid)
        allowed = set()
        for c in g_claims:
            for edge in EDGE_KEYS:
                allowed.update(eid for eid in c.refs(edge) if eid in g.analyses)
        for a in g.analyses.values():
            if gid in a.refs("motivated_by") or any(c.id in a.refs("motivated_by") for c in g_claims):
                allowed.add(a.id)
    if claim_id:
        cid = strip_wiki(claim_id)
        cnode = g.claims.get(cid)
        c_set: set[str] = set()
        if cnode:
            for edge in EDGE_KEYS:
                c_set.update(eid for eid in cnode.refs(edge) if eid in g.analyses)
        for a in g.analyses.values():
            if cid in a.refs("motivated_by"):
                c_set.add(a.id)
        allowed = c_set if allowed is None else (allowed & c_set)

    subj_clean = strip_wiki(subject) if subject else None
    filtered: list[tuple[ContourNode, str]] = []
    for a in sorted(g.analyses.values(), key=lambda x: x.id):
        if allowed is not None and a.id not in allowed:
            continue
        if kind and a.kind != kind:
            continue
        if subj_clean and subj_clean not in a.subjects:
            continue
        if status and a.status != status:
            continue
        ctrl = evidence_control_level(g, a)
        if control and ctrl != control:
            continue
        filtered.append((a, ctrl))

    if as_json:
        _emit_json({
            "count": len(filtered),
            "analyses": [
                {
                    "id": a.id,
                    "kind": a.kind,
                    "title": a.title,
                    "status": a.status,
                    "bugged": a.bugged,
                    "control_level": ctrl,
                    "subjects": a.subjects,
                    "script": a.script,
                    "script_sha256": a.script_sha256,
                    "launch_cmd": a.get("launch_cmd"),
                    "metrics": a.metrics,
                    "artifacts": a.artifacts,
                    "tldr": a.tldr,
                }
                for a, ctrl in filtered
            ],
        })
        return

    if not filtered:
        print("No matching analyses found.")
        return

    print(f"=== CONTOUR ANALYSES & EVALS ({len(filtered)}) ===")
    print(f"{'ID':<22} {'KIND':<9} {'STATUS':<8} {'CONTROL':<11} {'SUBJECTS':<28} {'TITLE'}")
    print("-" * 116)
    for a, ctrl in filtered:
        subj_s = ", ".join(a.subjects) if a.subjects else "-"
        print(f"{a.id:<22} {a.kind:<9} {a.status:<8} {ctrl:<11} {subj_s[:26]:<28} {a.title[:44]}")


def cmd_compare(g: ContourGraph, ctrl_id: str, treat_id: str, as_json: bool = False) -> None:
    try:
        out = compare_runs(g, strip_wiki(ctrl_id), strip_wiki(treat_id))
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if as_json:
        _emit_json(out)
        return

    print(
        f"=== COMPARE {out['control']} -> {out['treatment']} "
        f"[control={out['control_level']} | verdict={out['verdict']}] ==="
    )
    print(f"Recommended link: {out['recommended_link']}")
    _print_diff_details(out, indent="")


def cmd_frontier(
    g: ContourGraph,
    metric: str | None = None,
    as_json: bool = False,
) -> None:
    usable = g.usable_runs()
    metrics_to_show = frontier_metrics(g, usable, metric_filter=metric)
    tables = [compute_frontier(g, metric_filter=m) for m in (metrics_to_show or [metric])]

    def _serialize(pm: str | None, direction: str, runs: list[ContourNode], pareto_ids: set[str]) -> dict[str, Any]:
        return {
            "metric": pm,
            "direction": direction,
            "runs": [
                {
                    "id": r.id,
                    "title": r.title,
                    "gpu_hours": r.gpu_hours,
                    "value": g.effective_run_metrics(r).get(pm or ""),
                    "on_frontier": r.id in pareto_ids,
                    "dirty": bool(r.code.get("dirty", False)),
                }
                for r in runs
            ],
        }

    if as_json:
        payloads = [_serialize(pm, d, runs, pids) for pm, d, runs, pids in tables]
        out = dict(payloads[0])
        if len(payloads) > 1:
            out["additional_frontiers"] = payloads[1:]
        _emit_json(out)
        return

    printed_any = False
    for primary_metric, direction, runs, pareto_ids in tables:
        if not runs:
            continue
        if printed_any:
            print()
        printed_any = True
        print(f"=== PARETO FRONTIER (Metric: {primary_metric or 'N/A'} [{direction}] | Runs: {len(runs)}) ===")
        print(f"{'RUN ID':<22} {'GPU_HR':<8} {'METRIC':<10} {'FRONTIER':<10} {'TITLE'}")
        print("-" * 100)
        for r in runs:
            gpu_str = f"{r.gpu_hours:.1f}" if r.gpu_hours is not None else "-"
            val = g.effective_run_metrics(r).get(primary_metric or "", "-")
            mark = "  ▲" if r.id in pareto_ids else ""
            print(f"{r.id:<22} {gpu_str:<8} {str(val):<10} {mark:<10} {r.title[:50]}")

    if not printed_any:
        print("No matching usable runs found.")


def cmd_ideate(g: ContourGraph, goal_id: str | None = None, as_json: bool = False) -> None:
    out = generate_ideation(g, goal_id=goal_id)
    if as_json:
        _emit_json(out)
        return

    print("=== CONTOUR IDEATE: NEXT EXPERIMENT CANDIDATES ===\n")
    print(f"[Move 1: Investigate Exploratory / Flagged Wins] ({len(out['move_1_investigate_flagged'])})")
    for item in out["move_1_investigate_flagged"]:
        print(
            f"  * {item['run']} [{item['control_level']}] "
            f"(Δ={item['primary_delta']} | flags={item['flags']} | est={item['estimated_gpu_hours']}h): {item['suggestion']}"
        )

    print(f"\n[Move 2: Resolve Tensions & Tradeoffs] ({len(out['move_2_resolve_tensions'])})")
    for item in out["move_2_resolve_tensions"]:
        if item["kind"] == "claim_tension":
            print(f"  * {item['claim']} [{item['status']}]: contradicts={item['contradicts']} | qualifies={item['qualifies']} (est={item['estimated_gpu_hours']}h)")
        else:
            print(f"  * {item['run']} tradeoff (est={item['estimated_gpu_hours']}h): changed={item['changed_units']}")

    print(f"\n[Move 3: Combine & Compound Wins] ({len(out['move_3_combine_wins'])})")
    for item in out["move_3_combine_wins"]:
        cross = "CROSSES τ" if item["crosses_threshold"] else "sub-τ"
        print(
            f"  * {item['left']['id']} ({item['left']['verdict']}) + {item['right']['id']} ({item['right']['verdict']}) "
            f"-> {item['combined_keys']} | proj Δ={item['projected_primary_delta']:+.3g} [{cross}] (est={item['estimated_gpu_hours']}h)"
        )

    print(f"\n[Move 4: Strip Complexity (Occam's Razor)] ({len(out['move_4_strip_complexity'])})")
    for item in out["move_4_strip_complexity"]:
        strips = [a["strip"] for a in item["candidate_ablations"]]
        print(f"  * {item['run']} ({item['verdict']} | est={item['estimated_gpu_hours']}h) -> leave-one-out strips: {strips}")

    print(f"\n[Move 5: Open Questions & Unsupported Claims] ({len(out['move_5_open_questions'])})")
    for item in out["move_5_open_questions"]:
        if item["kind"] == "open_question":
            print(f"  ? ({item['goal']}) {item['question']}")
        else:
            print(f"  * Unsupported active claim {item['claim']}: {item['title']}")

    print(f"\n[Move 6: Diagnose Frontier Bottlenecks] ({len(out['move_6_diagnose_frontier'])})")
    for item in out["move_6_diagnose_frontier"]:
        obs = item["observations"][0] if item["observations"] else "See diagnostics"
        ana_s = f" | analyses={len(item['attached_analyses'])}" if item.get("attached_analyses") else ""
        print(
            f"  * {item['best_run']} ({item['metric']}={item['value']} @ {item['estimated_gpu_hours']}h{ana_s}): {obs}"
        )


def cmd_brief(g: ContourGraph, goal_id: str | None = None, as_json: bool = False) -> None:
    brief = build_brief(g, goal_id=goal_id)
    if as_json:
        _emit_json(brief)
        return

    h = brief["health"]
    aw = brief["active_work"]
    rev = brief["review"]
    ot = brief["open_threads"]
    print("=== CONTOUR MORNING BRIEF ===")
    print(f"Health: {h['total_issues']} issue(s) ({h['errors']} error, {h['warnings']} warn)")

    print("\n[1. Active & Unsynthesized Work]")
    if aw["running_runs"] or aw["running_analyses"]:
        for r in aw["running_runs"]:
            arts = ", ".join(a["path"] for a in r["artifacts"]) if r["artifacts"] else "no log artifact"
            print(f"  * RUNNING run {r['id']}: {r['title']} ({arts})")
        for a in aw["running_analyses"]:
            print(f"  * RUNNING {a['kind']} {a['id']}: {a['title']}")
    if aw["stuck"]:
        for s in aw["stuck"]:
            print(f"  * STUCK ({s['node']}): {s['message']}")
    if aw["unsynthesized"]:
        for u in aw["unsynthesized"]:
            print(f"  * [{u['code']}] ({u['node']}): {u['message']}")
    if not (aw["running_runs"] or aw["running_analyses"] or aw["stuck"] or aw["unsynthesized"]):
        print("  (None)")

    rev_total = len(rev["awaiting_signoff"]) + len(rev["stale_verified"]) + len(rev["dirty_frontier"])
    print(f"\n[2. Human Review Queue] ({rev_total})")
    for c in rev["awaiting_signoff"]:
        print(f"  * Sign-off: {c['id']} [{c['status']}]: {c['title']}")
    for c in rev["stale_verified"]:
        print(f"  * Stale:    {c['id']} [{c['status']} | was {c['stored_verification']}]: {c['title']}")
    for r in rev["dirty_frontier"]:
        print(f"  * Dirty frontier: {r['id']} ({r['metric']}={r['value']})")
    if rev_total == 0:
        print("  (Clear)")

    print("\n[3. Open Questions & Claim Threads]")
    for q in ot["open_questions"]:
        print(f"  ? ({q['goal']}) {q['question']}")
    for ct in ot["claim_threads"]:
        extras = []
        if ct["contradicts"]:
            extras.append(f"contradicts={len(ct['contradicts'])}")
        if ct["qualifies"]:
            extras.append(f"qualifies={len(ct['qualifies'])}")
        extra_s = f" ({', '.join(extras)})" if extras else ""
        print(f"  * {ct['id']} [{ct['status']} | {ct['verification']}]{extra_s}: {ct['title']}")
    if not (ot["open_questions"] or ot["claim_threads"]):
        print("  (None)")

    ide = brief["ideate"]
    counts = {k: len(v) for k, v in ide.items() if v}
    print("\n[4. Ideation Highlights]")
    if counts:
        print("  " + " | ".join(f"{k}={n}" for k, n in counts.items()))
    else:
        print("  (No pending ideation moves)")


def cmd_report(
    g: ContourGraph,
    *,
    goal_id: str | None = None,
    claim_ids: list[str] | None = None,
    run_ids: list[str] | None = None,
    analysis_ids: list[str] | None = None,
    out_dir: Path | None = None,
    strict: bool = False,
    as_json: bool = False,
) -> None:
    try:
        packet = build_report(
            g,
            goal_id=goal_id,
            claim_ids=claim_ids,
            run_ids=run_ids,
            analysis_ids=analysis_ids,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if strict and not packet["integrity"]["strict_ok"]:
        if as_json:
            _emit_json(packet)
        else:
            print("Report failed `--strict` verification gate:", file=sys.stderr)
            for w in packet["integrity"]["warnings"]:
                print(f"  - {w}", file=sys.stderr)
        raise SystemExit(1)

    export_info: dict[str, Any] | None = None
    if out_dir is not None:
        export_info = export_report(g, packet, out_dir)
        packet["export"] = export_info

    if as_json:
        _emit_json(packet)
        return

    if export_info is not None:
        print(f"Exported report to {export_info['report_path']}")
        if export_info["copied_assets"]:
            print(f"Copied {len(export_info['copied_assets'])} artifact(s) to assets/")
        if export_info["missing_assets"]:
            print(f"Warning: {len(export_info['missing_assets'])} local artifact(s) not found: {export_info['missing_assets']}")
        return

    print(render_report_markdown(packet))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=argparse.SUPPRESS, help="Path to contour root")
    common.add_argument("--json", dest="as_json", action="store_true", default=argparse.SUPPRESS, help="Emit JSON")

    parser = argparse.ArgumentParser(
        prog="contour",
        description="Contour: Epistemic ML research state, invariants, reporting, and ideation.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Path to contour root (with contour.yaml)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit machine-readable JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", parents=[common], help="Initialize a Contour root")
    p_init.add_argument("path", nargs="?", type=Path, default=None, help="Target directory (default: ./contour)")
    p_init.add_argument("--with-skill", action="store_true", help="Write .agent/skills/contour-research/SKILL.md")

    p_new = sub.add_parser("new", parents=[common], help="Create a new goal, claim, run, or analysis")
    p_new.add_argument("node_type", choices=["goal", "claim", "run", "analysis"])
    p_new.add_argument("slug", type=str)
    p_new.add_argument("--title", type=str, required=True)
    p_new.add_argument("--kind", choices=list(ANALYSIS_KINDS), default="analysis", help="For `analysis`: eval or analysis")

    p_link = sub.add_parser("link", parents=[common], help="Link a run or analysis onto a claim")
    p_link.add_argument("evidence_id", type=str)
    p_link.add_argument("claim_id", type=str)
    p_link.add_argument("--as", dest="relation", choices=list(EDGE_KEYS), default=None)

    p_check = sub.add_parser("check", parents=[common], help="Evaluate Tier-1 & Tier-2 mechanical invariants")
    p_check.add_argument("--strict", action="store_true", help="Exit non-zero on WARN")

    p_verify = sub.add_parser("verify", parents=[common], help="Record agentic or human verification on a claim")
    p_verify.add_argument("claim_id", type=str)
    grp = p_verify.add_mutually_exclusive_group(required=True)
    grp.add_argument("--agent", action="store_true")
    grp.add_argument("--human", action="store_true")
    p_verify.add_argument("--note", type=str, default=None)

    sub.add_parser("review", parents=[common], help="Tier-3 selective human verification inbox")
    sub.add_parser("status", parents=[common], help="Overview of Goals -> Claims -> Evidence counts")

    p_show = sub.add_parser("show", parents=[common], aliases=["inspect"], help="Inspect a Goal, Claim, Run, or Analysis")
    p_show.add_argument("node_id", type=str)

    p_runs = sub.add_parser("runs", parents=[common], help="Filter and tabulate training runs or render lineage tree")
    p_runs.add_argument("--tree", action="store_true")
    p_runs.add_argument("--goal", dest="goal_id", type=str, default=None)
    p_runs.add_argument("--claim", dest="claim_id", type=str, default=None)
    p_runs.add_argument("--eval", dest="eval_scope", type=str, default=None)
    p_runs.add_argument("--status", type=str, default=None)
    p_runs.add_argument("--control", type=str, default=None)
    p_runs.add_argument("--verdict", type=str, default=None)

    p_ana = sub.add_parser("analyses", parents=[common], help="Filter and tabulate post-hoc evals and analyses")
    p_ana.add_argument("--kind", choices=list(ANALYSIS_KINDS), default=None)
    p_ana.add_argument("--subject", type=str, default=None)
    p_ana.add_argument("--claim", dest="claim_id", type=str, default=None)
    p_ana.add_argument("--goal", dest="goal_id", type=str, default=None)
    p_ana.add_argument("--status", type=str, default=None)
    p_ana.add_argument("--control", type=str, default=None)

    p_cmp = sub.add_parser("compare", parents=[common], help="Ad-hoc comparison between two runs (composing attached evals)")
    p_cmp.add_argument("ctrl_id", type=str)
    p_cmp.add_argument("--vs", dest="treat_id", type=str, required=True)

    p_front = sub.add_parser("frontier", parents=[common], help="Primary metric vs GPU-hours Pareto frontier")
    p_front.add_argument("--metric", type=str, default=None)

    p_ideate = sub.add_parser("ideate", parents=[common], help="Propose next experiments from state diffs, tensions, and open questions")
    p_ideate.add_argument("--goal", dest="goal_id", type=str, default=None)

    p_brief = sub.add_parser("brief", parents=[common], help="Morning startup briefing on live runs, review inbox, and open threads")
    p_brief.add_argument("--goal", dest="goal_id", type=str, default=None)

    p_rep = sub.add_parser("report", parents=[common], help="Generate an evidence-grounded truth bundle for sharing results")
    p_rep.add_argument("--goal", dest="goal_id", type=str, default=None)
    p_rep.add_argument("--claim", dest="claim_ids", nargs="+", default=None)
    p_rep.add_argument("--runs", dest="run_ids", nargs="+", default=None)
    p_rep.add_argument("--analyses", dest="analysis_ids", nargs="+", default=None)
    p_rep.add_argument("--out", dest="out_dir", type=Path, default=None, help="Export report.md + copied local assets to directory")
    p_rep.add_argument("--strict", action="store_true", help="Exit non-zero if any cited item is unverified, stale, dirty, or preliminary")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "init":
        cmd_init(args.path, with_skill=args.with_skill, as_json=args.as_json)
        return

    try:
        root = find_root(explicit=args.root)
    except ConfigError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    g = ContourGraph(root)

    if args.cmd == "new":
        cmd_new(g, args.node_type, args.slug, args.title, kind=args.kind, as_json=args.as_json)
    elif args.cmd == "link":
        cmd_link(g, args.evidence_id, args.claim_id, relation=args.relation, as_json=args.as_json)
    elif args.cmd == "check":
        cmd_check(g, strict=args.strict, as_json=args.as_json)
    elif args.cmd == "verify":
        mode = "agent" if args.agent else "human"
        cmd_verify(g, args.claim_id, mode=mode, note=args.note, as_json=args.as_json)
    elif args.cmd == "review":
        cmd_review(g, as_json=args.as_json)
    elif args.cmd == "status":
        cmd_status(g, as_json=args.as_json)
    elif args.cmd in ("show", "inspect"):
        cmd_show(g, args.node_id, as_json=args.as_json)
    elif args.cmd == "runs":
        cmd_runs(
            g,
            tree=args.tree,
            goal_id=args.goal_id,
            claim_id=args.claim_id,
            eval_scope=args.eval_scope,
            status=args.status,
            control=args.control,
            verdict=args.verdict,
            as_json=args.as_json,
        )
    elif args.cmd == "analyses":
        cmd_analyses(
            g,
            kind=args.kind,
            subject=args.subject,
            claim_id=args.claim_id,
            goal_id=args.goal_id,
            status=args.status,
            control=args.control,
            as_json=args.as_json,
        )
    elif args.cmd == "compare":
        cmd_compare(g, args.ctrl_id, args.treat_id, as_json=args.as_json)
    elif args.cmd == "frontier":
        cmd_frontier(g, metric=args.metric, as_json=args.as_json)
    elif args.cmd == "ideate":
        cmd_ideate(g, goal_id=args.goal_id, as_json=args.as_json)
    elif args.cmd == "brief":
        cmd_brief(g, goal_id=args.goal_id, as_json=args.as_json)
    elif args.cmd == "report":
        cmd_report(
            g,
            goal_id=args.goal_id,
            claim_ids=args.claim_ids,
            run_ids=args.run_ids,
            analysis_ids=args.analysis_ids,
            out_dir=args.out_dir,
            strict=args.strict,
            as_json=args.as_json,
        )


if __name__ == "__main__":
    main()
