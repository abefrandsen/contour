"""Contour CLI: Research state integrity, introspection, and experiment ideation."""

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
    find_check_issues,
    find_review_items,
    fix_stale_vs_parent,
    frontier_metrics,
    issues_blocking_verify,
    select_primary_metric,
)
from .compare import compare_runs, get_fresh_vs_parent, recommend_relation
from .config import CONFIG_FILE, DEFAULT_CONFIG, SUBDIR_CANDIDATES, ConfigError, find_root
from .graph import EDGE_KEYS, ContourGraph, ContourNode
from .ideate import generate_ideation
from .mdio import as_wikilink, parse_markdown_file, strip_wiki, write_markdown_file

GOAL_BODY_TEMPLATE = (
    "## Objective & Success Criteria\n"
    "- TODO: State the overarching research problem and quantitative success criteria.\n\n"
    "## Current Synthesis & Open Questions\n"
    "- \n"
)

CLAIM_BODY_TEMPLATE = (
    "## Hypothesis & Mechanism\n"
    "- TODO: State the falsifiable hypothesis and conceptual mechanism.\n\n"
    "## Evidence Synthesis\n"
    "- **Holds:** \n"
    "- **Breaks:** \n"
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
        "description: Navigate, verify, synthesize, and ideate over the Contour ML research state.\n"
        "---\n\n"
        "# Contour: Epistemic ML Research Copilot Skill\n\n"
        "Run `contour status --json`, `contour runs --tree`, `contour check --json`, "
        "`contour frontier --json`, `contour ideate --json`, and `contour show <node_id>`.\n"
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

    for sub in ("goals", "claims", "runs"):
        (resolved / sub).mkdir(exist_ok=True)

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
            **({"skill_path": str(skill_path)} if skill_path else {}),
        })
        return
    status = "Created" if created_cfg else "Initialized"
    print(f"{status} Contour root at {resolved}")
    if skill_path:
        print(f"Installed agent skill at {skill_path}")


def cmd_new(g: ContourGraph, node_type: str, slug: str, title: str, as_json: bool = False) -> None:
    today = dt.date.today().isoformat()
    if node_type == "goal":
        node_id = slug if slug.startswith("goal_") else f"goal_{slug}"
        path = g.root / "goals" / f"{node_id}.md"
        fm: dict[str, Any] = {
            "id": node_id,
            "type": "goal",
            "title": title,
            "status": "active",
            "target_regimes": {"tasks": [], "scales": []},
        }
        body = GOAL_BODY_TEMPLATE
    elif node_type == "claim":
        node_id = slug if slug.startswith("clm_") else f"clm_{slug}"
        path = g.root / "claims" / f"{node_id}.md"
        fm = {
            "id": node_id,
            "type": "claim",
            "title": title,
            "status": "active",
            "superseded_by": None,
            "verification": "draft",
            "verified_hash": None,
            "audit_note": None,
            "goals": [],
            "supports": [],
            "contradicts": [],
            "qualifies": [],
            "scope": {"holds_in": [], "breaks_in": [], "untested": []},
        }
        body = CLAIM_BODY_TEMPLATE
    elif node_type == "run":
        date_str = today.replace("-", "")
        node_id = slug if slug.startswith("run_") else f"run_{date_str}_{slug}"
        path = g.root / "runs" / f"{node_id}.md"
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        fm = {
            "id": node_id,
            "type": "run",
            "title": title,
            "validity": "ok",
            "interventions": [],
            "intervention_note": None,
            "parent_runs": [],
            "baseline": None,
            "motivated_by": [],
            "status": "final",
            "created_at": now_iso,
            "finished_at": now_iso,
            "task": "unspecified",
            "arch_family": "unspecified",
            "arch_scale": "unspecified",
            "dataset_train": [],
            "tier": "proxy",
            "config": {},
            "compute": {"gpu_hours": None},
            "code": {"commit": None, "dirty": False},
            "launch_cmd": None,
            "metrics": {},
            "vs_parent": None,
        }
        body = "## Key Takeaway (TL;DR)\n- TODO: Summarize key outcome.\n\n## Observations & Anomalies\n- \n"
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
    run_id: str,
    claim_id: str,
    relation: str | None = None,
    as_json: bool = False,
) -> None:
    rid = strip_wiki(run_id)
    cid = strip_wiki(claim_id)
    run = g.runs.get(rid)
    if not run:
        raise SystemExit(f"Error: Unknown run {rid!r}")
    claim = g.claims.get(cid)
    if not claim:
        raise SystemExit(f"Error: Unknown claim {cid!r}")

    fresh_vs = get_fresh_vs_parent(g, run)
    if relation is None:
        if fresh_vs is not None:
            relation = recommend_relation(fresh_vs["control_level"], fresh_vs["verdict"])
        else:
            relation = "qualifies"
    if relation not in EDGE_KEYS:
        raise SystemExit(f"Error: Invalid relation {relation!r} (expected one of {EDGE_KEYS})")

    fm, body = parse_markdown_file(claim.path)
    for edge in EDGE_KEYS:
        existing = [strip_wiki(x) for x in (fm.get(edge) or [])]
        if edge == relation:
            if rid not in existing:
                existing.append(rid)
        else:
            existing = [x for x in existing if x != rid]
        fm[edge] = [as_wikilink(x) for x in existing]

    write_markdown_file(claim.path, fm, body)
    if as_json:
        _emit_json({
            "claim": cid,
            "run": rid,
            "relation": relation,
            "vs_parent": fresh_vs,
        })
        return
    badge = f" [{fresh_vs['control_level']}/{fresh_vs['verdict']}]" if fresh_vs else ""
    print(f"Linked {rid} -> {cid} under `{relation}`{badge}")


def cmd_check(g: ContourGraph, strict: bool = False, fix: bool = False, as_json: bool = False) -> None:
    fixed: list[str] = []
    if fix:
        fixed = fix_stale_vs_parent(g)
        g = ContourGraph(g.root, config=g.config)

    issues = find_check_issues(g)
    if as_json:
        _emit_json({
            "count": len(issues),
            "fixed": fixed,
            "issues": [i.as_dict() for i in issues],
        })
    else:
        print("=== CONTOUR MECHANICAL CHECK ===")
        if fixed:
            print(f"Fixed stale vs_parent on {len(fixed)} run(s): {', '.join(fixed)}")
        if not issues:
            print("All goals, claims, and runs pass mechanical verification!")
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
        print(f"  * {r['id']} ({r['task']}): {r['metric']}={r['value']} @ {r['gpu_hours']}h (commit={r['commit']})")

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
        f"check={len(issues)} | review={review_count}\n"
    )
    for goal in goals_out:
        print(f"[GOAL] {goal['id']} ({goal['status'].upper()}): {goal['title']}")
        if goal["tldr"]:
            print(f"  Objective: {goal['tldr']}")
        if not goal["claims"]:
            print("  (No linked claims)")
        for c in goal["claims"]:
            cnt = c["counts"]
            print(f"  └── [CLAIM] {c['id']} [{c['status']} | {c['verification']}]: {c['title']}")
            print(
                f"        Runs: +{cnt['supports']} supports | "
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
        if node.type == "run":
            fm_out["vs_parent"] = vs_parent
        extra: dict[str, Any] = {}
        if node.type == "claim":
            extra["effective_verification"] = effective_verification(g, node)
            extra["verified_hash_expected"] = compute_verified_hash(g, node)
        elif node.type == "goal":
            extra["claims"] = [c.id for c in g.goal_claims(node.id)]
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
        fm = node.frontmatter
        code = node.code
        print(f"Valid:  {node.validity} | Tier: {fm.get('tier', 'proxy')}")
        print(f"Task:   {fm.get('task')} | Arch: {fm.get('arch_family')} ({fm.get('arch_scale')})")
        print(f"Train:  {fm.get('dataset_train')} | Eval scopes: {node.eval_scopes}")
        print(f"GPU hr: {node.gpu_hours} | Commit: {code.get('commit')} | Dirty: {code.get('dirty')}")
        print(f"Interventions: {node.interventions}")
        if fm.get("intervention_note"):
            print(f"Note:   {fm.get('intervention_note')}")
        if node.config:
            print(f"Config: {node.config}")

        if vs_parent is not None:
            bid = strip_wiki(str(vs_parent.get("baseline", "")))
            print(
                f"\nvs_parent ({bid}) [control={vs_parent['control_level']} | "
                f"verdict={vs_parent['verdict']} | iso_compute={vs_parent['iso_compute']} | "
                f"hash={vs_parent['inputs_hash']}]:"
            )
            if vs_parent["facet_diff"]:
                print(f"  facet_diff:   {vs_parent['facet_diff']}")
            if vs_parent["config_diff"]:
                print(f"  config_diff:  {vs_parent['config_diff']}")
            if vs_parent["unaccounted"]:
                print(f"  UNACCOUNTED:  {vs_parent['unaccounted']}")
            if vs_parent["unrealized"]:
                print(f"  UNREALIZED:   {vs_parent['unrealized']}")
            for k, d in (vs_parent.get("deltas") or {}).items():
                sig = f" [{d['signal']}]" if d.get("signal") else ""
                print(f"  - {k}: {d['baseline']} -> {d['value']} ({d['delta']:+.4g}){sig}")
        elif node.metrics:
            print("\nMetrics:")
            for k, v in node.metrics.items():
                print(f"  - {k}: {v}")

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
                    rnode = g.runs.get(eid)
                    if rnode:
                        vs = get_fresh_vs_parent(g, rnode)
                        badge = f" [{vs['control_level']}/{vs['verdict']}]" if vs else ""
                        print(
                            f"  - {eid} [{rnode.status}/{rnode.validity}]{badge} "
                            f"({rnode.get('arch_family')} {rnode.get('arch_scale')}): {rnode.tldr}"
                        )
                    else:
                        print(f"  - {eid} (missing run)")

    elif node.type == "goal":
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
    task: str | None = None,
    eval_scope: str | None = None,
    arch: str | None = None,
    intervention: str | None = None,
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
                allowed_by_claim.update(c.refs(edge))
        for r in runs:
            if gid in r.refs("motivated_by") or any(c.id in r.refs("motivated_by") for c in g_claims):
                allowed_by_claim.add(r.id)
    if claim_id:
        cid = strip_wiki(claim_id)
        cnode = g.claims.get(cid)
        c_set: set[str] = set()
        if cnode:
            for edge in EDGE_KEYS:
                c_set.update(cnode.refs(edge))
        for r in runs:
            if cid in r.refs("motivated_by"):
                c_set.add(r.id)
        allowed_by_claim = c_set if allowed_by_claim is None else (allowed_by_claim & c_set)

    filtered: list[tuple[ContourNode, dict[str, Any] | None]] = []
    for r in sorted(runs, key=lambda x: x.id):
        if allowed_by_claim is not None and r.id not in allowed_by_claim:
            continue
        if task and r.get("task") != task:
            continue
        if eval_scope and eval_scope not in r.eval_scopes:
            continue
        if arch:
            arch_str = f"{r.get('arch_family')}-{r.get('arch_scale')}"
            if arch.lower() not in arch_str.lower():
                continue
        if intervention and not any(intervention.lower() in iv.lower() for iv in r.interventions):
            continue
        vs = get_fresh_vs_parent(g, r)
        ctrl_lvl = vs["control_level"] if vs else "-"
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
                    "validity": r.validity,
                    "task": r.get("task"),
                    "arch": f"{r.get('arch_family')}-{r.get('arch_scale')}",
                    "tier": r.get("tier", "proxy"),
                    "gpu_hours": r.gpu_hours,
                    "parents": r.parents,
                    "baseline": r.effective_baseline,
                    "control_level": vs["control_level"] if vs else None,
                    "verdict": vs["verdict"] if vs else None,
                    "primary_delta": _primary_delta_summary(g, vs),
                    "interventions": r.interventions,
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
            p = r.parents[0] if r.parents else None
            if p and p in by_id:
                children.setdefault(p, []).append(r.id)
            else:
                roots.append(r.id)

        def _print_branch(rid: str, prefix: str = "", is_last: bool = True, is_root: bool = True) -> None:
            r, vs = by_id[rid]
            badge = f"[{vs['control_level']}/{vs['verdict']} {_primary_delta_summary(g, vs)}]" if vs else "[baseline]"
            interv = ", ".join(r.interventions) if r.interventions else "-"
            conn = "" if is_root else ("└── " if is_last else "├── ")
            print(f"{prefix}{conn}{r.id} ({r.get('task')} | {r.get('arch_family')}-{r.get('arch_scale')}) {badge} :: {interv}")
            kids = children.get(rid, [])
            child_prefix = "" if is_root else (prefix + ("    " if is_last else "│   "))
            for idx, kid in enumerate(kids):
                _print_branch(kid, child_prefix, idx == len(kids) - 1, is_root=False)

        for root_id in roots:
            _print_branch(root_id)
        return

    print(f"=== CONTOUR RUNS ({len(filtered)}) ===")
    print(f"{'ID':<22} {'TASK':<20} {'ARCH':<10} {'TIER':<7} {'CONTROL':<11} {'VERDICT':<15} {'PRIMARY Δ':<20} {'INTERVENTIONS'}")
    print("-" * 120)
    for r, vs in filtered:
        ctrl_s = vs["control_level"] if vs else "-"
        verd_s = vs["verdict"] if vs else "-"
        pdelta = _primary_delta_summary(g, vs)
        arch_s = f"{r.get('arch_family')}-{r.get('arch_scale')}"
        interv_s = ", ".join(r.interventions) if r.interventions else "-"
        print(
            f"{r.id:<22} {str(r.get('task')):<20} {arch_s:<10} {str(r.get('tier', 'proxy')):<7} "
            f"{ctrl_s:<11} {verd_s:<15} {pdelta:<20} {interv_s}"
        )


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
        f"[control={out['control_level']} | verdict={out['verdict']} | iso_compute={out['iso_compute']}] ==="
    )
    print(f"Recommended link: {out['recommended_link']}")
    if out["facet_diff"]:
        print(f"Facet diff:   {out['facet_diff']}")
    if out["config_diff"]:
        print(f"Config diff:  {out['config_diff']}")
    if out["unaccounted"]:
        print(f"UNACCOUNTED:  {out['unaccounted']}")
    if out["unrealized"]:
        print(f"UNREALIZED:   {out['unrealized']}")
    for k, d in out["deltas"].items():
        sig = f" [{d['signal']}]" if d.get("signal") else ""
        print(f"  - {k}: {d['baseline']} -> {d['value']} ({d['delta']:+.4g}){sig}")


def cmd_frontier(
    g: ContourGraph,
    task: str | None = None,
    metric: str | None = None,
    as_json: bool = False,
) -> None:
    usable = g.usable_runs()
    if task:
        usable = [r for r in usable if r.get("task") == task]
    metrics_to_show = frontier_metrics(g, usable, metric_filter=metric)
    tables = [
        compute_frontier(g, task=task, metric_filter=m)
        for m in (metrics_to_show or [metric])
    ]

    def _serialize(pm: str | None, direction: str, runs: list[ContourNode], pareto_ids: set[str]) -> dict[str, Any]:
        return {
            "metric": pm,
            "direction": direction,
            "runs": [
                {
                    "id": r.id,
                    "task": r.get("task"),
                    "arch": f"{r.get('arch_family')}-{r.get('arch_scale')}",
                    "gpu_hours": r.gpu_hours,
                    "value": r.metrics.get(pm or ""),
                    "on_frontier": r.id in pareto_ids,
                    "dirty": bool(r.code.get("dirty", False)),
                    "interventions": r.interventions,
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
        print(f"{'RUN ID':<22} {'ARCH':<12} {'GPU_HR':<8} {'METRIC':<12} {'FRONTIER':<10} {'INTERVENTIONS'}")
        print("-" * 100)
        for r in runs:
            arch_str = f"{r.get('arch_family')}-{r.get('arch_scale')}"
            gpu_str = f"{r.gpu_hours:.1f}" if r.gpu_hours is not None else "-"
            val = r.metrics.get(primary_metric or "", "-")
            mark = "  ▲" if r.id in pareto_ids else ""
            interv = ", ".join(r.interventions) if r.interventions else str(r.get("intervention_note") or "baseline")
            print(f"{r.id:<22} {arch_str:<12} {gpu_str:<8} {str(val):<12} {mark:<10} {interv[:40]}")

    if not printed_any:
        print("No matching usable runs found.")


def cmd_ideate(g: ContourGraph, goal_id: str | None = None, as_json: bool = False) -> None:
    out = generate_ideation(g, goal_id=goal_id)
    if as_json:
        _emit_json(out)
        return

    print("=== CONTOUR IDEATE: NEXT EXPERIMENT CANDIDATES ===\n")
    print(f"[Move 1: Isolate Confounds] ({len(out['move_1_isolate_confounds'])})")
    for item in out["move_1_isolate_confounds"]:
        print(f"  * {item['run']} (Δ={item['primary_delta']} | est={item['estimated_gpu_hours']}h): {item['suggestion']}")

    print(f"\n[Move 2: Resolve Tensions & Tradeoffs] ({len(out['move_2_resolve_tensions'])})")
    for item in out["move_2_resolve_tensions"]:
        if item["kind"] == "claim_tension":
            print(f"  * {item['claim']} [{item['status']}]: contradicts={item['contradicts']} | qualifies={item['qualifies']} (est={item['estimated_gpu_hours']}h)")
        else:
            print(f"  * {item['run']} tradeoff (est={item['estimated_gpu_hours']}h): interventions={item['interventions']}")

    print(f"\n[Move 3: Combine & Compound Wins] ({len(out['move_3_combine_wins'])})")
    for item in out["move_3_combine_wins"]:
        cross = "CROSSES τ" if item["crosses_threshold"] else "sub-τ"
        print(
            f"  * {item['left']['id']} ({item['left']['verdict']}) + {item['right']['id']} ({item['right']['verdict']}) "
            f"-> {item['combined_interventions']} | proj Δ={item['projected_primary_delta']:+.3g} [{cross}] (est={item['estimated_gpu_hours']}h)"
        )

    print(f"\n[Move 4: Strip Complexity (Occam's Razor)] ({len(out['move_4_strip_complexity'])})")
    for item in out["move_4_strip_complexity"]:
        strips = [a["strip"] for a in item["candidate_ablations"]]
        print(f"  * {item['run']} ({item['verdict']} | est={item['estimated_gpu_hours']}h) -> leave-one-out strips: {strips}")

    print(f"\n[Move 5: Expand Untested Regimes] ({len(out['move_5_expand_regimes'])})")
    for item in out["move_5_expand_regimes"]:
        print(
            f"  * {item['claim']}: missing tasks={item['missing_target_tasks']}, "
            f"scales={item['missing_target_scales']}, untested={item['scope_untested'][:2]}"
        )

    print(f"\n[Move 6: Diagnose Frontier Bottlenecks] ({len(out['move_6_diagnose_frontier'])})")
    for item in out["move_6_diagnose_frontier"]:
        obs = item["observations"][0] if item["observations"] else "See diagnostics"
        print(f"  * {item['task']} -> {item['best_run']} ({item['metric']}={item['value']} @ {item['estimated_gpu_hours']}h): {obs}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=argparse.SUPPRESS, help="Path to contour root")
    common.add_argument("--json", dest="as_json", action="store_true", default=argparse.SUPPRESS, help="Emit JSON")

    parser = argparse.ArgumentParser(
        prog="contour",
        description="Contour: Epistemic ML research state, verification, and ideation.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Path to contour root (with contour.yaml)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit machine-readable JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", parents=[common], help="Initialize a Contour root")
    p_init.add_argument("path", nargs="?", type=Path, default=None, help="Target directory (default: ./contour)")
    p_init.add_argument("--with-skill", action="store_true", help="Write .agent/skills/contour-research/SKILL.md")

    p_new = sub.add_parser("new", parents=[common], help="Create a new goal, claim, or run")
    p_new.add_argument("node_type", choices=["goal", "claim", "run"])
    p_new.add_argument("slug", type=str)
    p_new.add_argument("--title", type=str, required=True)

    p_link = sub.add_parser("link", parents=[common], help="Link a run onto a claim")
    p_link.add_argument("run_id", type=str)
    p_link.add_argument("claim_id", type=str)
    p_link.add_argument("--as", dest="relation", choices=list(EDGE_KEYS), default=None)

    p_check = sub.add_parser("check", parents=[common], help="Run Tier-1 mechanical verification checks C001-C009")
    p_check.add_argument("--strict", action="store_true", help="Exit non-zero on WARN")
    p_check.add_argument("--fix", action="store_true", help="Recompute stale vs_parent blocks")

    p_verify = sub.add_parser("verify", parents=[common], help="Record agentic or human verification on a claim")
    p_verify.add_argument("claim_id", type=str)
    grp = p_verify.add_mutually_exclusive_group(required=True)
    grp.add_argument("--agent", action="store_true")
    grp.add_argument("--human", action="store_true")
    p_verify.add_argument("--note", type=str, default=None)

    sub.add_parser("review", parents=[common], help="Tier-3 selective human verification inbox")
    sub.add_parser("status", parents=[common], help="Overview of Goals -> Claims -> Run counts")

    p_show = sub.add_parser("show", parents=[common], aliases=["inspect"], help="Inspect a Goal, Claim, or Run")
    p_show.add_argument("node_id", type=str)

    p_runs = sub.add_parser("runs", parents=[common], help="Filter and tabulate runs or render lineage tree")
    p_runs.add_argument("--tree", action="store_true")
    p_runs.add_argument("--goal", dest="goal_id", type=str, default=None)
    p_runs.add_argument("--claim", dest="claim_id", type=str, default=None)
    p_runs.add_argument("--task", type=str, default=None)
    p_runs.add_argument("--eval", dest="eval_scope", type=str, default=None)
    p_runs.add_argument("--arch", type=str, default=None)
    p_runs.add_argument("--intervention", type=str, default=None)
    p_runs.add_argument("--control", type=str, default=None)
    p_runs.add_argument("--verdict", type=str, default=None)

    p_cmp = sub.add_parser("compare", parents=[common], help="Ad-hoc comparison between two runs")
    p_cmp.add_argument("ctrl_id", type=str)
    p_cmp.add_argument("--vs", dest="treat_id", type=str, required=True)

    p_front = sub.add_parser("frontier", parents=[common], help="Primary metric vs GPU-hours Pareto frontier")
    p_front.add_argument("--task", type=str, default=None)
    p_front.add_argument("--metric", type=str, default=None)

    p_ideate = sub.add_parser("ideate", parents=[common], help="Propose next experiments across the 6 Scientific Moves")
    p_ideate.add_argument("--goal", dest="goal_id", type=str, default=None)

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
        cmd_new(g, args.node_type, args.slug, args.title, as_json=args.as_json)
    elif args.cmd == "link":
        cmd_link(g, args.run_id, args.claim_id, relation=args.relation, as_json=args.as_json)
    elif args.cmd == "check":
        cmd_check(g, strict=args.strict, fix=args.fix, as_json=args.as_json)
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
            task=args.task,
            eval_scope=args.eval_scope,
            arch=args.arch,
            intervention=args.intervention,
            control=args.control,
            verdict=args.verdict,
            as_json=args.as_json,
        )
    elif args.cmd == "compare":
        cmd_compare(g, args.ctrl_id, args.treat_id, as_json=args.as_json)
    elif args.cmd == "frontier":
        cmd_frontier(g, task=args.task, metric=args.metric, as_json=args.as_json)
    elif args.cmd == "ideate":
        cmd_ideate(g, goal_id=args.goal_id, as_json=args.as_json)


if __name__ == "__main__":
    main()
