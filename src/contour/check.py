"""`contour check`, `contour verify`, review inbox helpers, and Pareto `frontier`."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
from pathlib import Path
from typing import Any

from .compare import (
    _analysis_hash_snapshot,
    _run_hash_snapshot,
    canonical_hash,
    compute_analysis_control_level,
    compute_standalone_control_level,
    get_fresh_vs_parent,
)
from .config import is_valid_metric_key
from .graph import (
    ANALYSIS_KINDS,
    CLAIM_STATUSES,
    EDGE_KEYS,
    GOAL_STATUSES,
    RUN_STATUSES,
    VERIFICATION_STATES,
    ContourGraph,
    ContourNode,
    is_remote_uri,
)
from .mdio import extract_wikilinks

ERROR = "error"
WARN = "warn"
STUCK_RUN_HOURS = 72.0

# Human-readable invariant rule names
RULE_SCHEMA_ERROR = "schema-error"
RULE_TAINTED_EVIDENCE = "tainted-evidence"
RULE_OVERCLAIMED_STATUS = "overclaimed-status"
RULE_STALE_VERIFICATION = "stale-verification"
RULE_UNSYNTHESIZED_EVIDENCE = "unsynthesized-evidence"
RULE_TODO_PLACEHOLDER = "todo-placeholder"
RULE_STUCK_JOB = "stuck-job"
RULE_UNDECLARED_METRIC = "undeclared-metric"
RULE_MISSING_ARTIFACT = "missing-artifact"


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    category: str  # "hygiene" | "epistemic"
    node: str
    message: str
    remediation: str
    related_nodes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.code,
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "node": self.node,
            "message": self.message,
            "remediation": self.remediation,
            "related_nodes": list(self.related_nodes),
        }


def evidence_control_level(g: ContourGraph, node: ContourNode) -> str:
    """Return mechanical cleanliness (`invalid | exploratory | clean`) for a Run or Analysis."""
    if node.type == "analysis":
        return compute_analysis_control_level(g, node)
    if node.type == "run":
        if not node.is_usable_evidence:
            return "invalid"
        if node.parent is not None:
            if node.parent not in g.runs:
                return "invalid"
            pnode = g.runs[node.parent]
            if not pnode.is_usable_evidence:
                return "invalid"
            fresh = get_fresh_vs_parent(g, node)
            if fresh is not None:
                return str(fresh["control_level"])
        return compute_standalone_control_level(node.frontmatter)
    return "invalid"


def compute_verified_hash(g: ContourGraph, claim: ContourNode) -> str:
    """Content hash of claim frontmatter (excluding verification fields), claim body prose, and linked evidence."""
    fm_part = {
        k: v
        for k, v in claim.frontmatter.items()
        if k not in ("verification", "verified_hash", "audit_note")
    }
    linked_hashes: dict[str, str] = {}
    for edge in EDGE_KEYS:
        for eid in claim.refs(edge):
            if eid in g.runs:
                run = g.runs[eid]
                fresh = get_fresh_vs_parent(g, run)
                if fresh is not None:
                    linked_hashes[eid] = fresh["inputs_hash"]
                else:
                    linked_hashes[eid] = canonical_hash(_run_hash_snapshot(run.id, run.frontmatter, run.config))
            elif eid in g.analyses:
                ana = g.analyses[eid]
                linked_hashes[eid] = canonical_hash(_analysis_hash_snapshot(ana, g))
            else:
                linked_hashes[eid] = "missing"
    return canonical_hash({
        "claim": fm_part,
        "body": claim.body.strip(),
        "linked_evidence": linked_hashes,
    })


def is_verification_stale(g: ContourGraph, claim: ContourNode) -> bool:
    if claim.verification == "draft":
        return False
    return claim.get("verified_hash") != compute_verified_hash(g, claim)


def effective_verification(g: ContourGraph, claim: ContourNode) -> str:
    """Return 'draft' if verification is stale, else stored verification."""
    if is_verification_stale(g, claim):
        return "draft"
    return claim.verification


def _parse_iso_utc(raw: Any) -> dt.datetime | None:
    if not raw:
        return None
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _check_run_cycles(g: ContourGraph, add: Any) -> None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}

    def dfs(nid: str) -> None:
        color[nid] = GRAY
        run = g.runs.get(nid)
        if run and run.parent and run.parent in g.runs:
            pid = run.parent
            if color.get(pid) == GRAY:
                add(Issue(
                    RULE_SCHEMA_ERROR, ERROR, "hygiene", nid,
                    f"Cycle in `parent` lineage involving {nid} -> {pid}",
                    "Remove the circular `parent` reference.",
                    (nid, pid),
                ))
            elif color.get(pid, WHITE) == WHITE:
                dfs(pid)
        color[nid] = BLACK

    for rid in g.runs:
        if color.get(rid, WHITE) == WHITE:
            dfs(rid)


def _check_local_artifacts(g: ContourGraph, node: ContourNode, add: Any) -> None:
    """Emit non-blocking hygiene WARN (`missing-artifact`) when a local relative artifact path is missing."""
    for art in node.artifacts:
        raw_path = str(art.get("path") or "").strip()
        if not raw_path or is_remote_uri(raw_path):
            continue
        p = Path(raw_path)
        candidates = [p] if p.is_absolute() else [g.root / p, g.root.parent / p]
        if not any(c.exists() for c in candidates):
            add(Issue(
                RULE_MISSING_ARTIFACT, WARN, "hygiene", node.id,
                f"{node.id} references missing local artifact {raw_path!r}",
                "Verify the artifact path (or ignore if generated on another machine).",
            ))


def find_check_issues(
    g: ContourGraph,
    now: dt.datetime | None = None,
) -> list[Issue]:
    """Evaluate all Tier-1 (ERROR) and Tier-2 (WARN) invariants."""
    issues: list[Issue] = []
    add = issues.append
    now_dt = now or dt.datetime.now(dt.timezone.utc)

    for err in g.load_errors:
        add(Issue(
            RULE_SCHEMA_ERROR, ERROR, "hygiene", "graph",
            f"Unparseable frontmatter: {err}",
            "Fix the YAML frontmatter syntax.",
        ))
    for nid in sorted(set(g.duplicate_ids)):
        add(Issue(
            RULE_SCHEMA_ERROR, ERROR, "hygiene", nid,
            f"Duplicate node id {nid!r}",
            "Ensure every Goal, Claim, Run, and Analysis has a unique filename stem.",
        ))

    # --- Goals --------------------------------------------------------------
    for goal in g.goals.values():
        if goal.status not in GOAL_STATUSES:
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", goal.id,
                f"{goal.id} has invalid status {goal.status!r} (expected one of {GOAL_STATUSES})",
                f"Set `status` to one of {GOAL_STATUSES}.",
            ))

    # --- Claims -------------------------------------------------------------
    for claim in g.claims.values():
        if claim.status not in CLAIM_STATUSES:
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", claim.id,
                f"{claim.id} has invalid status {claim.status!r} (expected one of {CLAIM_STATUSES})",
                f"Set `status` to one of {CLAIM_STATUSES}.",
            ))
        if claim.verification not in VERIFICATION_STATES:
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", claim.id,
                f"{claim.id} has invalid verification {claim.verification!r} (expected one of {VERIFICATION_STATES})",
                f"Set `verification` to one of {VERIFICATION_STATES}.",
            ))

        for gid in claim.refs("goals"):
            if gid not in g.goals:
                add(Issue(
                    RULE_SCHEMA_ERROR, ERROR, "hygiene", claim.id,
                    f"{claim.id}.goals references missing goal {gid}",
                    "Create the goal or remove the broken wikilink.",
                    (gid,),
                ))

        seen_edges: dict[str, str] = {}
        body_links = extract_wikilinks(claim.body)
        for edge in EDGE_KEYS:
            for eid in claim.refs(edge):
                if eid in seen_edges and seen_edges[eid] != edge:
                    add(Issue(
                        RULE_SCHEMA_ERROR, ERROR, "hygiene", claim.id,
                        f"{claim.id} lists {eid} under both `{seen_edges[eid]}` and `{edge}`",
                        f"Use `contour link {eid} {claim.id} --as <relation>` to place it under one relation.",
                        (eid,),
                    ))
                seen_edges[eid] = edge
                ev = g.get_evidence(eid)
                if ev is None:
                    add(Issue(
                        RULE_SCHEMA_ERROR, ERROR, "hygiene", claim.id,
                        f"{claim.id} ({edge}) references missing run/analysis {eid}",
                        "Remove the broken link or restore the missing evidence file.",
                        (eid,),
                    ))
                    continue
                if evidence_control_level(g, ev) == "invalid":
                    add(Issue(
                        RULE_TAINTED_EVIDENCE, ERROR, "hygiene", claim.id,
                        f"{claim.id} ({edge}) links invalid/quarantined {ev.type} {eid} (status={ev.status}, bugged={ev.bugged})",
                        "Unlink it from frontmatter (you may still note it in prose as a cautionary record).",
                        (eid,),
                    ))
                if eid not in body_links:
                    add(Issue(
                        RULE_UNSYNTHESIZED_EVIDENCE, WARN, "hygiene", claim.id,
                        f"{claim.id} links {eid} ({edge}) but does not cite [[{eid}]] in the Markdown body",
                        f"Synthesize [[{eid}]] in `## Evidence Synthesis`.",
                        (eid,),
                    ))

        if claim.status in ("active", "supported") and "- TODO:" in claim.body:
            add(Issue(
                RULE_TODO_PLACEHOLDER, WARN, "hygiene", claim.id,
                f"{claim.id} ({claim.status}) still contains unfilled `- TODO:` placeholder",
                "Replace `- TODO:` with concrete hypothesis/synthesis prose.",
            ))

        # overclaimed-status
        if claim.status == "supported":
            sup_ids = claim.refs("supports")
            con_ids = claim.refs("contradicts")
            if not sup_ids:
                add(Issue(
                    RULE_OVERCLAIMED_STATUS, WARN, "epistemic", claim.id,
                    f"{claim.id} has status `supported` but `supports` is empty",
                    "Link clean supporting evidence (run or analysis) or set `status: active`.",
                ))
            else:
                for eid in sup_ids:
                    ev = g.get_evidence(eid)
                    if ev is None:
                        continue
                    ctrl = evidence_control_level(g, ev)
                    if ctrl != "clean":
                        add(Issue(
                            RULE_OVERCLAIMED_STATUS, WARN, "epistemic", claim.id,
                            f"{claim.id} is `supported`, but `supports` evidence {eid} is `{ctrl}` (requires `clean`)",
                            f"Run `contour link {eid} {claim.id} --as qualifies` or calibrate claim `status: active`.",
                            (eid,),
                        ))
            if con_ids:
                add(Issue(
                    RULE_OVERCLAIMED_STATUS, WARN, "epistemic", claim.id,
                    f"{claim.id} is `supported` despite unresolved `contradicts` evidence ({', '.join(con_ids)})",
                    "Set `status: active`/`falsified`, or narrow the claim and move boundary evidence to `qualifies`.",
                    tuple(con_ids),
                ))
        elif claim.status == "falsified":
            con_ids = claim.refs("contradicts")
            has_clean = any(
                (ev := g.get_evidence(eid)) is not None and evidence_control_level(g, ev) == "clean"
                for eid in con_ids
            )
            if not has_clean:
                add(Issue(
                    RULE_OVERCLAIMED_STATUS, WARN, "epistemic", claim.id,
                    f"{claim.id} has status `falsified`, but no `contradicts` evidence is `clean`",
                    "Link clean contradicting evidence or set `status: active`.",
                    tuple(con_ids),
                ))

        # stale-verification
        if is_verification_stale(g, claim):
            add(Issue(
                RULE_STALE_VERIFICATION, WARN, "epistemic", claim.id,
                f"{claim.id} verification ({claim.verification}) is stale (verified_hash mismatch)",
                f"Re-audit and run `contour verify {claim.id} --agent` (or `--human`).",
            ))

    # --- Runs ---------------------------------------------------------------
    _check_run_cycles(g, add)
    claim_linked_evidence: dict[str, set[str]] = {
        cid: {eid for edge in EDGE_KEYS for eid in c.refs(edge)}
        for cid, c in g.claims.items()
    }

    for run in g.runs.values():
        if run.config_error:
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", run.id,
                f"{run.id}: {run.config_error}",
                "Restore or fix the tracked YAML file referenced by `config_path`.",
            ))
        if run.status not in RUN_STATUSES:
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", run.id,
                f"{run.id} has invalid status {run.status!r} (expected one of {RUN_STATUSES})",
                f"Set `status` to one of {RUN_STATUSES}.",
            ))
        if "bugged" in run.frontmatter and not isinstance(run.frontmatter["bugged"], bool):
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", run.id,
                f"{run.id}.bugged must be a boolean",
                "Set `bugged: true` (or remove the key if valid).",
            ))

        if run.parent and run.parent not in g.runs:
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", run.id,
                f"{run.id}.parent references missing run {run.parent}",
                "Fix or clear the `parent` wikilink.",
                (run.parent,),
            ))

        for ref in run.refs("motivated_by"):
            if ref not in g.all_nodes:
                add(Issue(
                    RULE_SCHEMA_ERROR, ERROR, "hygiene", run.id,
                    f"{run.id}.motivated_by references missing node {ref}",
                    "Fix the motivated_by wikilink.",
                    (ref,),
                ))
            elif ref in g.claims and run.is_usable_evidence and run.id not in claim_linked_evidence.get(ref, set()):
                add(Issue(
                    RULE_UNSYNTHESIZED_EVIDENCE, WARN, "hygiene", ref,
                    f"{run.id} was motivated by {ref}, but {ref} has not linked it",
                    f"Run `contour link {run.id} {ref}` and synthesize it into {ref}.",
                    (run.id,),
                ))

        if run.status == "final" and not run.bugged and "- TODO:" in run.body:
            add(Issue(
                RULE_TODO_PLACEHOLDER, WARN, "hygiene", run.id,
                f"{run.id} (final) still contains unfilled `- TODO:` placeholder",
                "Summarize the key takeaway in `## Key Takeaway (TL;DR)`.",
            ))

        if run.status == "running":
            created = _parse_iso_utc(run.get("created_at"))
            if created is not None:
                age_hours = (now_dt - created).total_seconds() / 3600.0
                if age_hours > STUCK_RUN_HOURS:
                    add(Issue(
                        RULE_STUCK_JOB, WARN, "hygiene", run.id,
                        f"{run.id} has been `running` for {age_hours:.1f}h (> {STUCK_RUN_HOURS:.0f}h)",
                        "Set `status: failed` if the job died, or finish/resume the run.",
                    ))

        for mkey in run.metrics:
            if not is_valid_metric_key(mkey):
                add(Issue(
                    RULE_UNDECLARED_METRIC, WARN, "hygiene", run.id,
                    f"{run.id} logs malformed metric key {mkey!r} (expected `<scope>/<lowercase_name>`)",
                    "Use `<scope>/<name>` metric naming in `contour.yaml` and training scripts.",
                ))
            elif not g.config.has_metric(mkey):
                add(Issue(
                    RULE_UNDECLARED_METRIC, WARN, "hygiene", run.id,
                    f"{run.id} logs undeclared metric {mkey!r} missing from contour.yaml",
                    f"Register `{mkey}` (direction, kind, min_delta / min_rel_delta) in contour.yaml.",
                ))

        _check_local_artifacts(g, run, add)

    # --- Analyses -----------------------------------------------------------
    for ana in g.analyses.values():
        if ana.kind not in ANALYSIS_KINDS:
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", ana.id,
                f"{ana.id} has invalid kind {ana.kind!r} (expected one of {ANALYSIS_KINDS})",
                f"Set `kind` to one of {ANALYSIS_KINDS}.",
            ))
        if ana.status not in RUN_STATUSES:
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", ana.id,
                f"{ana.id} has invalid status {ana.status!r} (expected one of {RUN_STATUSES})",
                f"Set `status` to one of {RUN_STATUSES}.",
            ))
        if "bugged" in ana.frontmatter and not isinstance(ana.frontmatter["bugged"], bool):
            add(Issue(
                RULE_SCHEMA_ERROR, ERROR, "hygiene", ana.id,
                f"{ana.id}.bugged must be a boolean",
                "Set `bugged: true` (or remove the key if valid).",
            ))

        for sid in ana.subjects:
            if sid not in g.runs:
                add(Issue(
                    RULE_SCHEMA_ERROR, ERROR, "hygiene", ana.id,
                    f"{ana.id}.subjects references missing run {sid}",
                    "Fix the subject wikilink or restore the missing run file.",
                    (sid,),
                ))
            elif ana.is_usable_evidence and not g.runs[sid].is_usable_evidence:
                srun = g.runs[sid]
                add(Issue(
                    RULE_TAINTED_EVIDENCE, ERROR, "hygiene", ana.id,
                    f"{ana.id} depends on non-usable subject run {sid} (status={srun.status}, bugged={srun.bugged})",
                    f"Quarantine {ana.id} (`bugged: true`) or re-run on a valid subject.",
                    (sid,),
                ))

        for ref in ana.refs("motivated_by"):
            if ref not in g.all_nodes:
                add(Issue(
                    RULE_SCHEMA_ERROR, ERROR, "hygiene", ana.id,
                    f"{ana.id}.motivated_by references missing node {ref}",
                    "Fix the motivated_by wikilink.",
                    (ref,),
                ))
            elif ref in g.claims and ana.is_usable_evidence and ana.id not in claim_linked_evidence.get(ref, set()):
                add(Issue(
                    RULE_UNSYNTHESIZED_EVIDENCE, WARN, "hygiene", ref,
                    f"{ana.id} was motivated by {ref}, but {ref} has not linked it",
                    f"Run `contour link {ana.id} {ref} --as <relation>` and synthesize it into {ref}.",
                    (ana.id,),
                ))

        if ana.status == "final" and not ana.bugged and "- TODO:" in ana.body:
            add(Issue(
                RULE_TODO_PLACEHOLDER, WARN, "hygiene", ana.id,
                f"{ana.id} (final) still contains unfilled `- TODO:` placeholder",
                "Summarize the key finding in `## Key Takeaway (TL;DR)`.",
            ))

        if ana.status == "running":
            created = _parse_iso_utc(ana.get("created_at"))
            if created is not None:
                age_hours = (now_dt - created).total_seconds() / 3600.0
                if age_hours > STUCK_RUN_HOURS:
                    add(Issue(
                        RULE_STUCK_JOB, WARN, "hygiene", ana.id,
                        f"{ana.id} has been `running` for {age_hours:.1f}h (> {STUCK_RUN_HOURS:.0f}h)",
                        "Set `status: failed` if the job died, or finish/resume the analysis.",
                    ))

        if ana.kind == "eval":
            for mkey in ana.metrics:
                if not is_valid_metric_key(mkey):
                    add(Issue(
                        RULE_UNDECLARED_METRIC, WARN, "hygiene", ana.id,
                        f"{ana.id} (eval) logs malformed metric key {mkey!r} (expected `<scope>/<lowercase_name>`)",
                        "Use `<scope>/<name>` metric naming in `contour.yaml` and eval scripts.",
                    ))
                elif not g.config.has_metric(mkey):
                    add(Issue(
                        RULE_UNDECLARED_METRIC, WARN, "hygiene", ana.id,
                        f"{ana.id} (eval) logs undeclared metric {mkey!r} missing from contour.yaml",
                        f"Register `{mkey}` (direction, kind, min_delta / min_rel_delta) in contour.yaml.",
                    ))

        _check_local_artifacts(g, ana, add)

    for mkey in sorted(g.config.metrics):
        kind = g.config.metric_kind(mkey)
        if kind in ("quality", "cost") and not g.config.has_threshold(mkey):
            add(Issue(
                RULE_UNDECLARED_METRIC, WARN, "hygiene", "contour.yaml",
                f"Metric {mkey!r} ({kind}) in contour.yaml lacks `min_delta` / `min_rel_delta`",
                f"Add `min_delta` and/or `min_rel_delta` for `{mkey}` in contour.yaml.",
            ))

    return issues


def fix_stale_vs_parent(g: ContourGraph) -> list[str]:
    """No-op kept for API compatibility: comparisons are computed dynamically in memory."""
    return []


def issues_blocking_verify(g: ContourGraph, claim_id: str) -> list[Issue]:
    """Gate: exits non-zero while any ERROR or epistemic issue applies to the claim or its supports."""
    claim = g.claims.get(claim_id)
    if not claim:
        return []
    sup_ids = set(claim.refs("supports"))
    related_ids = set(sup_ids)
    for eid in list(sup_ids):
        ana = g.analyses.get(eid)
        if ana is not None:
            related_ids.update(ana.subjects)
    all_issues = find_check_issues(g)
    blocking: list[Issue] = []
    for issue in all_issues:
        if issue.code == RULE_STALE_VERIFICATION and issue.node == claim_id:
            continue
        applies_to_claim = (
            issue.node == claim_id
            or bool(related_ids & set(issue.related_nodes))
            or issue.node in related_ids
        )
        if applies_to_claim and (issue.severity == ERROR or issue.category == "epistemic"):
            blocking.append(issue)
    return blocking


def select_primary_metric(
    g: ContourGraph,
    runs: list[ContourNode],
    metric_filter: str | None = None,
) -> str | None:
    if metric_filter:
        return metric_filter
    for r in runs:
        for k in g.effective_run_metrics(r):
            if g.config.is_primary_metric(k):
                return k
    for r in runs:
        for k in g.effective_run_metrics(r):
            if g.config.metric_kind(k) == "quality":
                return k
    for r in runs:
        eff = g.effective_run_metrics(r)
        if eff:
            return next(iter(eff))
    return None


def frontier_metrics(
    g: ContourGraph,
    runs: list[ContourNode],
    metric_filter: str | None = None,
) -> list[str]:
    """Return all primary metrics present across inline + attached `kind: eval` metrics."""
    if metric_filter:
        return [metric_filter]
    present = {k for r in runs for k in g.effective_run_metrics(r)}
    primaries = [k for k in g.config.metrics if k in present and g.config.is_primary_metric(k)]
    if primaries:
        return primaries
    fallback = select_primary_metric(g, runs)
    return [fallback] if fallback else []


def compute_frontier(
    g: ContourGraph,
    metric_filter: str | None = None,
) -> tuple[str | None, str, list[ContourNode], set[str]]:
    """Compute primary metric vs GPU-hours Pareto frontier (composing inline + attached `kind: eval`)."""
    runs = g.usable_runs()
    if not runs:
        return None, "max", [], set()

    primary_metric = select_primary_metric(g, runs, metric_filter)
    direction = g.config.metric_direction(primary_metric) if primary_metric else "max"
    lower_is_better = direction == "min"
    eff_map = {r.id: g.effective_run_metrics(r) for r in runs}
    if primary_metric:
        runs = [r for r in runs if primary_metric in eff_map[r.id]]

    def sort_key(node: ContourNode) -> tuple[int, float, float]:
        val = eff_map[node.id].get(primary_metric or "")
        if val is None:
            return (1, 0.0, float("inf"))
        q_key = float(val) if lower_is_better else -float(val)
        c_key = float(node.gpu_hours) if node.gpu_hours is not None else float("inf")
        return (0, q_key, c_key)

    runs.sort(key=sort_key)

    pareto_ids: set[str] = set()
    best_cost_so_far = float("inf")
    for r in runs:
        cost = r.gpu_hours
        val = eff_map[r.id].get(primary_metric or "")
        if val is not None and cost is not None and cost < best_cost_so_far:
            pareto_ids.add(r.id)
            best_cost_so_far = cost
    return primary_metric, direction, runs, pareto_ids


def find_review_items(g: ContourGraph) -> dict[str, Any]:
    """Tier-3 selective human review inbox (surfaced by `contour brief`)."""
    awaiting_signoff: list[dict[str, Any]] = []
    stale_verified: list[dict[str, Any]] = []

    for claim in sorted(g.claims.values(), key=lambda c: c.id):
        eff = effective_verification(g, claim)
        if claim.verification == "agent_audited" and eff == "agent_audited":
            awaiting_signoff.append({
                "id": claim.id,
                "status": claim.status,
                "title": claim.title,
                "audit_note": claim.get("audit_note"),
                "supports": claim.refs("supports"),
                "contradicts": claim.refs("contradicts"),
                "qualifies": claim.refs("qualifies"),
            })
        elif is_verification_stale(g, claim):
            has_contradicts = bool(claim.refs("contradicts"))
            stale_verified.append({
                "id": claim.id,
                "status": claim.status,
                "stored_verification": claim.verification,
                "title": claim.title,
                "has_contradicts": has_contradicts,
                "contradicts": claim.refs("contradicts"),
            })

    stale_verified.sort(key=lambda item: (not item["has_contradicts"], item["id"]))

    usable = g.usable_runs()
    dirty_frontier_seen: set[str] = set()
    dirty_frontier: list[dict[str, Any]] = []
    for fm_metric in frontier_metrics(g, usable):
        metric, _, runs, pareto_ids = compute_frontier(g, metric_filter=fm_metric)
        for r in runs:
            if r.id in pareto_ids and bool(r.code.get("dirty")) and r.id not in dirty_frontier_seen:
                dirty_frontier_seen.add(r.id)
                eff_m = g.effective_run_metrics(r)
                dirty_frontier.append({
                    "id": r.id,
                    "title": r.title,
                    "metric": metric,
                    "value": eff_m.get(metric or ""),
                    "gpu_hours": r.gpu_hours,
                    "commit": r.code.get("commit"),
                })

    return {
        "awaiting_signoff": awaiting_signoff,
        "stale_verified": stale_verified,
        "dirty_frontier": dirty_frontier,
    }
