"""`contour drift`: hygiene, provenance, and confound checks over the graph.

Codes are stable so humans and agents can reference, filter, and fix them:
  H000-H010  graph integrity & synthesis hygiene
  H011-H012  empirical comparison checks (haze noise floor & confounded parent diffs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .compare import compute_vs_parent
from .graph import EDGE_KEYS, UNRESOLVED_PREFIX, ContourGraph, ContourNode
from .haze import load_haze
from .ingest import VALIDITY_VALUES

ERROR = "error"
WARN = "warn"


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    message: str
    nodes: tuple[str, ...] = field(default_factory=tuple)
    hint: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "nodes": list(self.nodes),
            "hint": self.hint,
        }


def get_or_compute_vs_parent(
    g: ContourGraph, log: ContourNode, haze: dict[str, dict[str, dict[str, Any]]] | None = None
) -> dict[str, Any] | None:
    """Return a fieldlog's `vs_parent` dict (from frontmatter or computed on-the-fly)."""
    parents = log.parents
    if not parents or parents[0] not in g.fieldlogs:
        return None
    parent_node = g.fieldlogs[parents[0]]
    return compute_vs_parent(
        g.root,
        log.frontmatter,
        parent_node.frontmatter,
        haze=haze,
        config=g.config,
    )


def find_drift_issues(g: ContourGraph) -> list[Issue]:
    issues: list[Issue] = []
    add = issues.append

    for err in g.load_errors:
        add(Issue("H000", ERROR, f"unparseable node: {err}", hint="Fix the YAML frontmatter."))
    for nid in sorted(set(g.duplicate_ids)):
        add(Issue("H000", ERROR, f"duplicate node id {nid!r}", (nid,), "Node ids must be unique across the graph."))

    haze = load_haze(g.root)

    # --- cairns ------------------------------------------------------------------
    _check_cairn_dag(g, add)
    for c in g.cairns.values():
        if c.get("needs_synthesis"):
            add(Issue(
                "H001", WARN, f"{c.id} has new evidence awaiting synthesis", (c.id,),
                "Review the linked fieldlogs, place them under supports/contradicts/qualifies, "
                "update the prose, then set `needs_synthesis: false`.",
            ))
        seen: dict[str, str] = {}
        for edge in EDGE_KEYS:
            for lid in c.refs(edge):
                if lid in seen and seen[lid] != edge:
                    add(Issue(
                        "H009", WARN, f"{c.id} lists {lid} under both {seen[lid]} and {edge}",
                        (c.id, lid), "Pick one relation; use `qualifies` for boundary conditions.",
                    ))
                seen[lid] = edge
                ev = g.fieldlogs.get(lid)
                if ev is None:
                    add(Issue("H004", ERROR, f"{c.id} ({edge}) cites missing fieldlog {lid}", (c.id, lid)))
                elif not ev.is_usable_evidence:
                    why = f"status={ev.status!r}" if ev.status != "final" else f"validity={ev.validity!r}"
                    add(Issue(
                        "H002", ERROR, f"{c.id} ({edge}) cites {lid}, which is not usable evidence ({why})",
                        (c.id, lid), "Remove the link or explain the exclusion in the cairn prose.",
                    ))
                elif edge == "supports":
                    vs = get_or_compute_vs_parent(g, ev, haze=haze)
                    if vs is not None and vs.get("iso_compute") != "reduced":
                        deltas = vs.get("deltas") or {}
                        q_deltas = [
                            d for k, d in deltas.items()
                            if not g.config.is_cost_metric(k) and "clears_noise" in d
                        ]
                        if q_deltas and all(d.get("clears_noise") is False for d in q_deltas):
                            add(Issue(
                                "H011", WARN,
                                f"{c.id} cites {lid} as `supports`, but its quality delta vs parent is within the noise floor",
                                (c.id, lid),
                                "Treat small deltas within seed noise as `qualifies` or run additional seeds.",
                            ))

        for ref_key, pool in (("expeditions", g.expeditions), ("parent_cairns", g.cairns)):
            for rid in c.refs(ref_key):
                if rid not in pool:
                    add(Issue("H004", ERROR, f"{c.id}.{ref_key} references missing node {rid}", (c.id, rid)))

    for e in g.expeditions.values():
        for cid in e.refs("cairns"):
            if cid not in g.cairns:
                add(Issue("H004", ERROR, f"{e.id}.cairns references missing cairn {cid}", (e.id, cid)))

    # --- fieldlogs ---------------------------------------------------------------
    linked_by: dict[str, set[str]] = {}
    for c in g.cairns.values():
        for edge in EDGE_KEYS:
            for lid in c.refs(edge):
                linked_by.setdefault(lid, set()).add(c.id)

    for log in g.fieldlogs.values():
        if log.validity not in VALIDITY_VALUES:
            add(Issue(
                "H007", WARN, f"{log.id} has unknown validity {log.validity!r}", (log.id,),
                f"Use one of {VALIDITY_VALUES}.",
            ))
        if not log.get("bundle") or not (g.root / str(log.get("bundle"))).exists():
            add(Issue(
                "H008", WARN, f"{log.id} has no archived field bundle; provenance is unverifiable",
                (log.id,), "Ingest runs via `contour sync` so the raw bundle is archived.",
            ))
        if log.status == "running":
            add(Issue(
                "H006", WARN, f"{log.id} is still `running`", (log.id,),
                "If the job died without finishing, re-sync a bundle with status `failed`.",
            ))
        for p in log.refs("parent_runs"):
            if p.startswith(UNRESOLVED_PREFIX):
                add(Issue(
                    "H005", WARN, f"{log.id} has unresolved parent {p}", (log.id,),
                    "Sync the parent's bundle; links resolve automatically.",
                ))
            elif p not in g.fieldlogs:
                add(Issue("H004", ERROR, f"{log.id}.parent_runs references missing fieldlog {p}", (log.id, p)))
        for ref in log.refs("motivated_by"):
            if ref not in g.all_nodes:
                add(Issue("H004", ERROR, f"{log.id}.motivated_by references missing node {ref}", (log.id, ref)))
            elif ref in g.cairns and log.is_usable_evidence and ref not in linked_by.get(log.id, set()):
                add(Issue(
                    "H003", WARN, f"{log.id} was motivated by {ref}, but {ref} hasn't placed it",
                    (log.id, ref), f"Use `contour link {log.id} {ref} --relation <supports|contradicts|qualifies>`.",
                ))

        if log.is_usable_evidence:
            vs = get_or_compute_vs_parent(g, log, haze=haze)
            if vs is not None and vs.get("single_logical_diff") is False:
                fdiff = vs.get("confounding_facets") or {}
                changed = sorted(fdiff) or ["interventions"]
                add(Issue(
                    "H012", WARN,
                    f"{log.id} vs { log.parents[0] } is confounded (single_logical_diff=false; changed: {changed})",
                    (log.id, log.parents[0]),
                    "Run a controlled single-diff experiment to isolate the confounding facet.",
                ))

    return issues


def _check_cairn_dag(g: ContourGraph, add: Any) -> None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}

    def dfs(nid: str) -> None:
        color[nid] = GRAY
        node = g.cairns.get(nid)
        if node:
            for pid in node.refs("parent_cairns"):
                if pid not in g.cairns:
                    continue
                if color.get(pid) == GRAY:
                    add(Issue("H010", ERROR, f"Cycle in cairn parent_cairns DAG involving {nid} -> {pid}", (nid, pid)))
                elif color.get(pid, WHITE) == WHITE:
                    dfs(pid)
        color[nid] = BLACK

    for cid in g.cairns:
        if color.get(cid, WHITE) == WHITE:
            dfs(cid)
