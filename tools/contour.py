#!/usr/bin/env python3
"""Contour CLI: Navigate, survey, and reason over your ML Epistemic Graph.

Folder Hierarchy:
  expeditions/expd_*.md   Macro research goals & campaigns
  cairns/crn_*.md         Falsifiable claims, hypotheses, & principles
  fieldlogs/log_*.md      Empirical experiment cards distilled from Aim (`Theodolite`)

Primary Subcommands:
  contour map                      Print the Expedition -> Cairn -> FieldLog topography
  contour inspect <id>             Zoom in on any Expedition, Cairn, or FieldLog node
  contour ridge [--eval DATASET]   Compare verified fieldlogs & plot the Pareto ridgeline
  contour bearings [--expd ID]     Run the 6 Scientific Moves to propose next experiments
  contour drift                    Check for stale cairns or unlinked/invalidated fieldlogs
  contour sync [--from-json ...]   Sync Layer 1 Aim runs (or JSON bundles) into fieldlogs/
  contour new <type> <slug>        Scaffold a new expedition, cairn, or fieldlog node
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import datetime as dt
import json
from pathlib import Path
import re
from typing import Any
import yaml


WIKI_RE = re.compile(r"\[\[([^\[\]]+)\]\]")


def strip_wiki(val: Any) -> str:
    s = str(val).strip()
    m = WIKI_RE.search(s)
    return m.group(1).strip() if m else s


def strip_wiki_list(vals: Any) -> list[str]:
    if not vals:
        return []
    if isinstance(vals, str):
        return [strip_wiki(vals)]
    return [strip_wiki(v) for v in vals if v]


def as_wikilink(node_id: str) -> str:
    return f"[[{strip_wiki(node_id)}]]"


def parse_markdown_file(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            fm = yaml.safe_load(parts[0][4:]) or {}
            return fm, parts[1]
    return {}, text


def write_markdown_file(path: Path, frontmatter: dict[str, Any], body: str) -> None:
    fm_str = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()
    body_clean = body.lstrip("\n")
    content = f"---\n{fm_str}\n---\n\n{body_clean}"
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


@dataclass
class ContourNode:
    id: str
    type: str  # "expedition" | "cairn" | "fieldlog"
    title: str
    status: str
    path: Path
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def tldr(self) -> str:
        if "tldr" in self.frontmatter:
            return str(self.frontmatter["tldr"]).strip()
        for heading, text in self.sections.items():
            if any(
                k in heading.lower()
                for k in ("takeaway", "tldr", "proposition", "north star")
            ):
                first_line = next(
                    (ln.strip("- *").strip() for ln in text.splitlines() if ln.strip()),
                    "",
                )
                if first_line:
                    return first_line
        first_para = next(
            (
                ln.strip("- *").strip()
                for ln in self.body.splitlines()
                if ln.strip() and not ln.startswith("#")
            ),
            "",
        )
        return first_para[:140]


class ContourGraph:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.expeditions: dict[str, ContourNode] = {}
        self.cairns: dict[str, ContourNode] = {}
        self.fieldlogs: dict[str, ContourNode] = {}
        self.all_nodes: dict[str, ContourNode] = {}
        self._load()

    def _parse_node(self, path: Path, default_type: str) -> ContourNode:
        fm, body = parse_markdown_file(path)
        node_id = str(fm.get("id", path.stem))
        node_type = str(fm.get("type", default_type))
        title = str(fm.get("title", node_id))
        status = str(fm.get("status", fm.get("validity", "unknown")))

        sections: dict[str, str] = {}
        current_heading = "Overview"
        buf: list[str] = []
        for line in body.splitlines():
            if line.startswith("## "):
                if buf:
                    sections[current_heading] = "\n".join(buf).strip()
                current_heading = line[3:].strip()
                buf = []
            else:
                buf.append(line)
        if buf:
            sections[current_heading] = "\n".join(buf).strip()

        return ContourNode(
            id=node_id,
            type=node_type,
            title=title,
            status=status,
            path=path,
            frontmatter=fm,
            body=body.strip(),
            sections=sections,
        )

    def _load(self) -> None:
        for folder, ntype, store in [
            ("expeditions", "expedition", self.expeditions),
            ("cairns", "cairn", self.cairns),
            ("fieldlogs", "fieldlog", self.fieldlogs),
        ]:
            dpath = self.root / folder
            if not dpath.exists():
                continue
            for md_file in sorted(dpath.glob("*.md")):
                node = self._parse_node(md_file, ntype)
                store[node.id] = node
                self.all_nodes[node.id] = node

    def cairn_expeditions(self, cairn: ContourNode) -> list[str]:
        expds = set(strip_wiki_list(cairn.frontmatter.get("expeditions")))
        for e in self.expeditions.values():
            if cairn.id in strip_wiki_list(e.frontmatter.get("cairns")):
                expds.add(e.id)
        return sorted(expds)

    def expedition_cairns(self, expd_id: str) -> list[ContourNode]:
        enode = self.expeditions.get(expd_id)
        linked_ids: set[str] = set()
        if enode:
            linked_ids.update(strip_wiki_list(enode.frontmatter.get("cairns")))
        for c in self.cairns.values():
            if expd_id in self.cairn_expeditions(c):
                linked_ids.add(c.id)
        return [self.cairns[cid] for cid in sorted(linked_ids) if cid in self.cairns]


# ---------------------------------------------------------------------------
# Subcommand Implementations
# ---------------------------------------------------------------------------


def cmd_map(g: ContourGraph) -> None:
    print("=== CONTOUR TOPOGRAPHICAL MAP ===")
    print(
        f"Surveyed: {len(g.expeditions)} Expedition(s) | {len(g.cairns)} Cairn(s) | {len(g.fieldlogs)} FieldLog(s)\n"
    )

    covered_cairns: set[str] = set()
    for e in g.expeditions.values():
        print(f"[EXPEDITION] {e.id} ({e.status.upper()}): {e.title}")
        if e.tldr:
            print(f"  North Star: {e.tldr}")
        e_cairns = g.expedition_cairns(e.id)
        if not e_cairns:
            print("  (No linked cairns yet)")
        for c in e_cairns:
            covered_cairns.add(c.id)
            sup = strip_wiki_list(c.frontmatter.get("supports"))
            con = strip_wiki_list(c.frontmatter.get("contradicts"))
            qual = strip_wiki_list(c.frontmatter.get("qualifies"))
            drift_flag = (
                " [DRIFT / NEEDS SYNTHESIS]"
                if c.frontmatter.get("needs_synthesis")
                else ""
            )
            print(f"  └── [CAIRN] {c.id} [{c.status}]{drift_flag}: {c.title}")
            print(
                f"        FieldLogs: +{len(sup)} supports | -{len(con)} contradicts | ~{len(qual)} qualifies"
            )
        print()

    orphan_cairns = [c for cid, c in g.cairns.items() if cid not in covered_cairns]
    if orphan_cairns:
        print("[UNASSIGNED CAIRNS]")
        for c in orphan_cairns:
            print(f"  - {c.id} [{c.status}]: {c.title}")
        print()


def cmd_inspect(g: ContourGraph, node_id: str) -> None:
    nid = strip_wiki(node_id)
    node = g.all_nodes.get(nid)
    if not node:
        raise SystemExit(f"Node {nid!r} not found in Contour graph.")

    print(f"=== [{node.type.upper()}] {node.id}: {node.title} ===")
    print(f"Path:   {node.path.relative_to(g.root)}")
    print(f"Status: {node.status}")

    if node.type == "fieldlog":
        fm = node.frontmatter
        print(
            f"Task:   {fm.get('task')} | Arch: {fm.get('arch_family')} ({fm.get('arch_scale')})"
        )
        print(
            f"Train:  {fm.get('dataset_train')} | Eval: {fm.get('dataset_eval')}"
        )
        print(
            f"GPU hr: {fm.get('compute_gpu_hours')} | Commit: {fm.get('git_commit')}"
        )
        print(f"Intervention: {fm.get('intervention')}")
        metrics = fm.get("metrics") or {}
        parents = strip_wiki_list(fm.get("parent_runs"))
        if metrics:
            print("\nMetrics (with delta vs first parent if available):")
            p_metrics = (
                g.fieldlogs[parents[0]].frontmatter.get("metrics", {})
                if parents and parents[0] in g.fieldlogs
                else {}
            )
            for k, v in metrics.items():
                delta_str = ""
                if (
                    k in p_metrics
                    and isinstance(v, (int, float))
                    and isinstance(p_metrics[k], (int, float))
                ):
                    diff = v - p_metrics[k]
                    sign = "+" if diff >= 0 else ""
                    delta_str = f"  ({sign}{diff:.4g} vs {parents[0]})"
                print(f"  - {k}: {v}{delta_str}")

    elif node.type == "cairn":
        for edge_type in ("supports", "contradicts", "qualifies"):
            e_ids = strip_wiki_list(node.frontmatter.get(edge_type))
            if e_ids:
                print(f"\n{edge_type.upper()} ({len(e_ids)}):")
                for eid in e_ids:
                    enode = g.fieldlogs.get(eid)
                    if enode:
                        print(
                            f"  - {eid} [{enode.status}] ({enode.frontmatter.get('arch_family')} {enode.frontmatter.get('arch_scale')}): {enode.tldr}"
                        )
                    else:
                        print(f"  - {eid} (missing fieldlog)")

    elif node.type == "expedition":
        e_cairns = g.expedition_cairns(node.id)
        print(f"\nLINKED CAIRNS ({len(e_cairns)}):")
        for c in e_cairns:
            print(f"  - {c.id} [{c.status}]: {c.title}")

    print("\n--- Field Notes ---")
    print(node.body)


def find_drift_issues(g: ContourGraph) -> list[str]:
    issues: list[str] = []
    for c in g.cairns.values():
        if c.frontmatter.get("needs_synthesis"):
            issues.append(
                f"[STALE CAIRN] {c.id} has `needs_synthesis: true` -> review linked fieldlogs and set to false."
            )
        for edge_key in ("supports", "contradicts", "qualifies"):
            for lid in strip_wiki_list(c.frontmatter.get(edge_key)):
                log = g.fieldlogs.get(lid)
                if log and log.status != "verified":
                    issues.append(
                        f"[INVALIDATED EVIDENCE] {c.id} ({edge_key}) cites {lid}, whose validity is {log.status!r}."
                    )

    for log in g.fieldlogs.values():
        if log.status != "verified":
            continue
        for cid in strip_wiki_list(log.frontmatter.get("motivated_by")):
            cairn = g.cairns.get(cid)
            if not cairn:
                continue
            linked = (
                strip_wiki_list(cairn.frontmatter.get("supports"))
                + strip_wiki_list(cairn.frontmatter.get("contradicts"))
                + strip_wiki_list(cairn.frontmatter.get("qualifies"))
            )
            if log.id not in linked:
                issues.append(
                    f"[UNLINKED FIELDLOG] {log.id} was motivated_by {cid}, but {cid} has not placed it under supports/contradicts/qualifies."
                )
    return issues


def cmd_drift(g: ContourGraph) -> None:
    print("=== MOVE 0: TERRAIN DRIFT & HYGIENE CHECK ===")
    issues = find_drift_issues(g)
    if not issues:
        print("All cairns and fieldlog links are solid and up-to-date!")
    else:
        for item in issues:
            print(f" - {item}")


def cmd_ridge(
    g: ContourGraph,
    eval_ds: str | None = None,
    task: str | None = None,
    metric_filter: str | None = None,
) -> None:
    runs = [e for e in g.fieldlogs.values() if e.status == "verified"]
    if eval_ds:
        runs = [
            e
            for e in runs
            if eval_ds in [str(x) for x in (e.frontmatter.get("dataset_eval") or [])]
        ]
    if task:
        runs = [e for e in runs if e.frontmatter.get("task") == task]

    if not runs:
        print("No matching verified fieldlogs found.")
        return

    primary_metric = metric_filter
    if not primary_metric:
        for r in runs:
            mkeys = list((r.frontmatter.get("metrics") or {}).keys())
            if eval_ds:
                matching = [k for k in mkeys if eval_ds in k]
                if matching:
                    primary_metric = matching[0]
                    break
            if mkeys:
                primary_metric = mkeys[0]
                break

    lower_is_better = bool(
        primary_metric
        and any(
            tok in primary_metric.lower()
            for tok in ("loss", "err", "perplexity", "ppl", "mse", "mae")
        )
    )

    def sort_key(exp: ContourNode) -> float:
        val = (exp.frontmatter.get("metrics") or {}).get(primary_metric or "", None)
        if val is None:
            return float("inf") if lower_is_better else float("-inf")
        return float(val) if lower_is_better else -float(val)

    runs.sort(key=sort_key)

    pareto_ids: set[str] = set()
    best_cost_so_far = float("inf")
    for r in runs:
        cost = r.frontmatter.get("compute_gpu_hours")
        val = (r.frontmatter.get("metrics") or {}).get(primary_metric or "")
        if val is not None and isinstance(cost, (int, float)):
            if cost < best_cost_so_far:
                pareto_ids.add(r.id)
                best_cost_so_far = float(cost)

    print(
        f"=== PARETO RIDGELINE (Metric: {primary_metric or 'N/A'} | FieldLogs: {len(runs)}) ==="
    )
    print(
        f"{'FIELDLOG ID':<20} {'ARCH':<12} {'GPU_HR':<8} {'METRIC':<12} {'RIDGE':<7} {'INTERVENTION'}"
    )
    print("-" * 95)
    for r in runs:
        fm = r.frontmatter
        arch_str = f"{fm.get('arch_family')}-{fm.get('arch_scale')}"
        gpu_hr = fm.get("compute_gpu_hours")
        gpu_str = f"{gpu_hr:.1f}" if isinstance(gpu_hr, (int, float)) else "-"
        val = (fm.get("metrics") or {}).get(primary_metric or "", "-")
        ridge_mark = "  ▲" if r.id in pareto_ids else ""
        interv = str(fm.get("intervention", ""))[:38]
        print(
            f"{r.id:<20} {arch_str:<12} {gpu_str:<8} {str(val):<12} {ridge_mark:<7} {interv}"
        )


def cmd_bearings(g: ContourGraph, expd_id: str | None = None) -> None:
    """Run all 6 First-Principles Scientific Move detectors."""
    print("=== CONTOUR BEARINGS: NEXT EXPERIMENT CANDIDATES ===\n")

    target_cairns = (
        g.expedition_cairns(strip_wiki(expd_id))
        if expd_id
        else list(g.cairns.values())
    )

    # Move 0: Hygiene / Drift
    drift = find_drift_issues(g)
    print(f"[Move 0: Terrain Drift & Hygiene] ({len(drift)} item(s))")
    for d in drift[:5]:
        print(f"  * {d}")
    if not drift:
        print("  * All cairns and fieldlog links are clean.")
    print()

    # Move 1: Dialectics (Contradictions & Anomalies)
    print("[Move 1: Dialectics — Contradiction & Anomaly Resolution (O+ vs O-)]")
    m1_count = 0
    for c in target_cairns:
        con = strip_wiki_list(c.frontmatter.get("contradicts"))
        qual = strip_wiki_list(c.frontmatter.get("qualifies"))
        if con or qual or c.status == "contested":
            m1_count += 1
            print(
                f"  * {c.id} [{c.status}]: {c.title}\n"
                f"    -> Contradicted by: {con or 'none'} | Qualified by: {qual or 'none'}\n"
                f"    -> Action: Diff configs/regimes of supporting logs vs {con + qual} to isolate the confounder."
            )
    if m1_count == 0:
        print("  * No active contradictions detected.")
    print()

    # Move 2: Deduction (Regime Expansion & Scaling: R -> R')
    print("[Move 2: Deduction — Terra Incognita & Scale Expansion (R -> R')]")
    m2_count = 0
    for c in target_cairns:
        if c.status not in ("working_assumption", "well_supported"):
            continue
        sup_ids = strip_wiki_list(c.frontmatter.get("supports"))
        sup_runs = [g.fieldlogs[lid] for lid in sup_ids if lid in g.fieldlogs]
        tasks = {
            r.frontmatter.get("task")
            for r in sup_runs
            if r.frontmatter.get("task")
        }
        scales = {
            r.frontmatter.get("arch_scale")
            for r in sup_runs
            if r.frontmatter.get("arch_scale")
        }
        untested_lines = [
            ln.strip()
            for ln in c.body.splitlines()
            if "untested" in ln.lower() or "open:" in ln.lower()
        ]
        if len(tasks) <= 1 or len(scales) <= 1 or untested_lines:
            m2_count += 1
            print(
                f"  * {c.id} [{c.status}]: verified only on tasks={sorted(tasks)}, scales={sorted(scales)}"
            )
            if untested_lines:
                print(f"    -> Uncharted regime: {untested_lines[0]}")
    if m2_count == 0:
        print("  * All supported cairns span multiple tasks and scales.")
    print()

    # Move 3: Synthesis (Compositional Combination: I_1 + I_2)
    print(
        "[Move 3: Synthesis — Compositional Combination of Orthogonal Wins (I1 + I2)]"
    )

    def _is_pareto_win(child: ContourNode, parent: ContourNode) -> bool:
        c_metrics = child.frontmatter.get("metrics") or {}
        p_metrics = parent.frontmatter.get("metrics") or {}
        c_cost = child.frontmatter.get("compute_gpu_hours")
        p_cost = parent.frontmatter.get("compute_gpu_hours")
        for k, c_val in c_metrics.items():
            if k not in p_metrics or "throughput" in k:
                continue
            p_val = p_metrics[k]
            if not isinstance(c_val, (int, float)) or not isinstance(
                p_val, (int, float)
            ):
                continue
            lower_better = any(
                t in k.lower() for t in ("loss", "err", "ppl", "mse")
            )
            if not lower_better:
                if c_val > p_val:
                    return True
                if (
                    isinstance(c_cost, (int, float))
                    and isinstance(p_cost, (int, float))
                    and c_cost < p_cost * 0.9
                    and (p_val - c_val) <= 0.5
                ):
                    return True
        return False

    by_parent: dict[str, list[ContourNode]] = {}
    for log in g.fieldlogs.values():
        if log.status != "verified":
            continue
        for p in strip_wiki_list(log.frontmatter.get("parent_runs")):
            if p in g.fieldlogs and _is_pareto_win(log, g.fieldlogs[p]):
                by_parent.setdefault(p, []).append(log)

    m3_count = 0
    for parent_id, children in by_parent.items():
        if len(children) < 2:
            continue
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                c1, c2 = children[i], children[j]
                shared_evals = set(
                    c1.frontmatter.get("dataset_eval") or []
                ) & set(c2.frontmatter.get("dataset_eval") or [])
                if shared_evals and c1.frontmatter.get(
                    "intervention"
                ) != c2.frontmatter.get("intervention"):
                    m3_count += 1
                    print(
                        f"  * Combine {c1.id} ({c1.frontmatter.get('intervention')})\n"
                        f"        + {c2.id} ({c2.frontmatter.get('intervention')})\n"
                        f"    -> Both branched from {parent_id} on {sorted(shared_evals)}."
                    )
    if m3_count == 0:
        print("  * No uncombined sibling interventions found.")
    print()

    # Move 4: Occam's Razor (Subtractive Ablation / Simplification: I \ i_k)
    print(
        "[Move 4: Occam's Razor — Subtractive Ablation of Bundled Interventions (I \\ i_k)]"
    )
    all_parents = {
        p
        for e in g.fieldlogs.values()
        for p in strip_wiki_list(e.frontmatter.get("parent_runs"))
    }
    m4_count = 0
    for log in g.fieldlogs.values():
        if log.status != "verified":
            continue
        interv = str(log.frontmatter.get("intervention", ""))
        is_compound = any(
            sep in interv for sep in (" + ", " and ", " with ", ", ")
        )
        if is_compound and log.id not in all_parents:
            m4_count += 1
            print(
                f"  * {log.id} introduced compound intervention: {interv!r}\n"
                f"    -> Has no leave-one-out child ablations yet. Propose stripping components to test necessity."
            )
    if m4_count == 0:
        print("  * No unablated compound interventions detected.")
    print()

    # Move 5: Abduction (Bottleneck & Failure Diagnosis on Ridgeline Runs)
    print(
        "[Move 5: Abduction — Ridgeline Bottleneck & Anomaly Diagnosis (Errors(O) -> C_new)]"
    )
    eval_sets: set[str] = set()
    for log in g.fieldlogs.values():
        if log.status == "verified":
            eval_sets.update(
                str(x) for x in (log.frontmatter.get("dataset_eval") or [])
            )
    for eds in sorted(eval_sets):
        matching = [
            e
            for e in g.fieldlogs.values()
            if e.status == "verified"
            and eds in [str(x) for x in (e.frontmatter.get("dataset_eval") or [])]
        ]
        if matching:
            latest_or_best = matching[-1]
            obs = latest_or_best.sections.get(
                "Observations & Anomalies", ""
            ).strip()
            first_obs = next(
                (ln.strip() for ln in obs.splitlines() if ln.strip()),
                "See Aim curves",
            )
            print(
                f"  * Benchmark `{eds}` -> Inspect {latest_or_best.id} ({latest_or_best.title})\n"
                f"    -> Anomaly/Observation hook: {first_obs}"
            )


# ---------------------------------------------------------------------------
# Syncing Aim Runs / JSON Bundles -> `fieldlogs/log_*.md`
# ---------------------------------------------------------------------------


def _flag_cairns_stale(
    root: Path, cairn_ids: list[str], reason: str
) -> list[str]:
    cairns_dir = root / "cairns"
    flagged: list[str] = []
    for raw_id in cairn_ids:
        cid = strip_wiki(raw_id)
        cpath = cairns_dir / f"{cid}.md"
        if not cpath.exists():
            continue
        fm, body = parse_markdown_file(cpath)
        if not fm.get("needs_synthesis", False):
            fm["needs_synthesis"] = True
            write_markdown_file(cpath, fm, body)
            flagged.append(f"{cid} ({reason})")
    return flagged


def ingest_fieldlog_record(
    root: Path,
    aim_hash: str,
    meta: dict[str, Any],
    metrics: dict[str, float],
) -> tuple[Path, list[str]]:
    log_dir = root / "fieldlogs"
    log_dir.mkdir(parents=True, exist_ok=True)

    created_str = str(meta.get("created_at") or dt.date.today().isoformat())
    date_compact = created_str.replace("-", "")[:8]
    short_hash = aim_hash[:6]
    log_id = f"log_{date_compact}_{short_hash}"

    existing_path: Path | None = None
    for candidate in log_dir.glob("log_*.md"):
        fm_cand, _ = parse_markdown_file(candidate)
        if fm_cand.get("aim_hash") == aim_hash or fm_cand.get("id") == log_id:
            existing_path = candidate
            log_id = fm_cand.get("id", log_id)
            break

    target_path = existing_path or (log_dir / f"{log_id}.md")
    old_fm, existing_body = (
        parse_markdown_file(target_path) if target_path.exists() else ({}, "")
    )

    dataset_meta = meta.get("dataset", {})
    arch_meta = meta.get("arch", {})
    compute_meta = meta.get("compute", {})
    exec_meta = meta.get("execution", {})
    git_meta = meta.get("git", {})
    lineage_meta = meta.get("lineage", {})
    checkpoints = meta.get("checkpoints", {})

    parent_runs = [
        as_wikilink(p) for p in lineage_meta.get("parent_runs", []) if p
    ]
    motivated_by = [
        as_wikilink(c) for c in lineage_meta.get("motivated_by", []) if c
    ]

    new_validity = meta.get("validity", old_fm.get("validity", "verified"))
    old_validity = old_fm.get("validity")

    frontmatter: dict[str, Any] = {
        "id": log_id,
        "type": "fieldlog",
        "title": meta.get("title", old_fm.get("title", log_id)),
        "aim_hash": aim_hash,
        "created_at": created_str,
        "validity": new_validity,
        "task": meta.get("task", "unspecified"),
        "dataset_train": dataset_meta.get("train", []),
        "dataset_eval": dataset_meta.get("eval", []),
        "arch_family": arch_meta.get("family", "unspecified"),
        "arch_scale": arch_meta.get("scale", "unspecified"),
        "compute_gpu_hours": compute_meta.get("gpu_hours"),
        "intervention": lineage_meta.get("intervention", "baseline"),
        "git_commit": git_meta.get("commit"),
        "dirty_tree": bool(git_meta.get("dirty_tree", False)),
        "launch_cmd": exec_meta.get("launch_cmd"),
        "slurm_job_id": exec_meta.get("slurm_job_id"),
        "checkpoint_uri": checkpoints.get("best")
        or next(iter(checkpoints.values()), None),
        "parent_runs": parent_runs,
        "motivated_by": motivated_by,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
    }

    if not existing_body.strip():
        parent_ref = parent_runs[0] if parent_runs else "baseline"
        existing_body = (
            "## Key Takeaway (TL;DR)\n"
            f"- TODO: Summarize key outcome vs {parent_ref}.\n\n"
            "## Observations & Anomalies\n"
            "- Training dynamics / loss curve notes:\n"
        )

    write_markdown_file(target_path, frontmatter, existing_body)

    flagged: list[str] = []
    if not existing_path:
        flagged.extend(
            _flag_cairns_stale(root, motivated_by, f"new fieldlog {log_id}")
        )
    elif old_validity != new_validity:
        cairns_dir = root / "cairns"
        referencing = list(motivated_by)
        if cairns_dir.exists():
            for cpath in cairns_dir.glob("crn_*.md"):
                cfm, _ = parse_markdown_file(cpath)
                for edge_key in ("supports", "contradicts", "qualifies"):
                    edges = strip_wiki_list(cfm.get(edge_key))
                    if log_id in edges:
                        referencing.append(cfm.get("id", cpath.stem))
        flagged.extend(
            _flag_cairns_stale(
                root,
                referencing,
                f"validity of {log_id} changed {old_validity} -> {new_validity}",
            )
        )

    return target_path, flagged


def cmd_sync(
    root: Path, aim_repo_path: str | None, json_paths: list[Path] | None
) -> None:
    if json_paths:
        for jpath in json_paths:
            payload = json.loads(jpath.read_text(encoding="utf-8"))
            card_path, flagged = ingest_fieldlog_record(
                root=root,
                aim_hash=payload["aim_hash"],
                meta=payload["meta"],
                metrics=payload.get("metrics", {}),
            )
            print(
                f"[contour sync] Logged {card_path.relative_to(root)} from {jpath.name}"
            )
            for fc in flagged:
                print(f"  -> Flagged cairn needs_synthesis: {fc}")
        return

    try:
        import aim  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Error: `aim` package is not installed. Use `--from-json` or `pip install aim`."
        ) from exc

    repo = aim.Repo.from_path(aim_repo_path or str(root))
    synced = 0
    for run in repo.iter_runs():
        meta = run.get("meta")
        if not meta:
            continue
        metrics_dict: dict[str, float] = {}
        for metric in run.metrics():
            subset = (metric.context or {}).get("subset")
            key = f"{subset}/{metric.name}" if subset else metric.name
            _, vals = metric.values.sparse_numpy()
            if len(vals) > 0:
                metrics_dict[key] = float(vals[-1])
        card_path, flagged = ingest_fieldlog_record(
            root=root,
            aim_hash=run.hash,
            meta=meta,
            metrics=metrics_dict,
        )
        synced += 1
        print(f"[contour sync] Logged {card_path.relative_to(root)}")
        for fc in flagged:
            print(f"  -> Flagged cairn needs_synthesis: {fc}")
    print(f"[contour sync] Done ({synced} fieldlog(s) processed).")


def cmd_new(g: ContourGraph, node_type: str, slug: str, title: str) -> None:
    today = dt.date.today().isoformat()
    if node_type == "expedition":
        node_id = slug if slug.startswith("expd_") else f"expd_{slug}"
        path = g.root / "expeditions" / f"{node_id}.md"
        content = f"""---
id: {node_id}
type: expedition
title: "{title}"
status: active
cairns: []
---

## North Star & Motivation
- State the overarching research problem and summit criteria.

## Current State & Open Questions
- 
"""
    elif node_type == "cairn":
        node_id = slug if slug.startswith("crn_") else f"crn_{slug}"
        path = g.root / "cairns" / f"{node_id}.md"
        content = f"""---
id: {node_id}
type: cairn
title: "{title}"
status: speculative
needs_synthesis: false
expeditions: []
parent_cairns: []
supports: []
contradicts: []
qualifies: []
---

## Proposition & Intuition
- State the falsifiable hypothesis or principle and the conceptual mechanism behind it.

## Where This Holds vs. Breaks (Scope & Regimes)
- **Verified:** 
- **Breaks / Fails:** 
- **Untested / Open:** 
"""
    elif node_type == "fieldlog":
        date_str = today.replace("-", "")
        node_id = slug if slug.startswith("log_") else f"log_{date_str}_{slug}"
        path = g.root / "fieldlogs" / f"{node_id}.md"
        content = f"""---
id: {node_id}
type: fieldlog
title: "{title}"
aim_hash: "manual_{slug}"
created_at: {today}
validity: verified
task: unspecified
dataset_train: []
dataset_eval: []
arch_family: unspecified
arch_scale: unspecified
compute_gpu_hours: null
intervention: "baseline"
git_commit: null
dirty_tree: false
launch_cmd: null
parent_runs: []
motivated_by: []
metrics: {{}}
---

## Key Takeaway (TL;DR)
- 

## Observations & Anomalies
- 
"""
    else:
        raise SystemExit(f"Unknown node type: {node_type}")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"File already exists: {path}")
    path.write_text(content, encoding="utf-8")
    print(f"Created {path.relative_to(g.root)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="contour",
        description="Contour: Navigate, survey, and reason over your ML Epistemic Graph.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to contour root directory",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "map", aliases=["overview"], help="Print the Expedition -> Cairn -> FieldLog DAG"
    )

    p_insp = sub.add_parser(
        "inspect", help="Zoom in on an Expedition, Cairn, or FieldLog node"
    )
    p_insp.add_argument(
        "node_id", type=str, help="ID of expedition (expd_*), cairn (crn_*), or fieldlog (log_*)"
    )

    sub.add_parser(
        "drift",
        aliases=["stale"],
        help="Move 0: Check for stale cairns or unlinked/invalidated fieldlogs",
    )

    p_ridge = sub.add_parser(
        "ridge",
        aliases=["leaderboard"],
        help="Compare verified fieldlogs & plot the Pareto ridgeline",
    )
    p_ridge.add_argument(
        "--eval", dest="eval_ds", type=str, default=None, help="Filter by eval dataset"
    )
    p_ridge.add_argument("--task", type=str, default=None, help="Filter by task")
    p_ridge.add_argument(
        "--metric", type=str, default=None, help="Primary metric key to sort by"
    )

    p_bear = sub.add_parser(
        "bearings",
        aliases=["candidates"],
        help="Run the 6 Scientific Moves to propose next experiments",
    )
    p_bear.add_argument(
        "--expedition", dest="expd_id", type=str, default=None, help="Filter by expedition ID"
    )

    p_sync = sub.add_parser(
        "sync", help="Distill Layer 1 Aim runs (or JSON field bundles) into fieldlogs/"
    )
    p_sync.add_argument(
        "--repo", type=str, default=None, help="Aim repo path or aim:// server URI"
    )
    p_sync.add_argument(
        "--from-json",
        type=Path,
        nargs="+",
        default=None,
        help="Ingest exported JSON field bundles",
    )

    p_new = sub.add_parser(
        "new", help="Scaffold a new expedition, cairn, or fieldlog node"
    )
    p_new.add_argument("node_type", choices=["expedition", "cairn", "fieldlog"])
    p_new.add_argument("slug", type=str, help="Short identifier slug")
    p_new.add_argument("--title", type=str, required=True, help="Human-readable title")

    args = parser.parse_args()

    if args.cmd == "sync":
        cmd_sync(args.root, args.repo, args.from_json)
        return

    g = ContourGraph(args.root)

    if args.cmd in ("map", "overview"):
        cmd_map(g)
    elif args.cmd == "inspect":
        cmd_inspect(g, args.node_id)
    elif args.cmd in ("drift", "stale"):
        cmd_drift(g)
    elif args.cmd in ("ridge", "leaderboard"):
        cmd_ridge(g, args.eval_ds, args.task, args.metric)
    elif args.cmd in ("bearings", "candidates"):
        cmd_bearings(g, args.expd_id)
    elif args.cmd == "new":
        cmd_new(g, args.node_type, args.slug, args.title)


if __name__ == "__main__":
    main()
