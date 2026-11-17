"""Contour CLI: Navigate, survey, and reason over your ML Epistemic Graph."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from .bundle import BundleError, SCHEMA_ID
from .compare import compare_runs
from .config import CONFIG_FILE, DEFAULT_CONFIG, SUBDIR_CANDIDATES, ConfigError, find_root, load_config
from .drift import ERROR, find_drift_issues, get_or_compute_vs_parent
from .graph import EDGE_KEYS, ContourGraph, ContourNode
from .haze import load_haze, seed_haze_from_runs
from .ingest import (
    DEFAULT_VALIDITY,
    FieldlogIndex,
    ingest_bundle,
    iter_bundle_paths,
    load_bundle_file,
    relink_parents,
)
from .mdio import as_wikilink, parse_markdown_file, strip_wiki, write_markdown_file


def _emit_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
# Subcommand Implementations
# ---------------------------------------------------------------------------


def _bundled_skill_markdown() -> str:
    repo_skill = Path(__file__).resolve().parent.parent.parent / ".agent" / "skills" / "contour-research" / "SKILL.md"
    if repo_skill.exists():
        return repo_skill.read_text(encoding="utf-8")
    return (
        "---\n"
        "name: contour-research\n"
        "description: Navigate, survey, and reason over the Contour ML Epistemic Graph (Expeditions, Cairns, and FieldLogs).\n"
        "---\n\n"
        "# Contour: Epistemic ML Research Copilot Skill\n\n"
        "Run `contour map --json`, `contour bearings --json`, `contour ridge --json`, `contour drift --json`, "
        "and `contour inspect <node_id>` to navigate the research state and propose next experiments.\n"
    )


def _eject_standalone_logger(dest: Path) -> Path:
    pkg_dir = Path(__file__).resolve().parent
    bundle_src = (pkg_dir / "bundle.py").read_text(encoding="utf-8")
    theo_src = (pkg_dir / "theodolite.py").read_text(encoding="utf-8")
    # Strip relative imports in theodolite.py so it runs as a single standalone stdlib-only file
    theo_lines = [
        ln for ln in theo_src.splitlines()
        if not ln.startswith("from .bundle import") and not ln.startswith("from .config import")
    ]
    combined = (
        '"""Standalone zero-dependency Contour Run logger (ejected via `contour init --eject-logger`)."""\n\n'
        + bundle_src
        + "\n\nclass ConfigError(ValueError):\n    pass\n\n"
        + "def find_root(start=None, explicit=None):\n"
        + "    cur = (start or Path.cwd()).resolve()\n"
        + "    for d in (cur, *cur.parents):\n"
        + "        if (d / 'contour.yaml').exists(): return d\n"
        + "        for sub in ('contour', '.contour', 'research'):\n"
        + "            if (d / sub / 'contour.yaml').exists(): return d / sub\n"
        + "    raise ConfigError('No contour.yaml found')\n\n"
        + "\n".join(theo_lines)
        + "\n"
    )
    out_path = (dest / "contour_run.py") if dest.is_dir() else dest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(combined, encoding="utf-8")
    return out_path


def cmd_init(
    target: Path | None,
    with_skill: bool = False,
    eject_logger: Path | None = None,
    as_json: bool = False,
) -> None:
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
        data = copy.deepcopy(DEFAULT_CONFIG)
        cfg_path.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        created_cfg = True

    for sub in ("expeditions", "cairns", "fieldlogs", "bundles", "field_bundles"):
        (resolved / sub).mkdir(exist_ok=True)

    skill_path: Path | None = None
    if with_skill:
        repo_base = resolved.parent if resolved.name in SUBDIR_CANDIDATES else resolved
        skill_path = repo_base / ".agent" / "skills" / "contour-research" / "SKILL.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(_bundled_skill_markdown(), encoding="utf-8")

    logger_path: Path | None = None
    if eject_logger is not None:
        logger_path = _eject_standalone_logger(eject_logger.resolve())

    if as_json:
        _emit_json({
            "root": str(resolved),
            "created_config": created_cfg,
            **({"skill_path": str(skill_path)} if skill_path else {}),
            **({"ejected_logger": str(logger_path)} if logger_path else {}),
        })
        return
    status = "Created" if created_cfg else "Initialized"
    print(f"{status} Contour root at {resolved}")
    if skill_path:
        print(f"Installed agent skill at {skill_path}")
    if logger_path:
        print(f"Ejected standalone stdlib logger at {logger_path}")


def cmd_map(g: ContourGraph, as_json: bool = False) -> None:
    covered_cairns: set[str] = set()
    expeditions_out: list[dict[str, Any]] = []
    for e in g.expeditions.values():
        e_cairns = g.expedition_cairns(e.id)
        cairn_list = []
        for c in e_cairns:
            covered_cairns.add(c.id)
            edges = g.cairn_edges(c)
            cairn_list.append({
                "id": c.id,
                "status": c.status,
                "title": c.title,
                "needs_synthesis": bool(c.get("needs_synthesis")),
                "counts": {k: len(v) for k, v in edges.items()},
            })
        expeditions_out.append({
            "id": e.id,
            "status": e.status,
            "title": e.title,
            "tldr": e.tldr,
            "cairns": cairn_list,
        })

    orphan_cairns = [
        {"id": c.id, "status": c.status, "title": c.title}
        for cid, c in g.cairns.items()
        if cid not in covered_cairns
    ]

    if as_json:
        _emit_json({
            "counts": {
                "expeditions": len(g.expeditions),
                "cairns": len(g.cairns),
                "fieldlogs": len(g.fieldlogs),
                "usable_fieldlogs": len(g.usable_fieldlogs()),
            },
            "expeditions": expeditions_out,
            "unassigned_cairns": orphan_cairns,
        })
        return

    print("=== CONTOUR TOPOGRAPHICAL MAP ===")
    print(
        f"Surveyed: {len(g.expeditions)} Expedition(s) | {len(g.cairns)} Cairn(s) | "
        f"{len(g.fieldlogs)} FieldLog(s) ({len(g.usable_fieldlogs())} usable)\n"
    )
    for e in expeditions_out:
        print(f"[EXPEDITION] {e['id']} ({e['status'].upper()}): {e['title']}")
        if e["tldr"]:
            print(f"  North Star: {e['tldr']}")
        if not e["cairns"]:
            print("  (No linked cairns yet)")
        for c in e["cairns"]:
            drift_flag = " [DRIFT / NEEDS SYNTHESIS]" if c["needs_synthesis"] else ""
            cnt = c["counts"]
            print(f"  └── [CAIRN] {c['id']} [{c['status']}]{drift_flag}: {c['title']}")
            print(
                f"        FieldLogs: +{cnt['supports']} supports | "
                f"-{cnt['contradicts']} contradicts | ~{cnt['qualifies']} qualifies"
            )
        print()

    if orphan_cairns:
        print("[UNASSIGNED CAIRNS]")
        for c in orphan_cairns:
            print(f"  - {c['id']} [{c['status']}]: {c['title']}")
        print()


def cmd_inspect(g: ContourGraph, node_id: str, as_json: bool = False) -> None:
    nid = strip_wiki(node_id)
    node = g.all_nodes.get(nid)
    if not node:
        raise SystemExit(f"Node {nid!r} not found in Contour graph.")

    haze = load_haze(g.root)
    vs_parent = get_or_compute_vs_parent(g, node, haze=haze) if node.type == "fieldlog" else None

    if as_json:
        fm_out = dict(node.frontmatter)
        if vs_parent is not None:
            fm_out["vs_parent"] = vs_parent
        _emit_json({
            "id": node.id,
            "type": node.type,
            "title": node.title,
            "status": node.status,
            "path": str(node.path.relative_to(g.root)),
            "frontmatter": fm_out,
            "body": node.body,
        })
        return

    print(f"=== [{node.type.upper()}] {node.id}: {node.title} ===")
    print(f"Path:   {node.path.relative_to(g.root)}")
    print(f"Status: {node.status}")

    if node.type == "fieldlog":
        fm = node.frontmatter
        print(f"Valid:  {node.validity} | Tier: {fm.get('tier', 'proxy')} | Seed: {fm.get('seed')}")
        print(f"Task:   {fm.get('task')} | Arch: {fm.get('arch_family')} ({fm.get('arch_scale')})")
        print(f"Train:  {fm.get('dataset_train')} | Eval: {fm.get('dataset_eval')}")
        print(f"GPU hr: {node.gpu_hours} | Commit: {fm.get('git_commit')} | Dirty: {fm.get('dirty_tree')}")
        print(f"Interventions: {sorted(node.interventions)}")
        if fm.get("intervention_note"):
            print(f"Note:   {fm.get('intervention_note')}")

        if vs_parent is not None:
            pid = strip_wiki(str(vs_parent.get("parent", "")))
            print(
                f"\nDiff vs Parent ({pid}) [Grade {vs_parent.get('grade')} | "
                f"single_logical_diff={vs_parent.get('single_logical_diff')} | "
                f"iso_compute={vs_parent.get('iso_compute')}]:"
            )
            if vs_parent.get("confounding_facets"):
                print(f"  CONFOUNDING FACETS: {vs_parent['confounding_facets']}")
            if vs_parent.get("added_interventions"):
                print(f"  Added interventions:   {vs_parent['added_interventions']}")
            if vs_parent.get("removed_interventions"):
                print(f"  Removed interventions: {vs_parent['removed_interventions']}")
            deltas = vs_parent.get("deltas") or {}
            for k, d in deltas.items():
                snr_str = f" | SNR={d['snr']}x" if d.get("snr") is not None else ""
                clr_str = " [CLEARS NOISE]" if d.get("clears_noise") else ""
                print(f"  - {k}: {d['parent']} -> {d['value']} ({d['delta']:+.4g}){snr_str}{clr_str}")
        elif node.metrics:
            print("\nMetrics:")
            for k, v in node.metrics.items():
                print(f"  - {k}: {v}")

    elif node.type == "cairn":
        for edge_type in EDGE_KEYS:
            e_ids = node.refs(edge_type)
            if e_ids:
                print(f"\n{edge_type.upper()} ({len(e_ids)}):")
                for eid in e_ids:
                    enode = g.fieldlogs.get(eid)
                    if enode:
                        vs = get_or_compute_vs_parent(g, enode, haze=haze)
                        badge = f" [vs parent: Grade {vs['grade']}]" if vs else ""
                        print(
                            f"  - {eid} [{enode.status}/{enode.validity}]{badge} "
                            f"({enode.get('arch_family')} {enode.get('arch_scale')}): {enode.tldr}"
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


def cmd_drift(g: ContourGraph, as_json: bool = False, strict: bool = False) -> None:
    issues = find_drift_issues(g)
    if as_json:
        _emit_json({"count": len(issues), "issues": [i.as_dict() for i in issues]})
    else:
        print("=== MOVE 0: TERRAIN DRIFT & HYGIENE CHECK ===")
        if not issues:
            print("All cairns and fieldlog links are solid and up-to-date!")
        else:
            for item in issues:
                print(f" - [{item.code}/{item.severity.upper()}] {item.message}")
                if item.hint:
                    print(f"     hint: {item.hint}")

    if strict and any(i.severity == ERROR for i in issues):
        raise SystemExit(1)


def _select_primary_metric(
    g: ContourGraph, runs: list[ContourNode], eval_ds: str | None, metric_filter: str | None
) -> str | None:
    if metric_filter:
        return metric_filter
    if eval_ds:
        for r in runs:
            for k in r.metrics:
                if eval_ds in k and not g.config.is_cost_metric(k):
                    return k
    for r in runs:
        for k in r.metrics:
            if not g.config.is_cost_metric(k):
                return k
    for r in runs:
        if r.metrics:
            return next(iter(r.metrics))
    return None


def _compute_ridge(
    g: ContourGraph,
    eval_ds: str | None = None,
    task: str | None = None,
    metric_filter: str | None = None,
) -> tuple[str | None, str, list[ContourNode], set[str]]:
    runs = g.usable_fieldlogs()
    if eval_ds:
        runs = [r for r in runs if eval_ds in r.evals]
    if task:
        runs = [r for r in runs if r.get("task") == task]
    if not runs:
        return None, "max", [], set()

    primary_metric = _select_primary_metric(g, runs, eval_ds, metric_filter)
    direction = g.config.metric_direction(primary_metric) if primary_metric else "max"
    lower_is_better = direction == "min"

    def sort_key(node: ContourNode) -> tuple[int, float]:
        val = node.metrics.get(primary_metric or "")
        if val is None:
            return (1, 0.0)
        return (0, float(val) if lower_is_better else -float(val))

    runs.sort(key=sort_key)

    pareto_ids: set[str] = set()
    best_cost_so_far = float("inf")
    for r in runs:
        cost = r.gpu_hours
        val = r.metrics.get(primary_metric or "")
        if val is not None and cost is not None and cost < best_cost_so_far:
            pareto_ids.add(r.id)
            best_cost_so_far = cost
    return primary_metric, direction, runs, pareto_ids


def cmd_ridge(
    g: ContourGraph,
    eval_ds: str | None = None,
    task: str | None = None,
    metric_filter: str | None = None,
    as_json: bool = False,
) -> None:
    primary_metric, direction, runs, pareto_ids = _compute_ridge(
        g, eval_ds=eval_ds, task=task, metric_filter=metric_filter
    )
    if as_json:
        _emit_json({
            "metric": primary_metric,
            "direction": direction,
            "runs": [
                {
                    "id": r.id,
                    "arch": f"{r.get('arch_family')}-{r.get('arch_scale')}",
                    "gpu_hours": r.gpu_hours,
                    "value": r.metrics.get(primary_metric or ""),
                    "on_ridge": r.id in pareto_ids,
                    "interventions": sorted(r.interventions),
                }
                for r in runs
            ],
        })
        return

    if not runs:
        print("No matching usable fieldlogs found.")
        return

    print(
        f"=== PARETO RIDGELINE (Metric: {primary_metric or 'N/A'} [{direction}] | FieldLogs: {len(runs)}) ==="
    )
    print(f"{'FIELDLOG ID':<22} {'ARCH':<12} {'GPU_HR':<8} {'METRIC':<12} {'RIDGE':<7} {'INTERVENTIONS'}")
    print("-" * 100)
    for r in runs:
        arch_str = f"{r.get('arch_family')}-{r.get('arch_scale')}"
        gpu_str = f"{r.gpu_hours:.1f}" if r.gpu_hours is not None else "-"
        val = r.metrics.get(primary_metric or "", "-")
        ridge_mark = "  ▲" if r.id in pareto_ids else ""
        interv = ", ".join(sorted(r.interventions)) if r.interventions else str(r.get("intervention_note") or "baseline")
        print(f"{r.id:<22} {arch_str:<12} {gpu_str:<8} {str(val):<12} {ridge_mark:<7} {interv[:40]}")


def _is_pareto_win(g: ContourGraph, child: ContourNode, parent: ContourNode) -> bool:
    c_metrics, p_metrics = child.metrics, parent.metrics
    c_cost, p_cost = child.gpu_hours, parent.gpu_hours
    for k, c_val in c_metrics.items():
        if k not in p_metrics or g.config.is_cost_metric(k):
            continue
        p_val = p_metrics[k]
        direction = g.config.metric_direction(k)
        if direction == "min":
            if c_val < p_val:
                return True
            rel_tol = max(abs(p_val) * 0.01, 1e-6)
            if c_cost is not None and p_cost is not None and c_cost < p_cost * 0.9 and (c_val - p_val) <= rel_tol:
                return True
        else:
            if c_val > p_val:
                return True
            rel_tol = max(abs(p_val) * 0.01, 1e-6)
            if c_cost is not None and p_cost is not None and c_cost < p_cost * 0.9 and (p_val - c_val) <= rel_tol:
                return True
    return False


def cmd_bearings(
    g: ContourGraph,
    expd_id: str | None = None,
    as_json: bool = False,
) -> None:
    """Run the Scientific Move detectors over FieldLogs, embedded diffs, and Cairns."""
    target_cairns = g.expedition_cairns(strip_wiki(expd_id)) if expd_id else list(g.cairns.values())
    usable = g.usable_fieldlogs()
    haze = load_haze(g.root)

    # Move 0: Hygiene / Drift
    drift = find_drift_issues(g)

    # Move 1: Confound Isolation (De-confounding runs where multiple facets changed vs parent)
    clean_parent_intervs = {
        ( r.parents[0], r.interventions )
        for r in usable
        if r.parents
        and (get_or_compute_vs_parent(g, r, haze=haze) or {}).get("single_logical_diff") is True
    }
    m1_confounds: list[dict[str, Any]] = []
    for log in usable:
        vs = get_or_compute_vs_parent(g, log, haze=haze)
        if vs is None:
            continue
        if not vs.get("single_logical_diff"):
            pid = strip_wiki(str(vs.get("parent", "")))
            if (pid, log.interventions) in clean_parent_intervs:
                continue
            fdiff = vs.get("confounding_facets") or {}
            added = vs.get("added_interventions") or []
            m1_confounds.append({
                "run": log.id,
                "parent": pid,
                "added_interventions": added,
                "confounding_facets": fdiff,
                "suggestion": (
                    f"Run {added or 'the intervention'} on {pid}'s exact baseline facets "
                    f"(holding {sorted(fdiff)} fixed) to isolate the causal effect."
                    if fdiff
                    else f"Declare explicit atomic `interventions` separating {log.id} from {pid}."
                ),
            })

    # Move 2: Dialectics (Contradictions, Boundary Regimes & Noise-Floor Checks)
    m2_dialectics: list[dict[str, Any]] = []
    for c in target_cairns:
        con, qual = c.refs("contradicts"), c.refs("qualifies")
        if con or qual:
            m2_dialectics.append({
                "cairn": c.id,
                "status": c.status,
                "title": c.title,
                "contradicts": con,
                "qualifies": qual,
            })

    # Move 3: Synthesis (Compositional Combination of Orthogonal Wins)
    all_interventions_seen = {r.interventions for r in usable}
    by_parent: dict[str, list[ContourNode]] = {}
    for log in usable:
        for p in log.parents:
            if p in g.fieldlogs and _is_pareto_win(g, log, g.fieldlogs[p]):
                by_parent.setdefault(p, []).append(log)
    m3_synthesis: list[dict[str, Any]] = []
    for parent_id, children in by_parent.items():
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                c1, c2 = children[i], children[j]
                shared_evals = sorted(set(c1.evals) & set(c2.evals))
                combined = c1.interventions | c2.interventions
                if shared_evals and c1.interventions != c2.interventions and combined not in all_interventions_seen:
                    m3_synthesis.append({
                        "parent": parent_id,
                        "left": {"id": c1.id, "interventions": sorted(c1.interventions)},
                        "right": {"id": c2.id, "interventions": sorted(c2.interventions)},
                        "combined_interventions": sorted(combined),
                        "shared_evals": shared_evals,
                    })

    # Move 4: Occam's Razor (Subtractive Ablation of Multi-Tag Interventions)
    all_parents = {p for log in g.fieldlogs.values() for p in log.parents}
    m4_occam: list[dict[str, Any]] = []
    seen_tag_sets: set[frozenset[str]] = set()
    for log in usable:
        if len(log.interventions) > 1 and log.id not in all_parents and log.interventions not in seen_tag_sets:
            seen_tag_sets.add(log.interventions)
            tags = sorted(log.interventions)
            leave_one_out = [
                {"strip": t, "keep": [x for x in tags if x != t]}
                for t in tags
                if frozenset(x for x in tags if x != t) not in all_interventions_seen
            ]
            if leave_one_out:
                m4_occam.append({
                    "id": log.id,
                    "interventions": tags,
                    "candidate_ablations": leave_one_out,
                })

    # Move 5: Deduction (Regime & Scaling Coverage Grid across Cairns)
    all_tasks = sorted({str(r.get("task")) for r in usable if r.get("task")})
    all_scales = sorted({str(r.get("arch_scale")) for r in usable if r.get("arch_scale")})
    all_evals = sorted({e for r in usable for e in r.evals})
    m5_deduction: list[dict[str, Any]] = []
    for c in target_cairns:
        if c.status in ("falsified", "toppled"):
            continue
        linked_ids = c.refs("supports") + c.refs("qualifies")
        linked_runs = [g.fieldlogs[lid] for lid in linked_ids if lid in g.fieldlogs]
        tasks = sorted({str(r.get("task")) for r in linked_runs if r.get("task")})
        scales = sorted({str(r.get("arch_scale")) for r in linked_runs if r.get("arch_scale")})
        evals = sorted({e for r in linked_runs for e in r.evals})
        missing_evals = [e for e in all_evals if e not in evals]
        missing_scales = [s for s in all_scales if s not in scales]
        untested = [
            ln.strip() for ln in c.body.splitlines()
            if "untested" in ln.lower() or "open:" in ln.lower()
        ]
        if len(tasks) <= 1 or len(scales) <= 1 or missing_evals or untested:
            m5_deduction.append({
                "cairn": c.id,
                "status": c.status,
                "tasks": tasks,
                "scales": scales,
                "evals": evals,
                "missing_evals_in_repo": missing_evals,
                "missing_scales_in_repo": missing_scales,
                "untested": untested[:2],
            })

    # Move 6: Abduction (Ridgeline Bottleneck Diagnosis)
    m6_abduction: list[dict[str, Any]] = []
    for eds in all_evals:
        _, _, ranked, _ = _compute_ridge(g, eval_ds=eds)
        if ranked:
            best = ranked[0]
            obs = best.sections.get("Observations & Anomalies", "").strip()
            first_obs = next((ln.strip() for ln in obs.splitlines() if ln.strip()), "See telemetry / curves")
            m6_abduction.append({"eval": eds, "best_run": best.id, "title": best.title, "observation": first_obs})

    if as_json:
        _emit_json({
            "move_0_drift": [i.as_dict() for i in drift],
            "move_1_confound_isolation": m1_confounds,
            "move_2_dialectics": m2_dialectics,
            "move_3_synthesis": m3_synthesis,
            "move_4_occam": m4_occam,
            "move_5_deduction": m5_deduction,
            "move_6_abduction": m6_abduction,
        })
        return

    print("=== CONTOUR BEARINGS: NEXT EXPERIMENT CANDIDATES ===\n")
    print(f"[Move 0: Terrain Drift & Hygiene] ({len(drift)} item(s))")
    for d in drift[:5]:
        print(f"  * [{d.code}] {d.message}")
    if not drift:
        print("  * All cairns and fieldlog links are clean.")

    print("\n[Move 1: Confound Isolation — De-confound Multi-Diff Runs]")
    for item in m1_confounds:
        print(
            f"  * {item['run']} vs {item['parent']}: confounded on {sorted(item['confounding_facets'])}\n"
            f"    -> Next experiment: {item['suggestion']}"
        )
    if not m1_confounds:
        print("  * All parent->child comparisons are clean single-diff runs.")

    print("\n[Move 2: Dialectics — Contradictions & Boundary Regimes]")
    for item in m2_dialectics:
        print(
            f"  * {item['cairn']} [{item['status']}]: {item['title']}\n"
            f"    -> Contradicted by: {item['contradicts'] or 'none'} | Qualified by: {item['qualifies'] or 'none'}"
        )
    if not m2_dialectics:
        print("  * No active contradictions or boundary qualifications detected.")

    print("\n[Move 3: Synthesis — Compositional Combination of Orthogonal Wins]")
    for item in m3_synthesis:
        print(
            f"  * Combine {item['left']['id']} ({item['left']['interventions']})\n"
            f"        + {item['right']['id']} ({item['right']['interventions']})\n"
            f"    -> Both improved over {item['parent']} on {item['shared_evals']}. "
            f"Test union: {item['combined_interventions']}"
        )
    if not m3_synthesis:
        print("  * No uncombined sibling Pareto wins found.")

    print("\n[Move 4: Occam's Razor — Leave-One-Out Ablation of Bundled Interventions]")
    for item in m4_occam:
        strips = [c["strip"] for c in item["candidate_ablations"]]
        print(
            f"  * {item['id']} bundled {item['interventions']}\n"
            f"    -> Untested leave-one-out strips: {strips}"
        )
    if not m4_occam:
        print("  * No unablated multi-intervention runs detected.")

    print("\n[Move 5: Deduction — Regime & Scaling Coverage Gaps]")
    for item in m5_deduction:
        print(
            f"  * {item['cairn']} [{item['status']}]: verified on tasks={item['tasks']}, "
            f"scales={item['scales']}, evals={item['evals']}"
        )
        if item["missing_evals_in_repo"]:
            print(f"    -> Untested repo evals: {item['missing_evals_in_repo']}")
        if item["untested"]:
            print(f"    -> Open regime note: {item['untested'][0]}")
    if not m5_deduction:
        print("  * All active cairns span multiple tasks and scales.")

    print("\n[Move 6: Abduction — Ridgeline Bottleneck Diagnosis]")
    for item in m6_abduction:
        print(
            f"  * Benchmark `{item['eval']}` -> Top run: {item['best_run']} ({item['title']})\n"
            f"    -> Bottleneck hook: {item['observation']}"
        )


def _sync_from_aim(root: Path, aim_repo_path: str | None, index: FieldlogIndex, force: bool) -> list[dict[str, Any]]:
    try:
        import aim  # type: ignore
    except ImportError as exc:
        if aim_repo_path is not None:
            raise SystemExit("Error: `aim` package is not installed. Use `--from-json` or install `aim`.") from exc
        return []

    aim_dir = Path(aim_repo_path) if aim_repo_path else (root / ".aim")
    if not aim_repo_path and not aim_dir.exists():
        return []

    from .bundle import validate_bundle

    repo = aim.Repo.from_path(aim_repo_path or str(root))
    results: list[dict[str, Any]] = []
    for run in repo.iter_runs():
        raw = run.get("contour_bundle")
        if not raw:
            continue
        metrics_dict = dict(raw.get("metrics") or {})
        for metric in run.metrics():
            subset = (metric.context or {}).get("subset")
            key = f"{subset}/{metric.name}" if subset else metric.name
            _, vals = metric.values.sparse_numpy()
            if len(vals) > 0:
                metrics_dict[key] = float(vals[-1])
        raw["metrics"] = metrics_dict
        raw.setdefault("schema", SCHEMA_ID)
        bundle = validate_bundle(raw, source=f"aim:{run.hash}")
        res = ingest_bundle(root, bundle, index=index, force=force)
        results.append({
            "source": f"aim:{run.hash}",
            "action": res.action,
            "log_id": res.log_id,
            "message": res.message,
            "flagged": res.flagged,
        })
    return results


def cmd_sync(
    root: Path,
    aim_repo_path: str | None,
    json_paths: list[Path] | None,
    force: bool = False,
    as_json: bool = False,
) -> None:
    cfg = load_config(root)
    index = FieldlogIndex(root)
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    paths = iter_bundle_paths(json_paths) if json_paths else iter_bundle_paths(cfg.inbox)
    for jpath in paths:
        try:
            bundle = load_bundle_file(jpath)
            res = ingest_bundle(root, bundle, index=index, force=force)
            results.append({
                "source": str(jpath),
                "action": res.action,
                "log_id": res.log_id,
                "message": res.message,
                "flagged": res.flagged,
            })
        except BundleError as exc:
            errors.append(str(exc))

    if not json_paths:
        results.extend(_sync_from_aim(root, aim_repo_path, index, force))

    relinked = relink_parents(root, index)

    if as_json:
        _emit_json({"results": results, "relinked_parents": relinked, "errors": errors})
        if errors or any(r["action"] == "rejected" for r in results):
            raise SystemExit(1)
        return

    for r in results:
        if r["action"] == "unchanged":
            continue
        if r["action"] == "rejected":
            print(f"[contour sync] REJECTED {r['log_id']}: {r['message']}")
        else:
            print(f"[contour sync] {r['action'].upper()} {r['log_id']} from {Path(r['source']).name}")
            for fc in r["flagged"]:
                print(f"  -> Flagged cairn needs_synthesis: {fc}")
    for lid in relinked:
        print(f"[contour sync] Resolved parent link(s) on {lid}")
    for err in errors:
        print(f"[contour sync] ERROR: {err}", file=sys.stderr)
    if errors or any(r["action"] == "rejected" for r in results):
        raise SystemExit(1)


def cmd_new(
    g: ContourGraph,
    node_type: str,
    slug: str,
    title: str,
    as_json: bool = False,
) -> None:
    today = dt.date.today().isoformat()
    if node_type == "expedition":
        node_id = slug if slug.startswith("expd_") else f"expd_{slug}"
        path = g.root / "expeditions" / f"{node_id}.md"
        fm: dict[str, Any] = {
            "id": node_id,
            "type": "expedition",
            "title": title,
            "status": "active",
            "cairns": [],
        }
        body = (
            "## North Star & Motivation\n"
            "- State the overarching research problem and success criteria.\n\n"
            "## Current State & Open Questions\n"
            "- \n"
        )
    elif node_type == "cairn":
        node_id = slug if slug.startswith("crn_") else f"crn_{slug}"
        path = g.root / "cairns" / f"{node_id}.md"
        fm = {
            "id": node_id,
            "type": "cairn",
            "title": title,
            "status": "active",
            "needs_synthesis": False,
            "expeditions": [],
            "parent_cairns": [],
            "supports": [],
            "contradicts": [],
            "qualifies": [],
        }
        body = (
            "## Proposition & Intuition\n"
            "- State the falsifiable hypothesis or principle and the conceptual mechanism behind it.\n\n"
            "## Where This Holds vs. Breaks (Scope & Regimes)\n"
            "- **Verified:** \n"
            "- **Breaks / Fails:** \n"
            "- **Untested / Open:** \n"
        )
    elif node_type == "fieldlog":
        date_str = today.replace("-", "")
        node_id = slug if slug.startswith("log_") else f"log_{date_str}_{slug}"
        path = g.root / "fieldlogs" / f"{node_id}.md"
        fm = {
            "id": node_id,
            "type": "fieldlog",
            "title": title,
            "run_id": f"manual_{node_id}",
            "status": "final",
            "validity": DEFAULT_VALIDITY,
            "tier": "proxy",
            "created_at": today,
            "finished_at": today,
            "task": "unspecified",
            "arch_family": "unspecified",
            "arch_scale": "unspecified",
            "seed": None,
            "dataset_train": [],
            "dataset_eval": [],
            "interventions": [],
            "intervention_note": None,
            "parent_runs": [],
            "motivated_by": [],
            "compute": {"gpu_hours": None},
            "metrics": {},
        }
        body = "## Key Takeaway (TL;DR)\n- \n\n## Observations & Anomalies\n- \n"
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


def cmd_compare(
    g: ContourGraph,
    controls: list[str],
    treatments: list[str],
    iso_tol: float = 0.02,
    as_json: bool = False,
) -> None:
    try:
        cmp_out = compare_runs(
            g,
            control_ids=[strip_wiki(x) for x in controls],
            treatment_ids=[strip_wiki(x) for x in treatments],
            iso_tol=iso_tol,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if as_json:
        _emit_json(cmp_out)
        return

    conf = cmp_out["confounds"]
    fdiff = cmp_out["facet_diff"]
    print(
        f"=== COMPARE {cmp_out['controls']} -> {cmp_out['treatments']} "
        f"[Grade {cmp_out['grade']} | Tier: {cmp_out['tier']}] ==="
    )
    print(f"Regime: {cmp_out['regime'] or 'N/A'}")
    print(f"Added interventions: {fdiff['added_interventions']}")
    if fdiff["removed_interventions"]:
        print(f"Removed interventions: {fdiff['removed_interventions']}")
    if fdiff["confounding_facets"]:
        print(f"CONFOUNDED FACETS: {fdiff['confounding_facets']}")
    print(
        f"Confound flags: single_logical_diff={conf['single_logical_diff']} | "
        f"iso_compute={conf['iso_compute']} | all_usable={conf['all_usable']}"
    )
    for k, d in cmp_out["metrics"].items():
        snr = f" | SNR={d['snr']}x" if d.get("snr") is not None else ""
        noise = f" (noise_std={d['noise_std']})" if d.get("noise_std") is not None else ""
        clr = " [CLEARS NOISE]" if d.get("clears_noise") else ""
        print(f"  - {k}: {d['control_mean']} -> {d['treatment_mean']} ({d['delta']:+.4g}){snr}{noise}{clr}")


def cmd_noise(
    g: ContourGraph,
    action: str,
    regime: str | None = None,
    from_ids: list[str] | None = None,
    metrics_filter: list[str] | None = None,
    as_json: bool = False,
) -> None:
    if action == "list":
        data = load_haze(g.root)
        if as_json:
            _emit_json(data)
            return
        if not data:
            print("No noise-floor entries yet. Run `contour noise seed <regime> --from <log_ids...>`.")
            return
        print("=== NOISE FLOOR REGISTRY (noise.yaml) ===")
        for reg, m_map in sorted(data.items()):
            print(f"[{reg}]")
            for k, entry in sorted(m_map.items()):
                print(f"  - {k}: std={entry['std']} (n={entry['n_seeds']}, mean={entry.get('mean')})")
        return

    if action == "seed":
        if not regime or not from_ids:
            raise SystemExit("Error: `contour noise seed` requires <regime> and --from <log_ids...>.")
        runs = []
        for raw_id in from_ids:
            lid = strip_wiki(raw_id)
            node = g.fieldlogs.get(lid)
            if not node:
                raise SystemExit(f"Unknown fieldlog: {lid!r}")
            if not node.is_usable_evidence:
                raise SystemExit(
                    f"Fieldlog {lid!r} is not usable evidence (status={node.status}, validity={node.validity})."
                )
            runs.append(node)
        try:
            entries = seed_haze_from_runs(g.root, regime, runs, metrics_filter=metrics_filter)
        except ValueError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        # Refresh vs_parent SNR badges across fieldlogs
        relink_parents(g.root, FieldlogIndex(g.root))
        if as_json:
            _emit_json({"regime": regime, "entries": entries})
            return
        print(f"Updated noise.yaml for {regime} ({len(runs)} run(s)):")
        for k, entry in entries.items():
            print(f"  - {k}: std={entry['std']} (n={entry['n_seeds']})")


def cmd_link(
    g: ContourGraph,
    log_id: str,
    cairn_id: str,
    relation: str,
    as_json: bool = False,
) -> None:
    lid = strip_wiki(log_id)
    cid = strip_wiki(cairn_id)
    if lid not in g.fieldlogs:
        raise SystemExit(f"Error: Unknown fieldlog {lid!r}")
    cairn = g.cairns.get(cid)
    if not cairn:
        raise SystemExit(f"Error: Unknown cairn {cid!r}")
    if relation not in EDGE_KEYS:
        raise SystemExit(f"Error: Invalid relation {relation!r} (expected one of {EDGE_KEYS})")

    fm, body = parse_markdown_file(cairn.path)
    for edge in EDGE_KEYS:
        existing = [strip_wiki(x) for x in (fm.get(edge) or [])]
        if edge == relation:
            if lid not in existing:
                existing.append(lid)
        else:
            existing = [x for x in existing if x != lid]
        fm[edge] = [as_wikilink(x) for x in existing]

    fm["needs_synthesis"] = False
    write_markdown_file(cairn.path, fm, body)

    vs = get_or_compute_vs_parent(g, g.fieldlogs[lid])
    if as_json:
        _emit_json({
            "cairn": cid,
            "fieldlog": lid,
            "relation": relation,
            "vs_parent": vs,
        })
        return
    grade_str = f" [vs parent: Grade {vs['grade']}]" if vs else ""
    print(f"Linked {lid} -> {cid} under `{relation}`{grade_str}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to contour root (with contour.yaml)",
    )
    common.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit machine-readable JSON",
    )

    parser = argparse.ArgumentParser(
        prog="contour",
        description="Contour: Navigate, survey, and reason over your ML Epistemic Graph.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Path to contour root (with contour.yaml)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit machine-readable JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", parents=[common], help="Initialize a Contour root (defaults to ./contour/ subdirectory)")
    p_init.add_argument("path", nargs="?", type=Path, default=None, help="Target directory (default: ./contour, or . if passed)")
    p_init.add_argument("--with-skill", action="store_true", help="Write .agent/skills/contour-research/SKILL.md into the repo")
    p_init.add_argument(
        "--eject-logger",
        nargs="?",
        const=Path("contour_run.py"),
        default=None,
        type=Path,
        help="Write a standalone stdlib-only contour_run.py logger for external cluster/Docker jobs",
    )

    sub.add_parser("map", parents=[common], aliases=["overview"], help="Print the Expedition -> Cairn -> FieldLog DAG")

    p_insp = sub.add_parser("inspect", parents=[common], help="Zoom in on an Expedition, Cairn, or FieldLog node")
    p_insp.add_argument("node_id", type=str, help="ID of expedition, cairn, or fieldlog")

    p_drift = sub.add_parser("drift", parents=[common], aliases=["stale"], help="Move 0: Check for drift, broken links, and confound warnings")
    p_drift.add_argument("--strict", action="store_true", help="Exit non-zero if any error-severity issue exists")

    p_ridge = sub.add_parser("ridge", parents=[common], aliases=["leaderboard"], help="Compare usable fieldlogs & plot the Pareto ridgeline")
    p_ridge.add_argument("--eval", dest="eval_ds", type=str, default=None, help="Filter by eval dataset")
    p_ridge.add_argument("--task", type=str, default=None, help="Filter by task")
    p_ridge.add_argument("--metric", type=str, default=None, help="Primary metric key to sort by")

    p_bear = sub.add_parser("bearings", parents=[common], aliases=["candidates"], help="Run the Scientific Moves to generate next experiments")
    p_bear.add_argument("--expedition", dest="expd_id", type=str, default=None, help="Filter by expedition ID")

    p_sync = sub.add_parser("sync", parents=[common], help="Ingest Field Bundles (and Aim runs) into fieldlogs/")
    p_sync.add_argument("--repo", type=str, default=None, help="Aim repo path or aim:// server URI")
    p_sync.add_argument("--from-json", type=Path, nargs="+", default=None, help="Field Bundle JSON files or directories")
    p_sync.add_argument("--force", action="store_true", help="Allow overwriting terminal run bundles")

    p_new = sub.add_parser("new", parents=[common], help="Scaffold a new expedition, cairn, or fieldlog node")
    p_new.add_argument("node_type", choices=["expedition", "cairn", "fieldlog"])
    p_new.add_argument("slug", type=str, help="Short identifier slug")
    p_new.add_argument("--title", type=str, required=True, help="Human-readable title")

    p_link = sub.add_parser("link", parents=[common], help="Place a FieldLog under a Cairn's supports/contradicts/qualifies")
    p_link.add_argument("log_id", type=str, help="FieldLog ID (log_*)")
    p_link.add_argument("cairn_id", type=str, help="Cairn ID (crn_*)")
    p_link.add_argument("--relation", choices=list(EDGE_KEYS), required=True)

    p_cmp = sub.add_parser("compare", parents=[common], help="Compare control vs. treatment fieldlogs on the fly")
    p_cmp.add_argument("controls", nargs="+", help="Control fieldlog ID(s)")
    p_cmp.add_argument("--vs", dest="treatments", nargs="+", required=True, help="Treatment fieldlog ID(s)")
    p_cmp.add_argument("--iso-tol", type=float, default=0.02, help="Relative tolerance for iso-FLOPs/params (default: 0.02)")

    p_noise = sub.add_parser("noise", parents=[common], aliases=["haze"], help="Manage the optional noise.yaml noise-floor registry")
    noise_sub = p_noise.add_subparsers(dest="noise_cmd", required=True)
    noise_sub.add_parser("list", parents=[common], help="List registered noise floors")
    p_nseed = noise_sub.add_parser("seed", parents=[common], help="Estimate noise floor from repeated baseline runs")
    p_nseed.add_argument("regime", type=str, help="Regime key, e.g. image_classification/imagenet1k_val/22M")
    p_nseed.add_argument("--from", dest="from_ids", nargs="+", required=True, help="Baseline fieldlog IDs (>= 2)")
    p_nseed.add_argument("--metric", dest="metrics_filter", action="append", default=None, help="Limit to specific metric key(s)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "init":
        cmd_init(
            args.path,
            with_skill=args.with_skill,
            eject_logger=args.eject_logger,
            as_json=args.as_json,
        )
        return

    try:
        root = find_root(explicit=args.root)
    except ConfigError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if args.cmd == "sync":
        cmd_sync(root, args.repo, args.from_json, force=args.force, as_json=args.as_json)
        return

    g = ContourGraph(root)

    if args.cmd in ("map", "overview"):
        cmd_map(g, as_json=args.as_json)
    elif args.cmd == "inspect":
        cmd_inspect(g, args.node_id, as_json=args.as_json)
    elif args.cmd in ("drift", "stale"):
        cmd_drift(g, as_json=args.as_json, strict=args.strict)
    elif args.cmd in ("ridge", "leaderboard"):
        cmd_ridge(g, args.eval_ds, args.task, args.metric, as_json=args.as_json)
    elif args.cmd in ("bearings", "candidates"):
        cmd_bearings(g, args.expd_id, as_json=args.as_json)
    elif args.cmd == "new":
        cmd_new(g, args.node_type, args.slug, args.title, as_json=args.as_json)
    elif args.cmd == "link":
        cmd_link(g, args.log_id, args.cairn_id, relation=args.relation, as_json=args.as_json)
    elif args.cmd == "compare":
        cmd_compare(g, args.controls, args.treatments, iso_tol=args.iso_tol, as_json=args.as_json)
    elif args.cmd in ("noise", "haze"):
        cmd_noise(
            g,
            args.noise_cmd,
            regime=getattr(args, "regime", None),
            from_ids=getattr(args, "from_ids", None),
            metrics_filter=getattr(args, "metrics_filter", None),
            as_json=args.as_json,
        )


if __name__ == "__main__":
    main()
