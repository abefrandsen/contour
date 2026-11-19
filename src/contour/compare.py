"""Transparent structured run comparisons, analysis cleanliness, and hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .config import ContourConfig
from .graph import ContourGraph, ContourNode
from .mdio import as_wikilink

DECISIVE_VERDICTS = ("win", "efficiency_win", "tradeoff", "regression")

_VERDICT_TABLE: dict[tuple[str, str], str] = {
    ("clear", "better"): "win",
    ("clear", "matched"): "win",
    ("clear", "worse"): "tradeoff",
    ("marginal", "better"): "marginal",
    ("marginal", "matched"): "marginal",
    ("marginal", "worse"): "tradeoff",
    ("neutral", "better"): "efficiency_win",
    ("neutral", "matched"): "neutral",
    ("neutral", "worse"): "regression",
    ("regression", "better"): "tradeoff",
    ("regression", "matched"): "regression",
    ("regression", "worse"): "regression",
}


def canonical_hash(payload: Any) -> str:
    """Return a deterministic 8-hex-char truncated SHA-256 hash."""
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def flatten_config(cfg: dict[str, Any] | None, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config mappings into dot-separated leaf keys."""
    if not cfg or not isinstance(cfg, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten_config(v, prefix=key))
        else:
            out[key] = v
    return out


def _is_ignored_key(dot_key: str, ignored: set[str]) -> bool:
    if dot_key in ignored:
        return True
    parts = dot_key.split(".")
    if parts[0] in ignored or parts[-1] in ignored:
        return True
    for i in range(1, len(parts)):
        if ".".join(parts[:i]) in ignored:
            return True
    return False


def _run_hash_snapshot(
    run_id: str,
    fm: dict[str, Any],
    resolved_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    code = fm.get("code") or {}
    cfg = resolved_config if resolved_config is not None else (fm.get("config") or {})
    return {
        "id": run_id,
        "status": fm.get("status"),
        "bugged": bool(fm.get("bugged", False)),
        "config": flatten_config(cfg),
        "metrics": fm.get("metrics") or {},
        "compute": fm.get("compute") or {},
        "code_commit": code.get("commit"),
        "code_dirty": bool(code.get("dirty", False)),
    }


def _analysis_hash_snapshot(
    ana: ContourNode,
    g: ContourGraph | None = None,
) -> dict[str, Any]:
    code = ana.code
    subj_snaps: dict[str, Any] = {}
    if g is not None:
        for sid in ana.subjects:
            srun = g.runs.get(sid)
            subj_snaps[sid] = (
                _run_hash_snapshot(srun.id, srun.frontmatter, srun.config) if srun is not None else "missing"
            )
    return {
        "id": ana.id,
        "kind": ana.kind,
        "status": ana.status,
        "bugged": ana.bugged,
        "subjects": ana.subjects,
        "subject_snapshots": subj_snaps,
        "metrics": ana.metrics,
        "artifacts": ana.artifacts,
        "script": ana.script,
        "script_sha256": ana.script_sha256,
        "script_tracked": ana.script_tracked,
        "launch_cmd": ana.get("launch_cmd"),
        "code_commit": code.get("commit"),
        "code_dirty": bool(code.get("dirty", False)),
    }


def compute_inputs_hash(
    run_id: str,
    run_fm: dict[str, Any],
    base_id: str,
    base_fm: dict[str, Any],
    config: ContourConfig,
    *,
    run_config: dict[str, Any] | None = None,
    base_config: dict[str, Any] | None = None,
    extra_eval_snapshots: dict[str, Any] | None = None,
) -> str:
    """Truncated SHA-256 over every input read by run comparison."""
    r_metrics = set((run_fm.get("metrics") or {}).keys())
    b_metrics = set((base_fm.get("metrics") or {}).keys())
    shared = sorted(r_metrics & b_metrics)
    if extra_eval_snapshots:
        shared = sorted(set(shared) | set(extra_eval_snapshots.keys()))
    metric_specs = {m: config.metric_spec(m) for m in shared}
    payload = {
        "baseline_id": base_id,
        "baseline": _run_hash_snapshot(base_id, base_fm, base_config),
        "treatment": _run_hash_snapshot(run_id, run_fm, run_config),
        "shared_metric_specs": metric_specs,
        "ignored_config_keys": sorted(config.ignored_config_keys),
        "extra_evals": extra_eval_snapshots or {},
    }
    return canonical_hash(payload)


def compute_config_diff(
    base_cfg: dict[str, Any] | None,
    treat_cfg: dict[str, Any] | None,
    ignored_config_keys: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Compute nested dot-path `config_diff` and subtree summaries (`config_blocks`)."""
    b_flat = flatten_config(base_cfg)
    t_flat = flatten_config(treat_cfg)
    all_keys = sorted(
        k for k in (set(b_flat) | set(t_flat))
        if not _is_ignored_key(k, ignored_config_keys)
    )
    config_diff: dict[str, dict[str, Any]] = {}
    for k in all_keys:
        bv = b_flat.get(k)
        tv = t_flat.get(k)
        if bv != tv:
            config_diff[k] = {"baseline": bv, "value": tv}

    by_block: dict[str, list[str]] = {}
    for k in config_diff:
        if "." in k:
            blk = k.split(".", 1)[0]
            by_block.setdefault(blk, []).append(k)

    config_blocks: dict[str, dict[str, Any]] = {}
    for blk, keys in sorted(by_block.items()):
        if len(keys) >= 2:
            summary: dict[str, Any] = {"changed_keys": len(keys), "keys": keys}
            for tag_key in (f"{blk}.type", f"{blk}.name", f"{blk}.arch", f"{blk}.model"):
                if tag_key in config_diff:
                    summary["baseline"] = config_diff[tag_key]["baseline"]
                    summary["value"] = config_diff[tag_key]["value"]
                    break
            config_blocks[blk] = summary

    return config_diff, config_blocks


def compute_code_diff(base_fm: dict[str, Any], treat_fm: dict[str, Any]) -> dict[str, Any]:
    b_code = base_fm.get("code") or {}
    t_code = treat_fm.get("code") or {}
    b_commit = b_code.get("commit")
    t_commit = t_code.get("commit")
    same_commit = b_commit is not None and t_commit is not None and b_commit == t_commit
    return {
        "same_commit": same_commit,
        "baseline_commit": b_commit,
        "value_commit": t_commit,
        "baseline_dirty": bool(b_code.get("dirty", False)),
        "value_dirty": bool(t_code.get("dirty", False)),
    }


def compute_compute_diff(
    base_fm: dict[str, Any],
    treat_fm: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Surface raw `params` and `gpu_hours` deltas and ratios automatically for free."""
    b_comp = base_fm.get("compute") or {}
    t_comp = treat_fm.get("compute") or {}
    diff: dict[str, dict[str, Any]] = {}
    for key in ("params", "gpu_hours"):
        bv = b_comp.get(key)
        tv = t_comp.get(key)
        if (
            isinstance(bv, (int, float))
            and not isinstance(bv, bool)
            and isinstance(tv, (int, float))
            and not isinstance(tv, bool)
        ):
            bf, tf = float(bv), float(tv)
            if bf < 0 or tf < 0:
                continue
            entry: dict[str, Any] = {
                "baseline": bv,
                "value": tv,
                "delta": round(tf - bf, 6),
            }
            if bf > 0:
                entry["ratio"] = round(tf / bf, 4)
            diff[key] = entry
    return diff


def compute_standalone_control_level(run_fm: dict[str, Any]) -> str:
    """Assign `invalid | exploratory | clean` for a standalone run."""
    if str(run_fm.get("status", "unknown")) != "final" or bool(run_fm.get("bugged", False)):
        return "invalid"
    if bool((run_fm.get("code") or {}).get("dirty", False)):
        return "exploratory"
    return "clean"


def compute_control_level(base_fm: dict[str, Any], treat_fm: dict[str, Any]) -> str:
    """Assign `invalid | exploratory | clean` based strictly on hard mechanical facts."""
    if (
        str(base_fm.get("status", "unknown")) != "final"
        or str(treat_fm.get("status", "unknown")) != "final"
        or bool(base_fm.get("bugged", False))
        or bool(treat_fm.get("bugged", False))
    ):
        return "invalid"
    b_dirty = bool((base_fm.get("code") or {}).get("dirty", False))
    t_dirty = bool((treat_fm.get("code") or {}).get("dirty", False))
    if b_dirty or t_dirty:
        return "exploratory"
    return "clean"


def compute_analysis_control_level(g: ContourGraph, ana: ContourNode) -> str:
    """Assign `invalid | exploratory | clean` for an analysis node, propagating subject taint."""
    if not ana.is_usable_evidence:
        return "invalid"
    for sid in ana.subjects:
        srun = g.runs.get(sid)
        if srun is None or not srun.is_usable_evidence:
            return "invalid"
    code = ana.code
    if bool(code.get("dirty", False)) or not code.get("commit") or not ana.script_tracked:
        return "exploratory"
    return "clean"


def compute_advisory_flags(
    *,
    config_diff: dict[str, dict[str, Any]],
    code_diff: dict[str, Any],
    eval_flags: list[str] | None = None,
) -> list[str]:
    """Emit transparent advisory flags for Tier-3 Critic and human review."""
    flags: list[str] = []
    if code_diff["baseline_commit"] and code_diff["value_commit"] and not code_diff["same_commit"]:
        flags.append("code_changed")
    if code_diff["baseline_dirty"] or code_diff["value_dirty"]:
        flags.append("code_dirty")
    top_blocks = {k.split(".", 1)[0] for k in config_diff}
    if len(top_blocks) >= 2 and len(config_diff) >= 4:
        flags.append("multi_block_config_diff")
    if not config_diff and code_diff["same_commit"]:
        flags.append("replicate")
    for ef in (eval_flags or []):
        if ef not in flags:
            flags.append(ef)
    return flags


def _metric_signal(
    baseline: float,
    value: float,
    direction: str,
    min_delta: float | None,
    min_rel_delta: float | None,
) -> str | None:
    if min_delta is None and min_rel_delta is None:
        return None
    delta = value - baseline
    beneficial = delta if direction == "max" else -delta

    thresholds: list[float] = []
    if min_delta is not None:
        thresholds.append(min_delta)
    if min_rel_delta is not None:
        if baseline != 0:
            thresholds.append(abs(baseline) * min_rel_delta)
        elif min_rel_delta == 0:
            thresholds.append(0.0)
        elif not thresholds:
            # Relative delta against a 0 baseline with non-zero threshold
            if beneficial > 0:
                return "marginal"
            if beneficial < 0:
                return "regression"
            return "neutral"

    tau = max(thresholds)
    if beneficial >= tau:
        return "clear"
    if beneficial > 0:
        return "marginal"
    if beneficial > -tau:
        return "neutral"
    return "regression"


def compute_signals_and_verdict(
    base_metrics_raw: dict[str, Any],
    treat_metrics_raw: dict[str, Any],
    config: ContourConfig,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Compute per-metric deltas/signals (`min_delta` / `min_rel_delta`) and overall verdict."""
    b_metrics = {
        str(k): float(v)
        for k, v in (base_metrics_raw or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    t_metrics = {
        str(k): float(v)
        for k, v in (treat_metrics_raw or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    shared = sorted(set(b_metrics) & set(t_metrics))

    deltas: dict[str, dict[str, Any]] = {}
    primary_signals: list[str] = []
    non_primary_quality_signals: list[str] = []
    cost_signals: list[str] = []

    for k in shared:
        bv = b_metrics[k]
        tv = t_metrics[k]
        delta = round(tv - bv, 6)
        kind = config.metric_kind(k)
        direction = config.metric_direction(k)
        md = config.min_delta(k)
        mrd = config.min_rel_delta(k)

        entry: dict[str, Any] = {
            "baseline": bv,
            "value": tv,
            "delta": delta,
        }
        if bv != 0:
            entry["rel_delta"] = round((tv - bv) / abs(bv), 6)
        if kind in ("quality", "cost"):
            sig = _metric_signal(bv, tv, direction, md, mrd)
            if sig is not None:
                entry["signal"] = sig
            if kind == "quality":
                if config.is_primary_metric(k):
                    if sig is not None:
                        primary_signals.append(sig)
                elif sig is not None:
                    non_primary_quality_signals.append(sig)
            elif kind == "cost" and sig is not None:
                cost_signals.append(sig)

        deltas[k] = entry

    if not primary_signals:
        return deltas, "incomparable"

    if "regression" in primary_signals:
        q_out = "regression"
    elif "clear" in primary_signals:
        q_out = "clear"
    elif "marginal" in primary_signals:
        q_out = "marginal"
    else:
        q_out = "neutral"

    if "regression" in cost_signals:
        c_out = "worse"
    elif "clear" in cost_signals:
        c_out = "better"
    else:
        c_out = "matched"

    verdict = _VERDICT_TABLE[(q_out, c_out)]

    if "regression" in non_primary_quality_signals:
        if verdict in ("win", "marginal", "efficiency_win"):
            verdict = "tradeoff"
        elif verdict == "neutral":
            verdict = "regression"

    return deltas, verdict


def recommend_relation(control_level: str, verdict: str) -> str:
    """Advisory link relation for a comparative run."""
    if control_level == "clean" and verdict in ("win", "efficiency_win"):
        return "supports"
    if control_level == "clean" and verdict == "regression":
        return "contradicts"
    return "qualifies"


def _compose_attached_evals(
    g: ContourGraph,
    ctrl: ContourNode,
    treat: ContourNode,
) -> tuple[dict[str, float], dict[str, float], list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Compose inline run metrics with shared attached `kind: eval` metrics and detect eval drift."""
    ctrl_metrics = dict(ctrl.metrics)
    treat_metrics = dict(treat.metrics)
    ctrl_evals = g.attached_eval_metrics(ctrl.id)
    treat_evals = g.attached_eval_metrics(treat.id)

    eval_flags: list[str] = []
    eval_provenance: dict[str, dict[str, Any]] = {}
    eval_snapshots: dict[str, Any] = {}

    for mkey in sorted(set(ctrl_evals) & set(treat_evals)):
        if mkey in ctrl_metrics and mkey in treat_metrics:
            continue
        c_val, c_ev = ctrl_evals[mkey]
        t_val, t_ev = treat_evals[mkey]
        ctrl_metrics.setdefault(mkey, c_val)
        treat_metrics.setdefault(mkey, t_val)

        c_commit = c_ev.code.get("commit")
        t_commit = t_ev.code.get("commit")
        c_dirty = bool(c_ev.code.get("dirty", False))
        t_dirty = bool(t_ev.code.get("dirty", False))

        drifts: list[str] = []
        if (c_ev.script_sha256 and t_ev.script_sha256 and c_ev.script_sha256 != t_ev.script_sha256) or (
            c_ev.script and t_ev.script and c_ev.script != t_ev.script
        ):
            drifts.append("eval_script_changed")
        if c_commit and t_commit and c_commit != t_commit:
            drifts.append("eval_code_changed")
        if c_dirty or t_dirty:
            drifts.append("eval_code_dirty")
        for d in drifts:
            flag = f"eval_drift:{mkey}"
            if flag not in eval_flags:
                eval_flags.append(flag)
            if d not in eval_flags:
                eval_flags.append(d)

        eval_provenance[mkey] = {
            "baseline_eval": c_ev.id,
            "value_eval": t_ev.id,
            "baseline_launch_cmd": c_ev.get("launch_cmd"),
            "value_launch_cmd": t_ev.get("launch_cmd"),
            "drifts": drifts,
        }
        eval_snapshots[mkey] = {
            "baseline": _analysis_hash_snapshot(c_ev),
            "value": _analysis_hash_snapshot(t_ev),
        }

    return ctrl_metrics, treat_metrics, eval_flags, eval_provenance, eval_snapshots


def compute_vs_parent(
    run_fm: dict[str, Any],
    base_fm: dict[str, Any],
    config: ContourConfig,
    *,
    run_id: str | None = None,
    base_id: str | None = None,
    run_config: dict[str, Any] | None = None,
    base_config: dict[str, Any] | None = None,
    run_metrics: dict[str, Any] | None = None,
    base_metrics: dict[str, Any] | None = None,
    eval_flags: list[str] | None = None,
    eval_provenance: dict[str, dict[str, Any]] | None = None,
    eval_snapshots: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compute fresh structured comparison dict for a run against its parent."""
    if run_fm.get("status") != "final" or base_fm.get("status") != "final":
        return None
    rid = run_id or str(run_fm.get("id") or "run")
    bid = base_id or str(base_fm.get("id") or "base")
    r_cfg = run_config if run_config is not None else (run_fm.get("config") or {})
    b_cfg = base_config if base_config is not None else (base_fm.get("config") or {})
    r_met = run_metrics if run_metrics is not None else (run_fm.get("metrics") or {})
    b_met = base_metrics if base_metrics is not None else (base_fm.get("metrics") or {})

    compute_diff = compute_compute_diff(base_fm, run_fm)
    config_diff, config_blocks = compute_config_diff(b_cfg, r_cfg, config.ignored_config_keys)
    code_diff = compute_code_diff(base_fm, run_fm)
    control_level = compute_control_level(base_fm, run_fm)
    flags = compute_advisory_flags(
        config_diff=config_diff,
        code_diff=code_diff,
        eval_flags=eval_flags,
    )
    deltas, verdict = compute_signals_and_verdict(b_met, r_met, config)
    inputs_hash = compute_inputs_hash(
        rid,
        run_fm,
        bid,
        base_fm,
        config,
        run_config=r_cfg,
        base_config=b_cfg,
        extra_eval_snapshots=eval_snapshots,
    )
    out: dict[str, Any] = {
        "baseline": as_wikilink(bid),
        "inputs_hash": inputs_hash,
        "control_level": control_level,
        "verdict": verdict,
        "flags": flags,
        "config_diff": config_diff,
        "config_blocks": config_blocks,
        "code_diff": code_diff,
        "compute_diff": compute_diff,
        "deltas": deltas,
    }
    if eval_provenance:
        out["eval_provenance"] = eval_provenance
    return out


def get_fresh_vs_parent(g: ContourGraph, run: ContourNode) -> dict[str, Any] | None:
    """Always recompute structured comparison dynamically in memory from current node states."""
    base_id = run.parent
    if not base_id or base_id not in g.runs:
        return None
    base = g.runs[base_id]
    b_met, r_met, ev_flags, ev_prov, ev_snaps = _compose_attached_evals(g, base, run)
    return compute_vs_parent(
        run.frontmatter,
        base.frontmatter,
        g.config,
        run_id=run.id,
        base_id=base.id,
        run_config=run.config,
        base_config=base.config,
        run_metrics=r_met,
        base_metrics=b_met,
        eval_flags=ev_flags,
        eval_provenance=ev_prov,
        eval_snapshots=ev_snaps,
    )


def compare_runs(g: ContourGraph, ctrl_id: str, treat_id: str) -> dict[str, Any]:
    """Ad-hoc comparison between any two runs (composing inline + attached `kind: eval` metrics)."""
    ctrl = g.runs.get(ctrl_id)
    if not ctrl:
        raise ValueError(f"Unknown control run: {ctrl_id!r}")
    treat = g.runs.get(treat_id)
    if not treat:
        raise ValueError(f"Unknown treatment run: {treat_id!r}")

    c_met, t_met, ev_flags, ev_prov, _ = _compose_attached_evals(g, ctrl, treat)
    compute_diff = compute_compute_diff(ctrl.frontmatter, treat.frontmatter)
    config_diff, config_blocks = compute_config_diff(
        ctrl.config, treat.config, g.config.ignored_config_keys
    )
    code_diff = compute_code_diff(ctrl.frontmatter, treat.frontmatter)
    control_level = compute_control_level(ctrl.frontmatter, treat.frontmatter)
    flags = compute_advisory_flags(
        config_diff=config_diff,
        code_diff=code_diff,
        eval_flags=ev_flags,
    )
    deltas, verdict = compute_signals_and_verdict(c_met, t_met, g.config)
    out: dict[str, Any] = {
        "control": ctrl.id,
        "treatment": treat.id,
        "control_level": control_level,
        "verdict": verdict,
        "recommended_link": recommend_relation(control_level, verdict),
        "flags": flags,
        "config_diff": config_diff,
        "config_blocks": config_blocks,
        "code_diff": code_diff,
        "compute_diff": compute_diff,
        "deltas": deltas,
    }
    if ev_prov:
        out["eval_provenance"] = ev_prov
    return out
