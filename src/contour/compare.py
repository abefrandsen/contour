"""Controlled comparisons between runs: `vs_parent` and `contour compare`."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .config import ContourConfig
from .graph import FACET_KEYS, ContourGraph, ContourNode
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


def _norm_facet_value(key: str, val: Any) -> Any:
    if key == "dataset_train":
        if val is None:
            return []
        if isinstance(val, str):
            return [val]
        return sorted(str(x) for x in val)
    return val


def _run_hash_snapshot(fm: dict[str, Any]) -> dict[str, Any]:
    code = fm.get("code") or {}
    return {
        "id": fm.get("id"),
        "status": fm.get("status"),
        "validity": fm.get("validity", "ok"),
        "facets": {k: _norm_facet_value(k, fm.get(k)) for k in FACET_KEYS},
        "config": fm.get("config") or {},
        "interventions": [str(x) for x in (fm.get("interventions") or [])],
        "metrics": fm.get("metrics") or {},
        "compute": fm.get("compute") or {},
        "code_commit": code.get("commit"),
        "code_dirty": bool(code.get("dirty", False)),
    }


def compute_inputs_hash(
    run_fm: dict[str, Any],
    base_fm: dict[str, Any],
    config: ContourConfig,
) -> str:
    """Truncated SHA-256 over every input read by `vs_parent` (§4.1)."""
    r_metrics = set((run_fm.get("metrics") or {}).keys())
    b_metrics = set((base_fm.get("metrics") or {}).keys())
    shared = sorted(r_metrics & b_metrics)
    metric_specs = {m: config.metric_spec(m) for m in shared}
    payload = {
        "baseline_id": base_fm.get("id"),
        "baseline": _run_hash_snapshot(base_fm),
        "treatment": _run_hash_snapshot(run_fm),
        "shared_metric_specs": metric_specs,
        "ignored_config_keys": sorted(config.ignored_config_keys),
        "iso_tolerance": config.iso_tolerance,
    }
    return canonical_hash(payload)


def compute_iso_compute(
    base_fm: dict[str, Any],
    treat_fm: dict[str, Any],
    iso_tolerance: dict[str, float],
) -> str:
    """Compare whichever of `params` and `gpu_hours` are shared by both runs (§4.1)."""
    b_comp = base_fm.get("compute") or {}
    t_comp = treat_fm.get("compute") or {}
    checked = False
    reduced = False
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
            checked = True
            if bf == 0:
                if tf > 0:
                    return "increased"
                continue
            tol = float(iso_tolerance.get(key, 0.02))
            rel = (tf - bf) / bf
            if rel > tol:
                return "increased"
            if rel < -tol:
                reduced = True
    if not checked:
        return "unknown"
    return "reduced" if reduced else "matched"


def _str_matches(actual: Any, declared: str) -> bool:
    if actual is None:
        return declared.lower() in ("none", "null", "")
    if isinstance(actual, bool):
        return str(actual).lower() == declared.lower()
    if isinstance(actual, (list, tuple)):
        norm = [_norm_str_item(x) for x in actual]
        decl_parts = [p.strip() for p in declared.strip("[]").split(",") if p.strip()]
        return norm == decl_parts or str(actual) == declared
    if isinstance(actual, (int, float)):
        if str(actual) == declared:
            return True
        try:
            return float(actual) == float(declared)
        except ValueError:
            return False
    return str(actual) == declared


def _norm_str_item(x: Any) -> str:
    return str(x).strip()


def compute_diff_accountability(
    base_fm: dict[str, Any],
    treat_fm: dict[str, Any],
    ignored_config_keys: set[str],
) -> dict[str, Any]:
    """Compute facet_diff, config_diff, unaccounted, and unrealized (§4.2)."""
    facet_diff: dict[str, dict[str, Any]] = {}
    for key in FACET_KEYS:
        bv = _norm_facet_value(key, base_fm.get(key))
        tv = _norm_facet_value(key, treat_fm.get(key))
        if bv != tv:
            facet_diff[key] = {"baseline": bv, "value": tv}

    b_cfg = dict(base_fm.get("config") or {})
    t_cfg = dict(treat_fm.get("config") or {})
    all_cfg_keys = sorted((set(b_cfg) | set(t_cfg)) - ignored_config_keys)
    config_diff: dict[str, dict[str, Any]] = {}
    for k in all_cfg_keys:
        bv = b_cfg.get(k)
        tv = t_cfg.get(k)
        if bv != tv:
            config_diff[k] = {"baseline": bv, "value": tv}

    b_intervs = [str(x) for x in (base_fm.get("interventions") or [])]
    t_intervs = [str(x) for x in (treat_fm.get("interventions") or [])]

    declared_kv: dict[str, list[str]] = {}
    free_text: list[str] = []
    for entry in t_intervs:
        if "=" in entry:
            k, v = entry.split("=", 1)
            declared_kv.setdefault(k.strip(), []).append(v.strip())
        else:
            free_text.append(entry)

    unaccounted: list[str] = []
    for key in sorted(set(facet_diff) | set(config_diff)):
        if key not in declared_kv:
            unaccounted.append(key)

    unrealized: list[str] = []
    for key, vals in sorted(declared_kv.items()):
        actual = treat_fm.get(key) if key in FACET_KEYS else t_cfg.get(key)
        if not any(_str_matches(actual, v) for v in vals):
            unrealized.append(f"{key}={vals[-1]}")

    b_code = base_fm.get("code") or {}
    t_code = treat_fm.get("code") or {}
    b_commit = b_code.get("commit")
    t_commit = t_code.get("commit")
    t_dirty = bool(t_code.get("dirty", False))
    same_clean_commit = (
        b_commit is not None
        and t_commit is not None
        and b_commit == t_commit
        and not t_dirty
    )
    b_interv_set = set(b_intervs)
    for entry in free_text:
        if entry not in b_interv_set and same_clean_commit:
            unrealized.append(entry)

    return {
        "facet_diff": facet_diff,
        "config_diff": config_diff,
        "unaccounted": sorted(unaccounted),
        "unrealized": unrealized,
    }


def compute_control_level(
    base_fm: dict[str, Any],
    treat_fm: dict[str, Any],
    diff_info: dict[str, Any],
) -> str:
    """Assign `invalid | confounded | exploratory | clean` (§4.2)."""
    if str(base_fm.get("validity", "ok")) != "ok" or str(treat_fm.get("validity", "ok")) != "ok":
        return "invalid"
    if diff_info["unaccounted"] or diff_info["unrealized"]:
        return "confounded"
    b_dirty = bool((base_fm.get("code") or {}).get("dirty", False))
    t_dirty = bool((treat_fm.get("code") or {}).get("dirty", False))
    is_probe = base_fm.get("tier") == "probe" or treat_fm.get("tier") == "probe"
    t_intervs = list(treat_fm.get("interventions") or [])
    is_replicate = not t_intervs and not diff_info["config_diff"]
    if is_probe or b_dirty or t_dirty or is_replicate:
        return "exploratory"
    return "clean"


def _metric_signal(delta: float, direction: str, min_delta: float | None) -> str | None:
    if min_delta is None:
        return None
    beneficial = delta if direction == "max" else -delta
    if beneficial >= min_delta:
        return "clear"
    if beneficial > 0:
        return "marginal"
    if beneficial > -min_delta:
        return "neutral"
    return "regression"


def compute_signals_and_verdict(
    base_fm: dict[str, Any],
    treat_fm: dict[str, Any],
    iso_compute: str,
    config: ContourConfig,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Compute per-metric deltas/signals and the overall run verdict (§4.3)."""
    b_metrics = {
        str(k): float(v)
        for k, v in (base_fm.get("metrics") or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    t_metrics = {
        str(k): float(v)
        for k, v in (treat_fm.get("metrics") or {}).items()
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
        tau = config.min_delta(k)

        entry: dict[str, Any] = {
            "baseline": bv,
            "value": tv,
            "delta": delta,
        }
        if kind in ("quality", "cost"):
            sig = _metric_signal(tv - bv, direction, tau)
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

    # Quality outcome Q
    if "regression" in primary_signals:
        q_out = "regression"
    elif "clear" in primary_signals:
        q_out = "clear"
    elif "marginal" in primary_signals:
        q_out = "marginal"
    else:
        q_out = "neutral"

    # Cost outcome C
    if "regression" in cost_signals or iso_compute == "increased":
        c_out = "worse"
    elif "clear" in cost_signals or iso_compute == "reduced":
        c_out = "better"
    else:
        c_out = "matched"

    verdict = _VERDICT_TABLE[(q_out, c_out)]

    # Non-primary quality metric regression downgrade
    if "regression" in non_primary_quality_signals:
        if verdict in ("win", "marginal", "efficiency_win"):
            verdict = "tradeoff"
        elif verdict == "neutral":
            verdict = "regression"

    return deltas, verdict


def recommend_relation(control_level: str, verdict: str) -> str:
    """Advisory link relation (§4.3)."""
    if control_level == "clean" and verdict in ("win", "efficiency_win"):
        return "supports"
    if control_level == "clean" and verdict == "regression":
        return "contradicts"
    return "qualifies"


def compute_vs_parent(
    run_fm: dict[str, Any],
    base_fm: dict[str, Any],
    config: ContourConfig,
) -> dict[str, Any] | None:
    """Compute fresh `vs_parent` dict for a run against its effective baseline (§4.1)."""
    if run_fm.get("status") != "final" or base_fm.get("status") != "final":
        return None
    iso_compute = compute_iso_compute(base_fm, run_fm, config.iso_tolerance)
    diff_info = compute_diff_accountability(base_fm, run_fm, config.ignored_config_keys)
    control_level = compute_control_level(base_fm, run_fm, diff_info)
    deltas, verdict = compute_signals_and_verdict(base_fm, run_fm, iso_compute, config)
    inputs_hash = compute_inputs_hash(run_fm, base_fm, config)
    return {
        "baseline": as_wikilink(str(base_fm["id"])),
        "inputs_hash": inputs_hash,
        "control_level": control_level,
        "verdict": verdict,
        "iso_compute": iso_compute,
        "facet_diff": diff_info["facet_diff"],
        "config_diff": diff_info["config_diff"],
        "unaccounted": diff_info["unaccounted"],
        "unrealized": diff_info["unrealized"],
        "deltas": deltas,
    }


def get_fresh_vs_parent(g: ContourGraph, run: ContourNode) -> dict[str, Any] | None:
    """Always recompute `vs_parent` in memory from current node states (§4.1)."""
    base_id = run.effective_baseline
    if not base_id or base_id not in g.runs:
        return None
    return compute_vs_parent(run.frontmatter, g.runs[base_id].frontmatter, g.config)


def compare_runs(g: ContourGraph, ctrl_id: str, treat_id: str) -> dict[str, Any]:
    """Ad-hoc §4 comparison between any two runs. Writes nothing."""
    ctrl = g.runs.get(ctrl_id)
    if not ctrl:
        raise ValueError(f"Unknown control run: {ctrl_id!r}")
    treat = g.runs.get(treat_id)
    if not treat:
        raise ValueError(f"Unknown treatment run: {treat_id!r}")

    iso_compute = compute_iso_compute(ctrl.frontmatter, treat.frontmatter, g.config.iso_tolerance)
    diff_info = compute_diff_accountability(
        ctrl.frontmatter, treat.frontmatter, g.config.ignored_config_keys
    )
    control_level = compute_control_level(ctrl.frontmatter, treat.frontmatter, diff_info)
    deltas, verdict = compute_signals_and_verdict(
        ctrl.frontmatter, treat.frontmatter, iso_compute, g.config
    )
    return {
        "control": ctrl.id,
        "treatment": treat.id,
        "control_level": control_level,
        "verdict": verdict,
        "recommended_link": recommend_relation(control_level, verdict),
        "iso_compute": iso_compute,
        "facet_diff": diff_info["facet_diff"],
        "config_diff": diff_info["config_diff"],
        "unaccounted": diff_info["unaccounted"],
        "unrealized": diff_info["unrealized"],
        "deltas": deltas,
    }
