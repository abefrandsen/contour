"""Controlled comparisons between runs: embedded `vs_parent` diffs and on-the-fly `contour compare`."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .bundle import TIER_ALIASES
from .config import ContourConfig
from .graph import ContourGraph, ContourNode
from .haze import load_noise, lookup_noise, regime_key, sample_std
from .mdio import as_wikilink

TIER_ORDER = {"probe": 0, "proxy": 1, "full": 2}
SNR_THRESHOLD = 2.0


def _norm_tier(raw: Any) -> str:
    s = str(raw or "proxy")
    return TIER_ALIASES.get(s, s)


def _lowest_tier(fms: list[dict[str, Any]]) -> str:
    tiers = [_norm_tier(r.get("tier")) for r in fms]
    return min(tiers, key=lambda t: TIER_ORDER.get(t, 1))


def _arm_compute_values(fms: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for r in fms:
        comp = r.get("compute") or {}
        v = comp.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def _check_iso_compute(
    controls: list[dict[str, Any]], treatments: list[dict[str, Any]], tol: float
) -> str:
    """Return 'verified', 'reduced', 'violated', or 'unknown'.

    Using > (1 + tol) params/FLOPs in treatment is a compute confound ('violated').
    Using fewer params/FLOPs is a compression/efficiency reduction ('reduced'), not a confound.
    """
    checked = False
    reduced = False
    for key in ("params", "flops"):
        c_vals = _arm_compute_values(controls, key)
        t_vals = _arm_compute_values(treatments, key)
        if len(c_vals) == len(controls) and len(t_vals) == len(treatments) and c_vals and t_vals:
            checked = True
            c_mean = sum(c_vals) / len(c_vals)
            t_mean = sum(t_vals) / len(t_vals)
            if c_mean > 0:
                rel_diff = (t_mean - c_mean) / c_mean
                if rel_diff > tol:
                    return "violated"
                if rel_diff < -tol:
                    reduced = True
    if not checked:
        return "unknown"
    return "reduced" if reduced else "verified"


def _load_bundle_configs(root: Path, fms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for r in fms:
        bpath = r.get("bundle")
        if not bpath:
            continue
        p = root / str(bpath)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data.get("config"), dict):
                    configs.append(data["config"])
            except Exception:
                pass
    return configs


def _is_usable_fm(fm: dict[str, Any]) -> bool:
    return str(fm.get("status", "unknown")) == "final" and str(fm.get("validity", "ok")) == "ok"


def _interventions_set(fm: dict[str, Any]) -> frozenset[str]:
    return frozenset(str(x) for x in (fm.get("interventions") or []))


def _evals_list(fm: dict[str, Any]) -> list[str]:
    return [str(x) for x in (fm.get("dataset_eval") or [])]


def compute_facet_diff_from_fms(
    root: Path,
    controls: list[dict[str, Any]],
    treatments: list[dict[str, Any]],
    iso_tol: float = 0.02,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute facet differences and confound flags between control and treatment frontmatters."""
    all_runs = controls + treatments
    confounding_facets: dict[str, list[Any]] = {}

    for key in ("task", "arch_family", "arch_scale"):
        vals = sorted({str(r.get(key) or "") for r in all_runs})
        if len(vals) > 1:
            confounding_facets[key] = vals

    tier_vals = sorted({_norm_tier(r.get("tier")) for r in all_runs})
    if len(tier_vals) > 1:
        confounding_facets["tier"] = tier_vals

    train_sets = sorted({tuple(sorted(str(x) for x in (r.get("dataset_train") or []))) for r in all_runs})
    if len(train_sets) > 1:
        confounding_facets["dataset_train"] = [list(ts) for ts in train_sets]

    c_intervs = {_interventions_set(r) for r in controls}
    t_intervs = {_interventions_set(r) for r in treatments}
    arm_inconsistent = len(c_intervs) > 1 or len(t_intervs) > 1

    c_union: set[str] = set().union(*c_intervs) if c_intervs else set()
    t_union: set[str] = set().union(*t_intervs) if t_intervs else set()
    added = sorted(t_union - c_union)
    removed = sorted(c_union - t_union)

    c_evals = set().union(*(set(_evals_list(r)) for r in controls)) if controls else set()
    t_evals = set().union(*(set(_evals_list(r)) for r in treatments)) if treatments else set()
    shared_evals = sorted(c_evals & t_evals)

    c_cfgs = _load_bundle_configs(root, controls)
    t_cfgs = _load_bundle_configs(root, treatments)
    cfg_diff: dict[str, Any] = {}
    if len(c_cfgs) == len(controls) and len(t_cfgs) == len(treatments) and c_cfgs and t_cfgs:
        all_keys = sorted({k for d in (c_cfgs + t_cfgs) for k in d})
        for k in all_keys:
            cv = [d.get(k) for d in c_cfgs]
            tv = [d.get(k) for d in t_cfgs]
            if cv != tv:
                cfg_diff[k] = {
                    "control": cv[0] if len(set(map(str, cv))) == 1 else cv,
                    "treatment": tv[0] if len(set(map(str, tv))) == 1 else tv,
                }

    iso_compute = _check_iso_compute(controls, treatments, tol=iso_tol)
    all_usable = all(_is_usable_fm(r) for r in all_runs)
    has_intervention_diff = bool(added or removed)
    single_logical_diff = (
        len(confounding_facets) == 0
        and not arm_inconsistent
        and has_intervention_diff
    )

    facet_diff = {
        "confounding_facets": confounding_facets,
        "added_interventions": added,
        "removed_interventions": removed,
        "arm_inconsistent": arm_inconsistent,
        "shared_evals": shared_evals,
        "config_diff": cfg_diff,
    }
    confounds = {
        "single_logical_diff": single_logical_diff,
        "iso_compute": iso_compute,
        "tier": _lowest_tier(all_runs),
        "all_usable": all_usable,
        "n_control": len(controls),
        "n_treatment": len(treatments),
    }
    return facet_diff, confounds


def _series_map_fm(fm: dict[str, Any], is_telemetry: bool) -> dict[str, float]:
    raw = fm.get("telemetry" if is_telemetry else "metrics") or {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def compute_metric_deltas_from_fms(
    controls: list[dict[str, Any]],
    treatments: list[dict[str, Any]],
    noise: dict[str, dict[str, dict[str, Any]]],
    regime: str | None,
    is_telemetry: bool = False,
) -> dict[str, dict[str, Any]]:
    c_keys = {k for r in controls for k in _series_map_fm(r, is_telemetry)}
    t_keys = {k for r in treatments for k in _series_map_fm(r, is_telemetry)}
    shared = sorted(c_keys & t_keys)

    out: dict[str, dict[str, Any]] = {}
    for k in shared:
        c_vals = [_series_map_fm(r, is_telemetry)[k] for r in controls if k in _series_map_fm(r, is_telemetry)]
        t_vals = [_series_map_fm(r, is_telemetry)[k] for r in treatments if k in _series_map_fm(r, is_telemetry)]
        if not c_vals or not t_vals:
            continue
        c_mean = sum(c_vals) / len(c_vals)
        t_mean = sum(t_vals) / len(t_vals)
        delta = t_mean - c_mean
        rel_pct = (delta / abs(c_mean) * 100.0) if c_mean != 0 else None

        se_diff: float | None = None
        ci_95: list[float] | None = None
        if len(c_vals) >= 2 and len(t_vals) >= 2:
            sc = sample_std(c_vals)
            st = sample_std(t_vals)
            se_diff = math.sqrt((sc ** 2) / len(c_vals) + (st ** 2) / len(t_vals))
            margin = 1.96 * se_diff
            ci_95 = [round(delta - margin, 6), round(delta + margin, 6)]

        noise_entry = lookup_noise(noise, regime, k) if regime else None
        noise_std = float(noise_entry["std"]) if noise_entry and "std" in noise_entry else None

        noise_scale = se_diff if (se_diff is not None and se_diff > 0) else (
            noise_std if (noise_std is not None and noise_std > 0) else None
        )
        snr = round(abs(delta) / noise_scale, 3) if noise_scale else None
        clears_noise = (snr >= SNR_THRESHOLD) if snr is not None else None

        out[k] = {
            "control_mean": round(c_mean, 6),
            "treatment_mean": round(t_mean, 6),
            "delta": round(delta, 6),
            "rel_delta_pct": round(rel_pct, 3) if rel_pct is not None else None,
            "n_control": len(c_vals),
            "n_treatment": len(t_vals),
            "se_diff": round(se_diff, 6) if se_diff is not None else None,
            "ci_95": ci_95,
            "noise_std": noise_std,
            "snr": snr,
            "clears_noise": clears_noise,
        }
    return out


def compute_grade(
    confounds: dict[str, Any],
    metric_deltas: dict[str, dict[str, Any]],
    config: ContourConfig | None = None,
) -> str:
    """Assign ordinal evidence quality badge A / B / C."""
    if (
        not confounds["all_usable"]
        or not confounds["single_logical_diff"]
        or _norm_tier(confounds["tier"]) == "probe"
        or confounds["iso_compute"] == "violated"
    ):
        return "C"

    quality_deltas = [
        d for k, d in metric_deltas.items()
        if not (config.is_cost_metric(k) if config else (k.lower().startswith("sys/") or "throughput" in k.lower()))
    ]
    if any(d.get("clears_noise") is True for d in quality_deltas):
        return "A"
    if confounds["iso_compute"] == "reduced" and any(d.get("clears_noise") is True for d in metric_deltas.values()):
        return "A"
    return "B"


def _resolve_regime(
    facet_diff: dict[str, Any], first_treatment: dict[str, Any]
) -> str | None:
    shared_evals = facet_diff["shared_evals"]
    if (
        shared_evals
        and not facet_diff["confounding_facets"].get("task")
        and not facet_diff["confounding_facets"].get("arch_scale")
    ):
        return regime_key(
            str(first_treatment.get("task") or "unspecified"),
            shared_evals[0],
            str(first_treatment.get("arch_scale") or "unspecified"),
        )
    return None


def compute_vs_parent(
    root: Path,
    child_fm: dict[str, Any],
    parent_fm: dict[str, Any],
    haze: dict[str, dict[str, dict[str, Any]]] | None = None,
    config: ContourConfig | None = None,
    iso_tol: float = 0.02,
) -> dict[str, Any]:
    """Compute the embedded `vs_parent` summary stored on a child FieldLog."""
    noise_data = haze if haze is not None else load_noise(root)
    facet_diff, confounds = compute_facet_diff_from_fms(root, [parent_fm], [child_fm], iso_tol=iso_tol)
    regime = _resolve_regime(facet_diff, child_fm)
    metric_deltas = compute_metric_deltas_from_fms([parent_fm], [child_fm], noise_data, regime, is_telemetry=False)
    grade = compute_grade(confounds, metric_deltas, config=config)

    compact_deltas: dict[str, dict[str, Any]] = {}
    for k, d in metric_deltas.items():
        entry: dict[str, Any] = {
            "parent": d["control_mean"],
            "value": d["treatment_mean"],
            "delta": d["delta"],
        }
        if d.get("rel_delta_pct") is not None:
            entry["rel_delta_pct"] = d["rel_delta_pct"]
        if d.get("noise_std") is not None:
            entry["noise_std"] = d["noise_std"]
            entry["snr"] = d["snr"]
            entry["clears_noise"] = d["clears_noise"]
        compact_deltas[k] = entry

    out: dict[str, Any] = {
        "parent": as_wikilink(str(parent_fm["id"])),
        "grade": grade,
        "single_logical_diff": confounds["single_logical_diff"],
        "iso_compute": confounds["iso_compute"],
        "confounding_facets": facet_diff["confounding_facets"],
        "added_interventions": facet_diff["added_interventions"],
        "removed_interventions": facet_diff["removed_interventions"],
        "config_diff": facet_diff["config_diff"],
        "regime": regime,
        "deltas": compact_deltas,
    }
    return out


def compare_runs(
    g: ContourGraph,
    control_ids: list[str],
    treatment_ids: list[str],
    iso_tol: float = 0.02,
) -> dict[str, Any]:
    """Compute an on-the-fly comparison between control and treatment FieldLogs."""
    controls: list[ContourNode] = []
    for cid in control_ids:
        node = g.fieldlogs.get(cid)
        if not node:
            raise ValueError(f"Unknown control fieldlog: {cid!r}")
        controls.append(node)
    treatments: list[ContourNode] = []
    for tid in treatment_ids:
        node = g.fieldlogs.get(tid)
        if not node:
            raise ValueError(f"Unknown treatment fieldlog: {tid!r}")
        treatments.append(node)

    c_fms = [c.frontmatter for c in controls]
    t_fms = [t.frontmatter for t in treatments]

    facet_diff, confounds = compute_facet_diff_from_fms(g.root, c_fms, t_fms, iso_tol=iso_tol)
    regime = _resolve_regime(facet_diff, t_fms[0])
    noise = load_noise(g.root)
    metric_deltas = compute_metric_deltas_from_fms(c_fms, t_fms, noise, regime, is_telemetry=False)
    telemetry_deltas = compute_metric_deltas_from_fms(c_fms, t_fms, noise, regime, is_telemetry=True)
    grade = compute_grade(confounds, metric_deltas, config=g.config)

    return {
        "controls": [c.id for c in controls],
        "treatments": [t.id for t in treatments],
        "grade": grade,
        "tier": confounds["tier"],
        "regime": regime,
        "confounds": confounds,
        "facet_diff": facet_diff,
        "metrics": metric_deltas,
        "telemetry": telemetry_deltas,
    }
