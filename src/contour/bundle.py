"""Field Bundle v1: the platform-agnostic run-record contract.

Any training platform (a Theodolite-instrumented script, an Aim/W&B exporter, a SLURM
epilogue, a notebook) emits one JSON object per run. `contour sync` validates it, archives a
normalized copy under `bundles/`, and distills it into a `fieldlogs/log_*.md` card.

See docs/field_bundle.md for the human-readable specification.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import re
from typing import Any

SCHEMA_ID = "contour.field_bundle/v1"

RUN_STATUSES = ("running", "final", "failed")
TERMINAL_STATUSES = ("final", "failed")
TIERS = ("probe", "proxy", "full")
TIER_ALIASES = {"recce": "probe", "approach": "proxy", "summit": "full"}
DEFAULT_TIER = "proxy"

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{5,127}$")
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_.\-=+]*$")
NODE_REF_RE = re.compile(r"^[a-z]+_[A-Za-z0-9_.\-]+$")

_TOP_KEYS = {
    "schema", "run_id", "title", "status", "created_at", "finished_at",
    "code", "launch", "config", "data", "facets", "interventions", "intervention_note",
    "lineage", "compute", "metrics", "telemetry", "artifacts", "notes",
}
_REQUIRED = ("schema", "run_id", "status", "created_at", "facets", "data")
_CODE_KEYS = {"repo", "branch", "commit", "dirty", "dirty_hash", "dirty_patch", "untracked"}
_LAUNCH_KEYS = {"cmd", "host", "job_id", "node"}
_DATA_KEYS = {"train", "eval"}
_FACET_KEYS = {"task", "arch_family", "arch_scale", "tier", "seed"}
_LINEAGE_KEYS = {"parents", "motivated_by"}
_COMPUTE_KEYS = {"gpu_hours", "num_gpus", "wall_seconds", "flops", "params", "tokens_or_samples"}


class BundleError(ValueError):
    def __init__(self, errors: list[str], source: str = "bundle") -> None:
        self.errors = errors
        super().__init__(f"{source}: invalid field bundle:\n  - " + "\n  - ".join(errors))


def _parse_ts(val: Any) -> dt.datetime | None:
    if isinstance(val, dt.datetime):
        ts = val
    elif isinstance(val, dt.date):
        ts = dt.datetime(val.year, val.month, val.day)
    elif isinstance(val, str):
        try:
            ts = dt.datetime.fromisoformat(val.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _iso(ts: dt.datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _str_list(val: Any, path: str, errors: list[str]) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)) and all(isinstance(x, str) for x in val):
        return list(val)
    errors.append(f"{path}: must be a string or list of strings")
    return []


def _unknown(obj: dict[str, Any], allowed: set[str], path: str, errors: list[str]) -> None:
    for k in obj:
        if k not in allowed:
            errors.append(f"{path}.{k}: unknown field (allowed: {sorted(allowed)})")


def _sub(bundle: dict[str, Any], key: str, allowed: set[str], errors: list[str]) -> dict[str, Any]:
    val = bundle.get(key)
    if val is None:
        return {}
    if not isinstance(val, dict):
        errors.append(f"{key}: must be an object")
        return {}
    _unknown(val, allowed, key, errors)
    return dict(val)


def _scalars(val: Any, path: str, errors: list[str]) -> dict[str, float]:
    if val is None:
        return {}
    if not isinstance(val, dict):
        errors.append(f"{path}: must be an object of name -> number")
        return {}
    out: dict[str, float] = {}
    for k, v in val.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{path}.{k}: must be a number, got {type(v).__name__}")
            continue
        if not math.isfinite(float(v)):
            errors.append(f"{path}.{k}: must be finite (got {v}); record divergence in `notes` instead")
            continue
        out[str(k)] = float(v)
    return out


def validate_bundle(raw: dict[str, Any], source: str = "bundle") -> dict[str, Any]:
    """Validate a raw bundle and return a normalized copy. Raises BundleError listing every problem."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        raise BundleError(["top level must be a JSON object"], source)
    b = copy.deepcopy(raw)

    for key in _REQUIRED:
        if key not in b:
            errors.append(f"{key}: required")
    _unknown(b, _TOP_KEYS, "<root>", errors)

    if b.get("schema", SCHEMA_ID) != SCHEMA_ID:
        errors.append(f"schema: expected {SCHEMA_ID!r}, got {b.get('schema')!r}")

    run_id = b.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not RUN_ID_RE.match(run_id)):
        errors.append("run_id: 6-128 chars of [A-Za-z0-9_.:-], must be globally unique (e.g. a UUID)")

    if b.get("status") is not None and b.get("status") not in RUN_STATUSES:
        errors.append(f"status: must be one of {RUN_STATUSES}")

    created = _parse_ts(b.get("created_at")) if "created_at" in b else None
    if "created_at" in b and created is None:
        errors.append("created_at: must be an ISO-8601 timestamp")
    finished = None
    if b.get("finished_at") is not None:
        finished = _parse_ts(b["finished_at"])
        if finished is None:
            errors.append("finished_at: must be an ISO-8601 timestamp")
    if b.get("status") in TERMINAL_STATUSES and b.get("finished_at") is None:
        errors.append("finished_at: required when status is final or failed")

    facets = _sub(b, "facets", _FACET_KEYS, errors)
    for k in ("task", "arch_family", "arch_scale"):
        if k not in facets:
            errors.append(f"facets.{k}: required")
        elif not isinstance(facets[k], str) or not facets[k]:
            errors.append(f"facets.{k}: must be a non-empty string")
    facets.setdefault("tier", DEFAULT_TIER)
    if isinstance(facets["tier"], str) and facets["tier"] in TIER_ALIASES:
        facets["tier"] = TIER_ALIASES[facets["tier"]]
    if facets["tier"] not in TIERS:
        errors.append(f"facets.tier: must be one of {TIERS}")
    seed = facets.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        errors.append("facets.seed: must be an integer or null")
    facets.setdefault("seed", None)

    data = _sub(b, "data", _DATA_KEYS, errors)
    data = {
        "train": _str_list(data.get("train"), "data.train", errors),
        "eval": _str_list(data.get("eval"), "data.eval", errors),
    }
    if "data" in b and not data["eval"]:
        errors.append("data.eval: at least one eval dataset is required")

    interventions = _str_list(b.get("interventions"), "interventions", errors)
    for tag in interventions:
        if not TAG_RE.match(tag):
            errors.append(
                f"interventions: {tag!r} is not an atomic tag (lowercase [a-z0-9_.-=+], e.g. 'qk_rmsnorm', 'lr_peak=2x')"
            )
    if len(set(interventions)) != len(interventions):
        errors.append("interventions: duplicate tags")
    interventions = sorted(set(interventions))

    lineage = _sub(b, "lineage", _LINEAGE_KEYS, errors)
    lineage = {
        "parents": _str_list(lineage.get("parents"), "lineage.parents", errors),
        "motivated_by": _str_list(lineage.get("motivated_by"), "lineage.motivated_by", errors),
    }
    for ref in lineage["motivated_by"]:
        if not NODE_REF_RE.match(ref):
            errors.append(f"lineage.motivated_by: {ref!r} must be a node id such as 'crn_...' or 'expd_...'")

    code = _sub(b, "code", _CODE_KEYS, errors)
    if "dirty" in code and not isinstance(code["dirty"], bool):
        errors.append("code.dirty: must be a boolean")
    if "untracked" in code:
        code["untracked"] = _str_list(code["untracked"], "code.untracked", errors)
    launch = _sub(b, "launch", _LAUNCH_KEYS, errors)

    compute = _sub(b, "compute", _COMPUTE_KEYS, errors)
    for k, v in compute.items():
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0):
            errors.append(f"compute.{k}: must be a non-negative number or null")

    metrics = _scalars(b.get("metrics"), "metrics", errors)
    telemetry = _scalars(b.get("telemetry"), "telemetry", errors)
    if b.get("status") == "final" and not metrics:
        errors.append("metrics: a final run must report at least one external metric")

    artifacts = b.get("artifacts") or {}
    if not isinstance(artifacts, dict) or not all(isinstance(v, str) for v in artifacts.values()):
        errors.append("artifacts: must be an object of tag -> URI string")
        artifacts = {}

    config = b.get("config") or {}
    if not isinstance(config, dict):
        errors.append("config: must be an object")
        config = {}

    if errors:
        raise BundleError(errors, source)

    return {
        "schema": SCHEMA_ID,
        "run_id": run_id,
        "title": str(b.get("title") or run_id),
        "status": b["status"],
        "created_at": _iso(created),  # type: ignore[arg-type]
        "finished_at": _iso(finished) if finished else None,
        "code": code,
        "launch": launch,
        "config": config,
        "data": data,
        "facets": facets,
        "interventions": interventions,
        "intervention_note": b.get("intervention_note"),
        "lineage": lineage,
        "compute": compute,
        "metrics": metrics,
        "telemetry": telemetry,
        "artifacts": dict(artifacts),
        "notes": b.get("notes"),
    }


def canonical_json(bundle: dict[str, Any]) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def fieldlog_id_for(run_id: str, created_at: str) -> str:
    """Deterministic, collision-resistant fieldlog id: log_<YYYYMMDD>_<sha256(run_id)[:8]>."""
    ts = _parse_ts(created_at)
    date = ts.strftime("%Y%m%d") if ts else "00000000"
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    return f"log_{date}_{digest}"
