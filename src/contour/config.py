"""Contour root discovery and `contour.yaml` configuration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any

import yaml

CONFIG_FILE = "contour.yaml"
CONFIG_VERSION = 1
SUBDIR_CANDIDATES = ("contour", ".contour", "research")

VALID_DIRECTIONS = ("max", "min")
VALID_KINDS = ("quality", "cost", "diagnostic")
METRIC_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[a-z0-9_]+$")

DEFAULT_IGNORED_CONFIG_KEYS: list[str] = [
    "seed",
    "num_workers",
    "log_interval",
    "eval_interval",
    "checkpoint_dir",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "contour": CONFIG_VERSION,
    "ignored_config_keys": list(DEFAULT_IGNORED_CONFIG_KEYS),
    "metrics": {},
}

_LOWER_IS_BETTER_TOKENS = ("loss", "err", "perplexity", "ppl", "mse", "mae", "rmse", "nll", "fid", "latency")


class ConfigError(ValueError):
    pass


def is_valid_metric_key(key: str) -> bool:
    return bool(METRIC_KEY_RE.match(key))


@dataclass
class ContourConfig:
    root: Path
    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))

    @property
    def metrics(self) -> dict[str, dict[str, Any]]:
        return dict(self.data.get("metrics") or {})

    @property
    def ignored_config_keys(self) -> set[str]:
        return {str(k) for k in (self.data.get("ignored_config_keys") or [])}

    def metric_spec(self, key: str) -> dict[str, Any]:
        return dict((self.data.get("metrics") or {}).get(key) or {})

    def has_metric(self, key: str) -> bool:
        return key in (self.data.get("metrics") or {})

    def metric_direction(self, key: str) -> str:
        """Return 'max' or 'min' for a metric key."""
        spec = self.metric_spec(key)
        if spec.get("direction") in VALID_DIRECTIONS:
            return str(spec["direction"])
        leaf = key.lower().rsplit("/", 1)[-1]
        if any(tok in leaf for tok in _LOWER_IS_BETTER_TOKENS):
            return "min"
        return "max"

    def metric_kind(self, key: str) -> str:
        """Return 'quality', 'cost', or 'diagnostic'."""
        spec = self.metric_spec(key)
        if spec.get("kind") in VALID_KINDS:
            return str(spec["kind"])
        if key.lower().startswith("sys/") or "throughput" in key.lower():
            return "cost"
        return "quality"

    def is_cost_metric(self, key: str) -> bool:
        return self.metric_kind(key) == "cost"

    def is_primary_metric(self, key: str) -> bool:
        spec = self.metric_spec(key)
        return bool(spec.get("primary", False)) and self.metric_kind(key) == "quality"

    def min_delta(self, key: str) -> float | None:
        spec = self.metric_spec(key)
        val = spec.get("min_delta")
        return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None

    def min_rel_delta(self, key: str) -> float | None:
        spec = self.metric_spec(key)
        val = spec.get("min_rel_delta")
        return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None

    def has_threshold(self, key: str) -> bool:
        return self.min_delta(key) is not None or self.min_rel_delta(key) is not None


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def validate_config(data: dict[str, Any], *, require_threshold: bool = True) -> list[str]:
    errors: list[str] = []
    if data.get("contour") != CONFIG_VERSION:
        errors.append(f"`contour:` must be {CONFIG_VERSION} (config format version)")

    ignored = data.get("ignored_config_keys")
    if ignored is not None and (not isinstance(ignored, list) or not all(isinstance(x, str) for x in ignored)):
        errors.append("ignored_config_keys must be a list of strings")

    metrics = data.get("metrics") or {}
    if not isinstance(metrics, dict):
        errors.append("metrics must be a mapping")
        return errors

    for key, spec in metrics.items():
        if not is_valid_metric_key(str(key)):
            errors.append(
                f"metrics.{key!r} must match '<scope>/<name>' where <name> is lowercase [a-z0-9_]+"
            )
        if not isinstance(spec, dict):
            errors.append(f"metrics.{key} must be a mapping")
            continue
        direction = spec.get("direction")
        if direction not in VALID_DIRECTIONS:
            errors.append(f"metrics.{key}.direction must be 'max' or 'min'")
        kind = spec.get("kind")
        if kind not in VALID_KINDS:
            errors.append(f"metrics.{key}.kind must be 'quality', 'cost', or 'diagnostic'")
        if "primary" in spec and not isinstance(spec["primary"], bool):
            errors.append(f"metrics.{key}.primary must be a boolean")

        md = spec.get("min_delta")
        mrd = spec.get("min_rel_delta")
        if md is not None and (not isinstance(md, (int, float)) or isinstance(md, bool) or float(md) < 0):
            errors.append(f"metrics.{key}.min_delta must be >= 0")
        if mrd is not None and (not isinstance(mrd, (int, float)) or isinstance(mrd, bool) or float(mrd) < 0):
            errors.append(f"metrics.{key}.min_rel_delta must be >= 0")

        if kind in ("quality", "cost") and md is None and mrd is None and require_threshold:
            errors.append(
                f"metrics.{key} ({kind}) requires `min_delta` and/or `min_rel_delta` (>= 0)"
            )

    return errors


def find_root(start: Path | None = None, explicit: Path | None = None) -> Path:
    """Locate the Contour root: --root > $CONTOUR_ROOT > nearest ancestor (or its contour/ subdir) containing contour.yaml."""
    if explicit is not None:
        root = explicit.resolve()
        if (root / CONFIG_FILE).exists():
            return root
        for sub in SUBDIR_CANDIDATES:
            cand = root / sub
            if (cand / CONFIG_FILE).exists():
                return cand
        raise ConfigError(f"{root} has no {CONFIG_FILE}. Run `contour init` there first.")
    env = os.environ.get("CONTOUR_ROOT")
    if env:
        return find_root(explicit=Path(env))
    cur = (start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        if (d / CONFIG_FILE).exists():
            return d
        for sub in SUBDIR_CANDIDATES:
            cand = d / sub
            if (cand / CONFIG_FILE).exists():
                return cand
    raise ConfigError(
        f"No {CONFIG_FILE} found in {cur} (or its {SUBDIR_CANDIDATES} subdirectories) or any parent directory. "
        "Run `contour init` to create a Contour root, or pass --root."
    )


def load_config(root: Path) -> ContourConfig:
    path = root / CONFIG_FILE
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: must be a YAML mapping")
    data = _merge(DEFAULT_CONFIG, raw)
    errors = validate_config(data, require_threshold=False)
    if errors:
        raise ConfigError(f"{path}:\n  - " + "\n  - ".join(errors))
    return ContourConfig(root=root, data=data)
