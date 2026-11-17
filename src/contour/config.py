"""Contour root discovery and `contour.yaml` configuration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILE = "contour.yaml"
CONFIG_VERSION = 1
SUBDIR_CANDIDATES = ("contour", ".contour", "research")

DEFAULT_CONFIG: dict[str, Any] = {
    "contour": CONFIG_VERSION,
    "inbox": ["field_bundles"],
    "metrics": {},
}

# Fallback heuristic when a metric has no explicit `direction` in contour.yaml.
_LOWER_IS_BETTER_TOKENS = ("loss", "err", "perplexity", "ppl", "mse", "mae", "rmse", "nll", "fid", "latency")


class ConfigError(ValueError):
    pass


@dataclass
class ContourConfig:
    root: Path
    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))

    @property
    def inbox(self) -> list[Path]:
        return [self.root / p for p in (self.data.get("inbox") or [])]

    def metric_direction(self, key: str) -> str:
        """Return 'max' or 'min' for a metric key."""
        spec = (self.data.get("metrics") or {}).get(key) or {}
        if spec.get("direction") in ("max", "min"):
            return spec["direction"]
        leaf = key.lower().rsplit("/", 1)[-1]
        if any(tok in leaf for tok in _LOWER_IS_BETTER_TOKENS):
            return "min"
        return "max"

    def is_cost_metric(self, key: str) -> bool:
        """Metrics that describe throughput/cost rather than quality (excluded from 'wins')."""
        spec = (self.data.get("metrics") or {}).get(key) or {}
        if "kind" in spec:
            return spec["kind"] == "cost"
        return key.lower().startswith("sys/") or "throughput" in key.lower()


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def validate_config(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("contour") != CONFIG_VERSION:
        errors.append(f"`contour:` must be {CONFIG_VERSION} (config format version)")
    if not isinstance(data.get("inbox") or [], list):
        errors.append("inbox must be a list of relative directory paths")
    for key, spec in (data.get("metrics") or {}).items():
        if not isinstance(spec, dict):
            errors.append(f"metrics.{key} must be a mapping")
            continue
        if "direction" in spec and spec["direction"] not in ("max", "min"):
            errors.append(f"metrics.{key}.direction must be 'max' or 'min'")
        if "kind" in spec and spec["kind"] not in ("quality", "cost", "telemetry"):
            errors.append(f"metrics.{key}.kind must be 'quality', 'cost', or 'telemetry'")
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
    errors = validate_config(data)
    if errors:
        raise ConfigError(f"{path}:\n  - " + "\n  - ".join(errors))
    return ContourConfig(root=root, data=data)
