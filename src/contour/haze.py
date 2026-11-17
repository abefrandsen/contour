"""`noise.yaml`: noise-floor registry (seed-to-seed std per regime and metric).

A regime key identifies the comparable baseline setting, by default:
    `<task>/<eval>/<arch_scale>`  (e.g. `image_classification/imagenet1k_val/22M`)
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import yaml

from .graph import ContourNode

NOISE_FILE = "noise.yaml"
LEGACY_HAZE_FILE = "haze.yaml"


def regime_key(task: str, eval_ds: str, arch_scale: str) -> str:
    return f"{task}/{eval_ds}/{arch_scale}"


def load_noise(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = root / NOISE_FILE
    if not path.exists():
        legacy = root / LEGACY_HAZE_FILE
        if legacy.exists():
            path = legacy
        else:
            return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: must be a YAML mapping")
    return raw


def save_noise(root: Path, data: dict[str, dict[str, dict[str, Any]]]) -> Path:
    path = root / NOISE_FILE
    path.write_text(
        yaml.safe_dump(data, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def lookup_noise(
    noise: dict[str, dict[str, dict[str, Any]]],
    regime: str,
    metric: str,
) -> dict[str, Any] | None:
    return (noise.get(regime) or {}).get(metric)


def sample_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def seed_noise_from_runs(
    root: Path,
    regime: str,
    runs: list[ContourNode],
    metrics_filter: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Estimate sample std per metric across `runs` and record in `noise.yaml`."""
    if len(runs) < 2:
        raise ValueError("Need at least 2 runs (3+ recommended) to estimate noise floor.")

    # Collect metrics present in >= 2 runs
    by_metric: dict[str, list[float]] = {}
    for r in runs:
        for k, v in r.metrics.items():
            if metrics_filter and k not in metrics_filter:
                continue
            by_metric.setdefault(k, []).append(float(v))

    entries: dict[str, dict[str, Any]] = {}
    today = dt.date.today().isoformat()
    source_ids = [r.id for r in runs]
    for k, vals in sorted(by_metric.items()):
        if len(vals) < 2:
            continue
        entries[k] = {
            "std": round(sample_std(vals), 6),
            "mean": round(sum(vals) / len(vals), 6),
            "n_seeds": len(vals),
            "source_runs": source_ids,
            "updated_at": today,
        }

    if not entries:
        raise ValueError("No metrics appeared in at least 2 of the given runs.")

    data = load_noise(root)
    data.setdefault(regime, {}).update(entries)
    save_noise(root, data)
    return entries


# Backward-compatible aliases
load_haze = load_noise
save_haze = save_noise
lookup_haze = lookup_noise
seed_haze_from_runs = seed_noise_from_runs
