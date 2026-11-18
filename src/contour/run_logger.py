"""Local-first Contour Run logger writing directly to `runs/run_<YYYYMMDD>_<8hex>.md`."""

from __future__ import annotations

import datetime as dt
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

from .compare import compute_vs_parent
from .config import ConfigError, find_root, load_config
from .graph import DEFAULT_VALIDITY, RUN_TIERS
from .mdio import as_wikilink, parse_markdown_file, strip_wiki, write_markdown_file

RUNS_DIR = "runs"

AUTHOR_KEYS = (
    "id",
    "type",
    "title",
    "validity",
    "interventions",
    "intervention_note",
    "parent_runs",
    "baseline",
    "motivated_by",
)

LOGGER_KEYS = (
    "status",
    "created_at",
    "finished_at",
    "task",
    "arch_family",
    "arch_scale",
    "dataset_train",
    "tier",
    "config",
    "compute",
    "code",
    "launch_cmd",
    "metrics",
)


def _normalize_str_list(val: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    return [str(x) for x in val]


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _detect_rank() -> int:
    for var in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def _detect_gpu_count() -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        devices = [d.strip() for d in cuda_visible.split(",") if d.strip()]
        return max(len(devices), 1)
    try:
        out = (
            subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        if out:
            return len(out.splitlines())
    except Exception:
        pass
    return 1


def _git_output(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", *args],
                cwd=str(cwd) if cwd else None,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _get_git_info(cwd: Path | None = None) -> dict[str, Any]:
    """Collect git commit, branch, and dirty status."""
    commit = _git_output(["rev-parse", "--short", "HEAD"], cwd=cwd)
    if not commit:
        return {"commit": None, "dirty": False}
    porcelain = _git_output(["status", "--porcelain"], cwd=cwd) or ""
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    info: dict[str, Any] = {
        "commit": commit,
        "dirty": bool(porcelain),
    }
    if branch:
        info["branch"] = branch
    return info


def _run_id_for(created_at: str) -> str:
    date_part = created_at[:10].replace("-", "")
    return f"run_{date_part}_{uuid.uuid4().hex[:8]}"


def _resolve_run_fm(runs_dir: Path, ref_id: str) -> dict[str, Any] | None:
    rid = strip_wiki(ref_id)
    p = runs_dir / f"{rid}.md"
    if not p.exists():
        return None
    fm, _ = parse_markdown_file(p)
    return fm


def _default_run_body(parent_refs: list[str], notes: str | None) -> str:
    ref = parent_refs[0] if parent_refs else "baseline"
    obs = f"- {notes}\n" if notes else "- \n"
    return (
        "## Key Takeaway (TL;DR)\n"
        f"- TODO: Summarize key outcome vs {ref}.\n\n"
        "## Observations & Anomalies\n"
        f"{obs}"
    )


def ingest_record(root: Path, record: dict[str, Any], notes: str | None = None) -> Path:
    """Write or update `runs/<id>.md` from a run record dict.

    Preserves author-owned keys and Markdown body if the run file already exists.
    Computes `vs_parent` on `status == 'final'` when an effective baseline is available.
    """
    runs_dir = root / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(record["id"])
    path = runs_dir / f"{run_id}.md"

    existing_fm: dict[str, Any] = {}
    body = ""
    if path.exists():
        existing_fm, body = parse_markdown_file(path)

    # Author keys: keep existing if file already exists, else seed from record
    merged: dict[str, Any] = {}
    for k in AUTHOR_KEYS:
        if path.exists() and k in existing_fm:
            merged[k] = existing_fm[k]
        else:
            merged[k] = record.get(k)

    # Logger keys: always authoritative from record
    for k in LOGGER_KEYS:
        merged[k] = record.get(k)

    # Preserve any extra author-added frontmatter keys
    known = set(AUTHOR_KEYS) | set(LOGGER_KEYS) | {"vs_parent"}
    for k, v in existing_fm.items():
        if k not in known:
            merged[k] = v

    # Derive vs_parent when status == 'final'
    vs_parent = None
    if merged.get("status") == "final":
        base_ref = merged.get("baseline") or (
            merged["parent_runs"][0] if merged.get("parent_runs") else None
        )
        if base_ref:
            base_fm = _resolve_run_fm(runs_dir, str(base_ref))
            if base_fm is not None:
                try:
                    cfg = load_config(root)
                except ConfigError:
                    cfg = None
                if cfg is not None:
                    vs_parent = compute_vs_parent(merged, base_fm, cfg)
    merged["vs_parent"] = vs_parent

    if not body.strip():
        body = _default_run_body(merged.get("parent_runs") or [], notes)

    write_markdown_file(path, merged, body)
    return path


class Run:
    """Local-first Contour run logger writing `runs/run_<YYYYMMDD>_<8hex>.md`."""

    def __init__(
        self,
        title: str | None = None,
        *,
        exp_name: str | None = None,
        task: str = "unspecified",
        dataset_train: str | list[str] = (),
        arch_family: str = "unspecified",
        arch_scale: str = "unspecified",
        interventions: list[str] | str | None = None,
        intervention_note: str | None = None,
        parent_runs: list[str] | None = None,
        baseline: str | None = None,
        motivated_by: list[str] | None = None,
        tier: str = "proxy",
        config: dict[str, Any] | None = None,
        launch_cmd: str | None = None,
        num_gpus: int | None = None,
        rank_zero_only: bool = True,
        root: str | Path | None = None,
        git_cwd: str | Path | None = None,
        notes: str | None = None,
    ) -> None:
        if tier not in RUN_TIERS:
            raise ValueError(f"tier must be one of {RUN_TIERS}, got {tier!r}")
        resolved_title = title or exp_name
        if not resolved_title:
            raise ValueError("Run requires `title`")

        self.rank = _detect_rank()
        self.enabled = (self.rank == 0) or (not rank_zero_only)

        self.root: Path | None
        if root is not None:
            self.root = find_root(explicit=Path(root))
        else:
            try:
                self.root = find_root(start=Path(git_cwd) if git_cwd else None)
            except ConfigError:
                self.root = None

        self.created_at = _utc_now_iso()
        self.finished_at: str | None = None
        self.id = _run_id_for(self.created_at)
        self.title = resolved_title
        self.validity = DEFAULT_VALIDITY
        self.status = "running"
        self.tier = tier
        self.task = task
        self.arch_family = arch_family
        self.arch_scale = arch_scale
        self.dataset_train = _normalize_str_list(dataset_train)
        self.interventions = _normalize_str_list(interventions)
        self.intervention_note = intervention_note
        self.parent_runs = [as_wikilink(p) for p in _normalize_str_list(parent_runs)]
        self.baseline = as_wikilink(baseline) if baseline else None
        self.motivated_by = [as_wikilink(m) for m in _normalize_str_list(motivated_by)]
        self.config: dict[str, Any] = dict(config or {})
        self.metrics: dict[str, float] = {}
        self.notes = notes

        self.start_time = time.monotonic()
        self.num_gpus = num_gpus if num_gpus is not None else _detect_gpu_count()
        self.compute: dict[str, Any] = {
            "num_gpus": self.num_gpus,
            "gpu_hours": None,
            "params": None,
        }
        git_dir = Path(git_cwd) if git_cwd else self.root
        self.code = _get_git_info(git_dir) if self.enabled else {"commit": None, "dirty": False}
        self.launch_cmd = launch_cmd or os.environ.get("LAUNCH_CMD") or " ".join(sys.argv)
        self._closed = False
        self.path: Path | None = None

        if self.enabled and self.root is not None:
            self.path = ingest_record(self.root, self.to_record(), notes=self.notes)

    def track(
        self,
        value: float,
        name: str,
        step: int | None = None,  # accepted for training-loop compatibility
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a scalar metric (`<scope>/<name>` when `context={'subset': ...}` is used)."""
        if not self.enabled:
            return
        val = float(value)
        if not math.isfinite(val):
            raise ValueError(f"Metric {name!r} must be finite, got {value!r}")
        key = f"{context['subset']}/{name}" if context and "subset" in context else name
        self.metrics[key] = round(val, 6)

    def set_compute_budget(
        self,
        *,
        gpu_hours: float | None = None,
        params: int | None = None,
        num_gpus: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        if gpu_hours is not None:
            self.compute["gpu_hours"] = round(float(gpu_hours), 3)
        if params is not None:
            self.compute["params"] = int(params)
        if num_gpus is not None:
            self.compute["num_gpus"] = int(num_gpus)

    def to_record(self) -> dict[str, Any]:
        """Serialize Run fields into a canonical record dict."""
        return {
            "id": self.id,
            "type": "run",
            "title": self.title,
            "validity": self.validity,
            "interventions": list(self.interventions),
            "intervention_note": self.intervention_note,
            "parent_runs": list(self.parent_runs),
            "baseline": self.baseline,
            "motivated_by": list(self.motivated_by),
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "task": self.task,
            "arch_family": self.arch_family,
            "arch_scale": self.arch_scale,
            "dataset_train": list(self.dataset_train),
            "tier": self.tier,
            "config": dict(self.config),
            "compute": {k: v for k, v in self.compute.items() if v is not None},
            "code": dict(self.code),
            "launch_cmd": self.launch_cmd,
            "metrics": dict(self.metrics),
        }

    def close(self, failed: bool = False) -> Path | None:
        if self._closed or not self.enabled:
            return self.path
        self._closed = True
        wall = max(time.monotonic() - self.start_time, 0.0)
        if self.compute.get("gpu_hours") is None:
            self.compute["gpu_hours"] = round((wall / 3600.0) * self.num_gpus, 6)
        self.finished_at = _utc_now_iso()
        if self.status == "running":
            self.status = "failed" if failed else "final"
        if self.root is not None:
            self.path = ingest_record(self.root, self.to_record(), notes=self.notes)
        return self.path

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close(failed=exc_type is not None)


def init_run(**kwargs: Any) -> Run:
    """Convenience constructor for `Run`."""
    return Run(**kwargs)
