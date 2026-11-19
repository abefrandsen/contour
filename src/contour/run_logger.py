"""Local-first Contour Run & Analysis loggers writing lean Markdown + sidecar YAML configs."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

import yaml

from .config import ConfigError, find_root
from .graph import ANALYSIS_KINDS, normalize_artifacts
from .mdio import as_wikilink, parse_markdown_file, write_markdown_file

RUNS_DIR = "runs"
RUN_CONFIGS_SUBDIR = "runs/configs"
ANALYSES_DIR = "analyses"

RUN_AUTHOR_KEYS = (
    "title",
    "bugged",
    "parent",
    "motivated_by",
)

RUN_LOGGER_KEYS = (
    "status",
    "created_at",
    "finished_at",
    "config_path",
    "config",
    "compute",
    "code",
    "launch_cmd",
    "artifacts",
    "metrics",
)

ANALYSIS_AUTHOR_KEYS = (
    "kind",
    "title",
    "bugged",
    "subjects",
    "motivated_by",
)

ANALYSIS_LOGGER_KEYS = (
    "status",
    "created_at",
    "finished_at",
    "script",
    "script_sha256",
    "code",
    "launch_cmd",
    "artifacts",
    "metrics",
)

AUTHOR_KEYS = RUN_AUTHOR_KEYS
LOGGER_KEYS = RUN_LOGGER_KEYS

_OMIT_WHEN_EMPTY = frozenset({
    "parent",
    "motivated_by",
    "finished_at",
    "config_path",
    "config",
    "compute",
    "artifacts",
    "script",
    "script_sha256",
    "launch_cmd",
    "subjects",
})


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


def _script_info(script: str | Path | None, root: Path | None = None) -> tuple[str | None, str | None]:
    """Resolve script path (preferring relative to `root`) and compute its 16-hex SHA-256."""
    raw = str(script) if script is not None else (sys.argv[0] if sys.argv else "")
    raw = raw.strip()
    if not raw or raw in ("-c", "-m", "<stdin>"):
        return None, None
    p = Path(raw)
    resolved = p if p.is_absolute() else p.resolve()
    sha: str | None = None
    if resolved.is_file():
        sha = hashlib.sha256(resolved.read_bytes()).hexdigest()[:16]
    display = raw
    if root is not None:
        try:
            display = str(resolved.relative_to(root.resolve()))
        except ValueError:
            try:
                display = str(resolved.relative_to(root.resolve().parent))
            except ValueError:
                display = raw
    return display, sha


def _id_for(prefix: str, created_at: str) -> str:
    date_part = created_at[:10].replace("-", "")
    return f"{prefix}_{date_part}_{uuid.uuid4().hex[:8]}"


def _default_run_body(parent_ref: str | None, notes: str | None) -> str:
    ref = parent_ref or "baseline"
    obs = f"- {notes}\n" if notes else "- \n"
    return (
        "## Key Takeaway (TL;DR)\n"
        f"- TODO: Summarize key outcome vs {ref}.\n\n"
        "## Observations & Anomalies\n"
        f"{obs}"
    )


def _default_analysis_body(subjects: list[str], notes: str | None) -> str:
    subj_str = ", ".join(subjects) if subjects else "target dataset/checkpoints"
    obs = f"- {notes}\n" if notes else "- \n"
    return (
        "## Key Takeaway (TL;DR)\n"
        f"- TODO: Summarize key finding on {subj_str}.\n\n"
        "## Findings & Observations\n"
        f"{obs}"
    )


def _compact_frontmatter(fm: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in fm.items():
        if k in ("id", "type", "vs_parent"):
            continue
        if k == "bugged" and not v:
            continue
        if k in _OMIT_WHEN_EMPTY and (v is None or v == [] or v == {}):
            continue
        out[k] = v
    return out


def _write_sidecar_config(root: Path, subdir: str, node_id: str, cfg: dict[str, Any]) -> str:
    rel = f"{subdir}/{node_id}.yaml"
    cfg_path = root / rel
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    return rel


def ingest_record(root: Path, record: dict[str, Any], notes: str | None = None) -> Path:
    """Write or update `runs/<id>.md` (and sidecar `runs/configs/<id>.yaml`) from a run record dict."""
    runs_dir = root / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(record["id"])
    path = runs_dir / f"{run_id}.md"

    existing_fm: dict[str, Any] = {}
    body = ""
    if path.exists():
        existing_fm, body = parse_markdown_file(path)

    rec = dict(record)
    raw_cfg = rec.get("config")
    if isinstance(raw_cfg, dict) and raw_cfg:
        rec["config_path"] = _write_sidecar_config(root, RUN_CONFIGS_SUBDIR, run_id, raw_cfg)
        rec["config"] = None

    merged: dict[str, Any] = {}
    for k in RUN_AUTHOR_KEYS:
        if path.exists() and k in existing_fm:
            merged[k] = existing_fm[k]
        else:
            merged[k] = rec.get(k)

    for k in RUN_LOGGER_KEYS:
        merged[k] = rec.get(k)

    known = set(RUN_AUTHOR_KEYS) | set(RUN_LOGGER_KEYS) | {"id", "type", "vs_parent"}
    for k, v in existing_fm.items():
        if k not in known:
            merged[k] = v

    merged = _compact_frontmatter(merged)
    if not body.strip():
        body = _default_run_body(merged.get("parent"), notes)

    write_markdown_file(path, merged, body)
    return path


def ingest_analysis_record(root: Path, record: dict[str, Any], notes: str | None = None) -> Path:
    """Write or update `analyses/<id>.md` from an analysis record dict."""
    ana_dir = root / ANALYSES_DIR
    ana_dir.mkdir(parents=True, exist_ok=True)
    ana_id = str(record["id"])
    path = ana_dir / f"{ana_id}.md"

    existing_fm: dict[str, Any] = {}
    body = ""
    if path.exists():
        existing_fm, body = parse_markdown_file(path)

    merged: dict[str, Any] = {}
    for k in ANALYSIS_AUTHOR_KEYS:
        if path.exists() and k in existing_fm:
            merged[k] = existing_fm[k]
        else:
            merged[k] = record.get(k)

    for k in ANALYSIS_LOGGER_KEYS:
        merged[k] = record.get(k)

    known = set(ANALYSIS_AUTHOR_KEYS) | set(ANALYSIS_LOGGER_KEYS) | {"id", "type"}
    for k, v in existing_fm.items():
        if k not in known:
            merged[k] = v

    merged = _compact_frontmatter(merged)
    if not body.strip():
        body = _default_analysis_body(merged.get("subjects") or [], notes)

    write_markdown_file(path, merged, body)
    return path


class Run:
    """Local-first Contour training run logger writing `runs/run_<YYYYMMDD>_<8hex>.md`."""

    def __init__(
        self,
        title: str | None = None,
        *,
        exp_name: str | None = None,
        parent: str | None = None,
        motivated_by: list[str] | None = None,
        config: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | dict[str, Any] | list[str] | None = None,
        launch_cmd: str | None = None,
        num_gpus: int | None = None,
        rank_zero_only: bool = True,
        root: str | Path | None = None,
        git_cwd: str | Path | None = None,
        notes: str | None = None,
    ) -> None:
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
        self.id = _id_for("run", self.created_at)
        self.title = resolved_title
        self.bugged = False
        self.status = "running"
        self.parent = as_wikilink(parent) if parent else None
        self.motivated_by = [as_wikilink(m) for m in _normalize_str_list(motivated_by)]
        self.config: dict[str, Any] = dict(config or {})
        self.artifacts: list[dict[str, Any]] = normalize_artifacts(artifacts)
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
        if not self.enabled:
            return
        val = float(value)
        if not math.isfinite(val):
            raise ValueError(f"Metric {name!r} must be finite, got {value!r}")
        key = f"{context['subset']}/{name}" if context and "subset" in context else name
        self.metrics[key] = round(val, 6)

    def add_artifact(
        self,
        path: str | Path,
        *,
        kind: str | None = None,
        note: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        entry: dict[str, Any] = {"path": str(path)}
        if kind:
            entry["kind"] = kind
        if note:
            entry["note"] = note
        self.artifacts.append(entry)

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
        return {
            "id": self.id,
            "title": self.title,
            "bugged": self.bugged,
            "parent": self.parent,
            "motivated_by": list(self.motivated_by),
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "config": dict(self.config),
            "compute": {k: v for k, v in self.compute.items() if v is not None},
            "code": dict(self.code),
            "launch_cmd": self.launch_cmd,
            "artifacts": list(self.artifacts),
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


class Analysis:
    """Local-first Contour analysis/eval logger writing `analyses/ana_<YYYYMMDD>_<8hex>.md`."""

    def __init__(
        self,
        title: str,
        *,
        kind: str = "analysis",
        subjects: list[str] | str | None = None,
        motivated_by: list[str] | None = None,
        artifacts: list[dict[str, Any]] | dict[str, Any] | list[str] | None = None,
        script: str | Path | None = None,
        launch_cmd: str | None = None,
        rank_zero_only: bool = True,
        root: str | Path | None = None,
        git_cwd: str | Path | None = None,
        notes: str | None = None,
    ) -> None:
        if kind not in ANALYSIS_KINDS:
            raise ValueError(f"kind must be one of {ANALYSIS_KINDS}, got {kind!r}")
        if not title:
            raise ValueError("Analysis requires `title`")

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
        self.id = _id_for("ana", self.created_at)
        self.kind = kind
        self.title = title
        self.bugged = False
        self.status = "running"
        self.subjects = [as_wikilink(s) for s in _normalize_str_list(subjects)]
        self.motivated_by = [as_wikilink(m) for m in _normalize_str_list(motivated_by)]
        self.artifacts: list[dict[str, Any]] = normalize_artifacts(artifacts)
        self.metrics: dict[str, float] = {}
        self.notes = notes

        git_dir = Path(git_cwd) if git_cwd else self.root
        self.code = _get_git_info(git_dir) if self.enabled else {"commit": None, "dirty": False}
        self.script, self.script_sha256 = _script_info(script, self.root) if self.enabled else (None, None)
        self.launch_cmd = launch_cmd or os.environ.get("LAUNCH_CMD") or " ".join(sys.argv)
        self._closed = False
        self.path: Path | None = None

        if self.enabled and self.root is not None:
            self.path = ingest_analysis_record(self.root, self.to_record(), notes=self.notes)

    def track(
        self,
        value: float,
        name: str,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        val = float(value)
        if not math.isfinite(val):
            raise ValueError(f"Metric {name!r} must be finite, got {value!r}")
        key = f"{context['subset']}/{name}" if context and "subset" in context else name
        self.metrics[key] = round(val, 6)

    def add_artifact(
        self,
        path: str | Path,
        *,
        kind: str | None = None,
        note: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        entry: dict[str, Any] = {"path": str(path)}
        if kind:
            entry["kind"] = kind
        if note:
            entry["note"] = note
        self.artifacts.append(entry)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "bugged": self.bugged,
            "subjects": list(self.subjects),
            "motivated_by": list(self.motivated_by),
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "script": self.script,
            "script_sha256": self.script_sha256,
            "code": dict(self.code),
            "launch_cmd": self.launch_cmd,
            "artifacts": list(self.artifacts),
            "metrics": dict(self.metrics),
        }

    def close(self, failed: bool = False) -> Path | None:
        if self._closed or not self.enabled:
            return self.path
        self._closed = True
        self.finished_at = _utc_now_iso()
        if self.status == "running":
            self.status = "failed" if failed else "final"
        if self.root is not None:
            self.path = ingest_analysis_record(self.root, self.to_record(), notes=self.notes)
        return self.path

    def __enter__(self) -> Analysis:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close(failed=exc_type is not None)


def Eval(title: str, **kwargs: Any) -> Analysis:
    """Convenience constructor for a standardized post-hoc `Analysis(kind='eval')`."""
    kwargs.setdefault("kind", "eval")
    return Analysis(title, **kwargs)


def init_run(**kwargs: Any) -> Run:
    """Convenience constructor for `Run`."""
    return Run(**kwargs)
