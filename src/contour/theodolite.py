"""Theodolite: Reference training-run adapter emitting Contour Field Bundles (v1).

Records standardized survey coordinates alongside freeform model configs into:
  1. A portable Field Bundle JSON (`contour.field_bundle/v1`) — always written on
     close (and optionally at init/checkpoints), so runs without Aim still persist.
  2. Optionally an Aim run (`aim.Run`) if `aim` is installed and `use_aim=True`.

Multi-GPU / DDP: Only rank 0 logs by default (`rank_zero_only=True`).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
import uuid

from .bundle import DEFAULT_TIER, SCHEMA_ID, validate_bundle


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
    """Collect git commit, branch, remote, dirty status (including untracked files)."""
    info: dict[str, Any] = {
        "repo": None,
        "branch": None,
        "commit": None,
        "dirty": False,
        "dirty_hash": None,
        "dirty_patch": None,
        "untracked": [],
    }
    commit = _git_output(["rev-parse", "--short", "HEAD"], cwd=cwd)
    if not commit:
        return info
    info["commit"] = commit
    info["branch"] = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    info["repo"] = _git_output(["config", "--get", "remote.origin.url"], cwd=cwd)

    diff = _git_output(["diff", "HEAD"], cwd=cwd) or ""
    porcelain = _git_output(["status", "--porcelain"], cwd=cwd) or ""
    untracked = [
        line[3:].strip()
        for line in porcelain.splitlines()
        if line.startswith("?? ")
    ]
    info["untracked"] = untracked

    if diff or porcelain:
        info["dirty"] = True
        if diff:
            info["dirty_patch"] = diff
        payload = (diff + "\n" + porcelain).encode("utf-8")
        info["dirty_hash"] = hashlib.sha256(payload).hexdigest()[:16]

    return info


class Run:
    """Reference Layer-1 run logger producing Field Bundle v1 records."""

    def __init__(
        self,
        exp_name: str,
        task: str,
        dataset_train: str | list[str],
        dataset_eval: str | list[str],
        arch_family: str,
        arch_scale: str,
        interventions: list[str] | str | None = None,
        intervention_note: str | None = None,
        parent_runs: list[str] | None = None,
        motivated_by: list[str] | None = None,
        tier: str = DEFAULT_TIER,
        seed: int | None = None,
        raw_config: dict[str, Any] | None = None,
        launch_cmd: str | None = None,
        bundle_dir: str | Path | None = "field_bundles",
        repo_path: str | None = None,
        num_gpus: int | None = None,
        use_aim: bool = True,
        rank_zero_only: bool = True,
        git_cwd: str | Path | None = None,
    ) -> None:
        self.exp_name = exp_name
        self.rank = _detect_rank()
        self.enabled = (self.rank == 0) or (not rank_zero_only)
        self.repo_path = repo_path or os.environ.get("AIM_REPO", ".")
        self.bundle_dir = Path(bundle_dir) if bundle_dir is not None else None
        self.start_time = time.monotonic()
        self.created_at = _utc_now_iso()
        self.finished_at: str | None = None
        self.num_gpus = num_gpus if num_gpus is not None else _detect_gpu_count()
        self.run_id = uuid.uuid4().hex
        self.hash = self.run_id
        self._aim_run = None
        self._closed = False

        self.status = "running"
        self.raw_config = dict(raw_config or {})
        self.summary_metrics: dict[str, float] = {}
        self.telemetry: dict[str, float] = {}
        self.artifacts: dict[str, str] = {}
        self.notes: str | None = None

        git_info = _get_git_info(Path(git_cwd) if git_cwd else None) if self.enabled else {}
        resolved_cmd = launch_cmd or os.environ.get("LAUNCH_CMD") or " ".join(sys.argv)

        self.code: dict[str, Any] = {
            k: v
            for k, v in {
                "repo": git_info.get("repo"),
                "branch": git_info.get("branch"),
                "commit": git_info.get("commit"),
                "dirty": git_info.get("dirty", False),
                "dirty_hash": git_info.get("dirty_hash"),
                "dirty_patch": git_info.get("dirty_patch"),
                "untracked": git_info.get("untracked") or [],
            }.items()
            if v is not None and v != []
        }
        if "dirty" not in self.code and self.enabled:
            self.code["dirty"] = False

        self.launch: dict[str, Any] = {
            k: v
            for k, v in {
                "cmd": resolved_cmd,
                "host": socket.gethostname(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURM_NODELIST") or os.environ.get("SKY_NODE_NAME"),
            }.items()
            if v is not None
        }

        self.data: dict[str, list[str]] = {
            "train": _normalize_str_list(dataset_train),
            "eval": _normalize_str_list(dataset_eval),
        }

        self.facets: dict[str, Any] = {
            "task": task,
            "arch_family": arch_family,
            "arch_scale": arch_scale,
            "tier": tier,
            "seed": seed,
        }

        self.interventions = _normalize_str_list(interventions)
        self.intervention_note = intervention_note

        self.lineage: dict[str, Any] = {
            "parents": _normalize_str_list(parent_runs),
            "motivated_by": _normalize_str_list(motivated_by),
        }

        self.compute: dict[str, Any] = {
            "num_gpus": self.num_gpus,
            "gpu_hours": None,
            "wall_seconds": None,
            "flops": None,
            "params": None,
            "tokens_or_samples": None,
        }

        if self.enabled and use_aim:
            try:
                import aim  # type: ignore

                self._aim_run = aim.Run(
                    repo=self.repo_path,
                    experiment=exp_name,
                    log_system_params=True,
                )
                self.run_id = self._aim_run.hash
                self.hash = self.run_id
                self._sync_aim()
            except ImportError:
                self._aim_run = None

        if self.enabled and self.bundle_dir is not None:
            self.export_json_bundle(self.bundle_dir, validate=False)

    def track(
        self,
        value: float,
        name: str,
        step: int | None = None,
        epoch: int | None = None,
        context: dict[str, Any] | None = None,
        kind: str = "metric",
    ) -> None:
        """Track a scalar series (`kind='metric'` or `'telemetry'`)."""
        if not self.enabled:
            return
        key = f"{context['subset']}/{name}" if context and "subset" in context else name
        target = self.telemetry if kind == "telemetry" else self.summary_metrics
        target[key] = float(value)

        if self._aim_run is not None:
            self._aim_run.track(value, name=name, step=step, epoch=epoch, context=context)

    def log_telemetry(self, key: str, value: float, step: int | None = None) -> None:
        """Track an internal telemetry scalar (e.g. `attn/entropy_mid`)."""
        self.track(value, name=key, step=step, kind="telemetry")

    def log_checkpoint(self, tag: str, uri: str) -> None:
        if not self.enabled:
            return
        self.artifacts[tag] = uri
        self._sync_aim()

    def set_compute_budget(
        self,
        gpu_hours: float | None = None,
        flops: float | None = None,
        params: int | None = None,
        tokens_or_samples: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        if gpu_hours is not None:
            self.compute["gpu_hours"] = round(float(gpu_hours), 3)
        if flops is not None:
            self.compute["flops"] = float(flops)
        if params is not None:
            self.compute["params"] = int(params)
        if tokens_or_samples is not None:
            self.compute["tokens_or_samples"] = int(tokens_or_samples)
        self._sync_aim()

    def to_bundle(self, validate: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SCHEMA_ID,
            "run_id": self.run_id,
            "title": self.exp_name,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "code": self.code,
            "launch": self.launch,
            "config": self.raw_config,
            "data": self.data,
            "facets": self.facets,
            "interventions": self.interventions,
            "intervention_note": self.intervention_note,
            "lineage": self.lineage,
            "compute": {k: v for k, v in self.compute.items() if v is not None},
            "metrics": self.summary_metrics,
            "telemetry": self.telemetry,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }
        if validate:
            return validate_bundle(payload, source=f"theodolite:{self.run_id}")
        return payload

    def export_json_bundle(self, out_dir: str | Path, validate: bool = True) -> Path | None:
        """Write a portable Field Bundle v1 JSON file."""
        if not self.enabled:
            return None
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        bundle_file = out_path / f"bundle_{self.run_id}.json"
        payload = self.to_bundle(validate=validate)
        bundle_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return bundle_file

    def close(self, failed: bool = False) -> None:
        if self._closed or not self.enabled:
            return
        self._closed = True
        wall = round(time.monotonic() - self.start_time, 3)
        self.compute["wall_seconds"] = wall
        if self.compute["gpu_hours"] is None:
            self.compute["gpu_hours"] = round((wall / 3600.0) * self.num_gpus, 3)
        self.finished_at = _utc_now_iso()
        if self.status == "running":
            self.status = "failed" if failed else "final"
        self._sync_aim()
        if self._aim_run is not None:
            self._aim_run.close()
        if self.bundle_dir is not None:
            # Validate on terminal close only if final with metrics, else write unvalidated on failed
            self.export_json_bundle(
                self.bundle_dir, validate=(self.status == "final" and bool(self.summary_metrics))
            )

    def _sync_aim(self) -> None:
        if self._aim_run is not None:
            self._aim_run["contour_bundle"] = self.to_bundle(validate=False)
            if self.raw_config:
                self._aim_run["cfg"] = self.raw_config

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close(failed=exc_type is not None)


# Backward-compatible alias
Theodolite = Run


def init_run(**kwargs: Any) -> Run:
    """Convenience constructor for `Run`."""
    return Run(**kwargs)
