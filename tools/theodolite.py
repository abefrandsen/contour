"""Theodolite: Layer 1 Aim Logging Instrument for the Contour Research Framework.

Provides the `Theodolite` context manager (and `init_run` helper) that records
standardized survey coordinates (`run["meta"]`) alongside freeform model configs
(`run["cfg"]`) into Aim (or portable JSON field bundles on isolated cluster nodes).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


def _normalize_str_list(val: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    return [str(x) for x in val]


def _get_git_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "repo": None,
        "branch": None,
        "commit": None,
        "dirty_tree": False,
        "dirty_diff": None,
    }
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        info["commit"] = commit

        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        info["branch"] = branch

        remote = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        info["repo"] = remote

        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        if diff.strip():
            info["dirty_tree"] = True
            info["dirty_diff"] = diff
    except Exception:
        pass
    return info


def _detect_gpu_count() -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        devices = [d.strip() for d in cuda_visible.split(",") if d.strip()]
        return len(devices)
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        if out:
            return len(out.splitlines())
    except Exception:
        pass
    return 1


class Theodolite:
    """Layer 1 survey instrument wrapping `aim.Run` with standardized `meta` facets.

    Can be used directly or as a `with Theodolite(...) as run:` context manager
    that automatically records elapsed GPU hours upon exit.
    """

    def __init__(
        self,
        exp_name: str,
        task: str,
        dataset_train: str | list[str],
        dataset_eval: str | list[str],
        arch_family: str,
        arch_scale: str,
        intervention: str = "baseline",
        parent_runs: list[str] | None = None,
        motivated_by: list[str] | None = None,
        raw_config: dict[str, Any] | None = None,
        launch_cmd: str | None = None,
        repo_path: str | None = None,
        num_gpus: int | None = None,
    ) -> None:
        self.exp_name = exp_name
        self.repo_path = repo_path or os.environ.get("AIM_REPO", ".")
        self.start_time = time.time()
        self.num_gpus = num_gpus if num_gpus is not None else _detect_gpu_count()
        self._aim_run = None

        git_info = _get_git_info()
        resolved_cmd = (
            launch_cmd
            or os.environ.get("LAUNCH_CMD")
            or " ".join(sys.argv)
        )

        self.meta: dict[str, Any] = {
            "title": exp_name,
            "created_at": dt.date.today().isoformat(),
            "task": task,
            "dataset": {
                "train": _normalize_str_list(dataset_train),
                "eval": _normalize_str_list(dataset_eval),
            },
            "arch": {
                "family": arch_family,
                "scale": arch_scale,
            },
            "compute": {
                "num_gpus": self.num_gpus,
                "gpu_hours": None,
                "flops_total": None,
                "tokens_or_samples": None,
            },
            "execution": {
                "launch_cmd": resolved_cmd,
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cluster_node": (
                    os.environ.get("SLURM_NODELIST")
                    or os.environ.get("SKY_NODE_NAME")
                ),
            },
            "git": {
                "repo": git_info["repo"],
                "branch": git_info["branch"],
                "commit": git_info["commit"],
                "dirty_tree": git_info["dirty_tree"],
            },
            "lineage": {
                "intervention": intervention,
                "parent_runs": _normalize_str_list(parent_runs),
                "motivated_by": _normalize_str_list(motivated_by),
            },
            "validity": "verified",
            "checkpoints": {},
        }
        self.raw_config = raw_config or {}
        self.summary_metrics: dict[str, float] = {}

        # Initialize underlying Aim run if `aim` is installed
        try:
            import aim  # type: ignore

            self._aim_run = aim.Run(
                repo=self.repo_path,
                experiment=exp_name,
                log_system_params=True,
            )
            self.hash: str = self._aim_run.hash
            self._aim_run["meta"] = self.meta
            if self.raw_config:
                self._aim_run["cfg"] = self.raw_config
            if git_info["dirty_diff"]:
                self._aim_run["repro_dirty_diff"] = git_info["dirty_diff"]
        except ImportError:
            self.hash = f"local_{int(self.start_time):x}"

    def track(
        self,
        value: float,
        name: str,
        step: int | None = None,
        epoch: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Track a scalar metric series (forwards to `aim.Run.track` and updates summary)."""
        if context and "subset" in context:
            key = f"{context['subset']}/{name}"
        else:
            key = name
        self.summary_metrics[key] = float(value)

        if self._aim_run is not None:
            self._aim_run.track(
                value, name=name, step=step, epoch=epoch, context=context
            )

    def log_checkpoint(self, tag: str, uri: str) -> None:
        """Record a checkpoint path or URI (e.g., tag='best', uri='/data/ckpt/best.pt')."""
        self.meta["checkpoints"][tag] = uri
        self._sync_meta()

    def set_compute_budget(
        self,
        gpu_hours: float | None = None,
        flops_total: float | None = None,
        tokens_or_samples: int | None = None,
    ) -> None:
        """Explicitly set compute budget counters."""
        if gpu_hours is not None:
            self.meta["compute"]["gpu_hours"] = round(float(gpu_hours), 3)
        if flops_total is not None:
            self.meta["compute"]["flops_total"] = float(flops_total)
        if tokens_or_samples is not None:
            self.meta["compute"]["tokens_or_samples"] = int(tokens_or_samples)
        self._sync_meta()

    def mark_validity(self, validity: str) -> None:
        """Set run validity ('verified', 'bugged', 'aborted', 'noisy', 'superseded')."""
        allowed = {"verified", "bugged", "aborted", "noisy", "superseded"}
        if validity not in allowed:
            raise ValueError(f"validity must be one of {allowed}, got {validity!r}")
        self.meta["validity"] = validity
        self._sync_meta()

    def export_json_bundle(self, out_dir: str | Path) -> Path:
        """Export a portable JSON field bundle (useful on isolated cluster nodes)."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        bundle_file = out_path / f"fieldlog_{self.hash[:8]}.json"
        payload = {
            "aim_hash": self.hash,
            "meta": self.meta,
            "cfg": self.raw_config,
            "metrics": self.summary_metrics,
        }
        bundle_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return bundle_file

    def close(self) -> None:
        """Finalize compute hours and close the underlying Aim run."""
        if self.meta["compute"]["gpu_hours"] is None:
            elapsed_hours = (time.time() - self.start_time) / 3600.0
            self.meta["compute"]["gpu_hours"] = round(
                elapsed_hours * self.num_gpus, 3
            )
        self._sync_meta()
        if self._aim_run is not None:
            self._aim_run.close()

    def _sync_meta(self) -> None:
        if self._aim_run is not None:
            self._aim_run["meta"] = self.meta

    def __enter__(self) -> Theodolite:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.meta["validity"] = "aborted"
        self.close()


def init_run(**kwargs: Any) -> Theodolite:
    """Convenience constructor for `Theodolite`."""
    return Theodolite(**kwargs)
