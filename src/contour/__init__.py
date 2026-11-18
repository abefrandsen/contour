"""Contour: Epistemic ML research state, verification, and ideation."""

from .check import (
    Issue,
    compute_frontier,
    compute_verified_hash,
    effective_verification,
    find_check_issues,
    find_review_items,
    fix_stale_vs_parent,
)
from .compare import compare_runs, compute_vs_parent, get_fresh_vs_parent
from .config import ConfigError, ContourConfig, find_root, load_config
from .graph import ContourGraph, ContourNode
from .ideate import generate_ideation
from .run_logger import Run, ingest_record, init_run

__version__ = "1.0.0"

__all__ = [
    "ConfigError",
    "ContourConfig",
    "ContourGraph",
    "ContourNode",
    "Issue",
    "Run",
    "compare_runs",
    "compute_frontier",
    "compute_verified_hash",
    "compute_vs_parent",
    "effective_verification",
    "find_check_issues",
    "find_review_items",
    "find_root",
    "fix_stale_vs_parent",
    "generate_ideation",
    "get_fresh_vs_parent",
    "ingest_record",
    "init_run",
    "load_config",
]
