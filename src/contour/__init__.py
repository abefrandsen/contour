"""Contour: Epistemic ML research state, verification, reporting, and ideation."""

from .check import (
    Issue,
    compute_frontier,
    compute_verified_hash,
    effective_verification,
    evidence_control_level,
    find_check_issues,
    find_review_items,
    fix_stale_vs_parent,
)
from .compare import (
    compare_runs,
    compute_analysis_control_level,
    compute_vs_parent,
    get_fresh_vs_parent,
)
from .config import ConfigError, ContourConfig, find_root, load_config
from .graph import ContourGraph, ContourNode
from .ideate import generate_ideation
from .report import build_brief, build_report, export_report, render_report_markdown
from .run_logger import (
    Analysis,
    Eval,
    Run,
    ingest_analysis_record,
    ingest_record,
    init_run,
)

__version__ = "1.1.0"

__all__ = [
    "Analysis",
    "ConfigError",
    "ContourConfig",
    "ContourGraph",
    "ContourNode",
    "Eval",
    "Issue",
    "Run",
    "build_brief",
    "build_report",
    "compare_runs",
    "compute_analysis_control_level",
    "compute_frontier",
    "compute_verified_hash",
    "compute_vs_parent",
    "effective_verification",
    "evidence_control_level",
    "export_report",
    "find_check_issues",
    "find_review_items",
    "find_root",
    "fix_stale_vs_parent",
    "generate_ideation",
    "get_fresh_vs_parent",
    "ingest_analysis_record",
    "ingest_record",
    "init_run",
    "load_config",
    "render_report_markdown",
]
