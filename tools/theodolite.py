"""Compatibility shim: re-exports `Run`, `Theodolite`, and `init_run` from `contour.theodolite`."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from contour.theodolite import Run, Theodolite, _get_git_info, init_run  # noqa: E402

__all__ = ["Run", "Theodolite", "init_run", "_get_git_info"]
