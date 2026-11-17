"""Compatibility shim: delegates to `contour.cli`."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from contour.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
