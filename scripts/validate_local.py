"""Delegates to norgesgruppen/scripts/validate_local.py."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "norgesgruppen" / "scripts" / "validate_local.py"
    runpy.run_path(str(target), run_name="__main__")
