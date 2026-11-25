"""Ensure the src layout is importable even if pytest's pythonpath
ini option is not picked up (for example, when run from another cwd)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
