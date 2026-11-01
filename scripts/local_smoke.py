"""Run the repository-local guarded smoke CLI from any working directory."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ["PYTHONPATH"] = str(REPO / "src")

from hm_recsys.cli import main  # noqa: E402 -- Resolve the uninstalled src tree.

if __name__ == "__main__":
    main()
