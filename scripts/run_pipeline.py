"""Call shared Colab plan, staging, stage, status, and recovery interfaces."""

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT / "src")

from hm_recsys.runtime.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
