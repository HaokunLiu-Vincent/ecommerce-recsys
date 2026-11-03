"""Run a capped reference/optimized compute diagnostic, never a full training job."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hm_recsys.runtime.benchmark import main  # noqa: E402

if __name__ == "__main__":
    main()
