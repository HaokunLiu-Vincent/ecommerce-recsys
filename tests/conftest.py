"""Bound numerical libraries before test modules import their dependencies."""

import os
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[name] = "2"
os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
