"""Bound numerical libraries before test modules import their dependencies."""

# Use filesystem atomic publication, environment and process boundaries.
import os

# Use portable paths resolved beneath project-owned roots.
from pathlib import Path

# Process each artifact, feature or stage identifier used in this operation from the
# declared ('OMP NUM THREADS', 'OPENBLAS NUM THREADS', 'MKL NUM THREADS', 'VECLIB
# MAXIMUM THREADS', 'NUMEXPR NUM THREADS', 'BLIS NUM THREADS') sequence for bound
# numerical libraries before test modules import their dependencies.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    # Initialize environ to the explicit '2' value required by this operation.
    os.environ[name] = "2"
# Resolve environ for bound numerical libraries before test modules import their
# dependencies before the downstream consumer uses it.
os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
