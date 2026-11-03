"""Canonical H&M next-seven-day recommendation implementation.

The eight source layers share temporal and schema contracts across smoke and
full execution. Importing the package does not read source data, train models,
load archived code or initialize cloud services. CLI adapters live in scripts/.
"""
