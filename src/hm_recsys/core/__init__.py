"""Shared temporal, identifier and progress contracts without training side effects.

This namespace exposes implementation modules without fitting or reading runtime
data at import time. Numerical entry points receive explicit paths and versioned
contracts; caller orchestration owns permissions and artifact publication.
"""
