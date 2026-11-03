"""Run one canonical numerical phase in an independent Python process.

The resource supervisor supplies a private work view and resolved profile. This
entry point validates runtime source bindings, dispatches shared algorithms, and
exits nonzero on failure. It never chooses another profile or falls back from
requested CUDA. Test phases require an explicit authorized options record.
"""

import sys
from pathlib import Path

from hm_recsys.core.contracts import read_json
from hm_recsys.runtime.stages import run_stage


def main() -> None:
    """Execute the named phase without interpreting unchecked shell text."""
    phase, location = sys.argv[1:]
    run = Path(location)
    options = read_json(run / "colab_stage.json")
    profile = read_json(run / "profile.resolved.json")
    if options.get("synthetic_fixture"):
        from hm_recsys.runtime.testing import assert_fixture

        assert_fixture(Path(options["data_root"]), full=profile["profile"] == "full")
    elif profile["profile"] != "smoke":
        from hm_recsys.runtime.profiles import is_colab_runtime

        if not is_colab_runtime():
            raise PermissionError("Full training is prohibited on the development host")
    for prefix in ("reload_", "test_"):
        logical = phase.removeprefix(prefix)
        if phase.startswith(prefix) and logical.startswith(
            ("channel_", "fusion_", "features_")
        ):
            if prefix == "test_" and not options.get("final_test_active"):
                raise PermissionError("Test worker requires explicit authorization")
            phase = logical
            break
    run_stage(phase, run, profile, options)
    print(f"Canonical phase {phase}: passed", flush=True)


if __name__ == "__main__":
    main()
