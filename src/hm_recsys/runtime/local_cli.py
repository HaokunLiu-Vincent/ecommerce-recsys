"""Run bounded local smoke through the same canonical stages as Colab full.

Default behavior validates inputs without training. Explicit --run uses the
existing smoke budget; --synthetic generates only the documented offline fixture.
No option authorizes real full training or final Test on the host.
"""

import argparse
import json
from pathlib import Path

from hm_recsys.core.contracts import ROOT, Config
from hm_recsys.runtime.acceptance import preflight
from hm_recsys.runtime.files import stage_inputs
from hm_recsys.runtime.modular import ORDER, resolved_profile, run_all
from hm_recsys.runtime.testing import create_source


def main() -> None:
    """Keep local preflight and explicit end-to-end smoke bounded and reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--run-id", default="single_source_local_smoke")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--until", choices=ORDER, default="reload")
    args = parser.parse_args()
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("Run ID must be a simple name")
    run = args.data_root / "runs/smoke" / args.run_id
    source = args.data_root
    if args.synthetic:
        source = run / "synthetic_source"
        if not source.exists():
            create_source(source)
    profile = resolved_profile(ROOT, "smoke")
    if not args.run:
        preflight(run, source, Config(profile["compute"]))
        result = {"status": "passed", "scope": "input_preflight", "run": str(run)}
    else:
        staged = run / "staged_inputs"
        stage_inputs(source, staged)
        result = run_all(
            ROOT,
            staged,
            run / "local",
            run / "persistent",
            profile=profile,
            until=args.until,
            synthetic_fixture=args.synthetic,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
