"""Run an isolated bounded shared-tower device update and reload probe."""

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT / "src")

from hm_recsys.colab.files import atomic_json  # noqa: E402
from hm_recsys.contracts import Config  # noqa: E402
from hm_recsys.resources import run_stage  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Expose the actual CLI contract without importing or running CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Keep probe evidence separate from actual H&M smoke results."""
    parser = build_parser()
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(PROJECT / "data/runs/colab"):
        parser.error("Device probe must remain under data/runs/colab")
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config.load()
    atomic_json(args.output / "config.resolved.json", config.values)
    atomic_json(args.output / "colab_stage.json", {})
    run_stage(
        f"probe_{args.device}",
        args.output,
        config,
        module="hm_recsys.colab.worker",
        cwd=PROJECT,
    )
    print(args.output / "probe/device_probe.json")


if __name__ == "__main__":
    main()
