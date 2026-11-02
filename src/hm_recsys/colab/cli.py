"""Notebook-facing stage, status, recovery, and input-staging commands."""

import argparse
import json
import os
import re
from pathlib import Path

from hm_recsys.colab.files import atomic_json, stage_inputs
from hm_recsys.colab.profiles import is_colab_runtime, load_profile, plan, stages_for
from hm_recsys.colab.runner import execute, execution_identity
from hm_recsys.colab.state import STAGES, StageStore
from hm_recsys.contracts import ROOT


def main() -> None:
    """Default to a plan; require explicit opt-in for Colab pilot fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "plan",
            "preflight",
            "run",
            "resume",
            "status",
            "review",
            "stage-inputs",
        ),
        nargs="?",
        default="plan",
    )
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "full"), default="smoke"
    )
    parser.add_argument("--run-id", default="colab_smoke_v1")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--persistent-root", type=Path)
    parser.add_argument("--mount-root", type=Path)
    from hm_recsys.pilot.worker import STAGES as PILOT_STAGES

    parser.add_argument(
        "--until", choices=sorted(set(STAGES + PILOT_STAGES)), default="reload"
    )
    parser.add_argument("--allow-full", action="store_true")
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--reverify", action="store_true")
    parser.add_argument("--recover-lock", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("Run ID must be a simple name")
    if args.final_test:
        parser.error("Final Test is disabled")
    if args.action == "stage-inputs":
        if args.source_root is None:
            parser.error("stage-inputs requires --source-root")
        result = stage_inputs(args.source_root, args.data_root, reverify=args.reverify)
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "input_digest": result["input_digest"],
                },
                indent=2,
            )
        )
        return
    if args.persistent_root is None:
        parser.error(
            "Provide --persistent-root explicitly; training never defaults to Drive"
        )
    if is_colab_runtime() and args.action != "plan":
        if args.mount_root is None or not os.path.ismount(args.mount_root):
            parser.error("Colab persistence requires a verified --mount-root")
        if not args.persistent_root.resolve().is_relative_to(args.mount_root.resolve()):
            parser.error("Persistence must be beneath the supplied mounted root")
        if args.data_root.resolve().is_relative_to(args.mount_root.resolve()):
            parser.error("Training inputs must be local, not inside the Drive mount")
    local = args.data_root / "runs/colab" / args.run_id
    persistent = args.persistent_root / args.run_id
    if args.action == "plan":
        result = plan(
            ROOT, args.profile, args.data_root, local, persistent, device=args.device
        )
    elif args.action in ("status", "review"):
        if not (persistent / "identity.json").is_file():
            parser.error("No persistent run identity to inspect")
        profile = load_profile(ROOT, args.profile)
        if args.device is not None:
            if args.profile != "pilot":
                parser.error("Device override is pilot-only")
            profile["device"] = args.device
        identity = execution_identity(ROOT, profile, args.data_root)
        store = StageStore(
            local,
            persistent,
            identity,
            mount_root=args.mount_root,
            stages=stages_for(profile),
        )
        result = store.status()
        if args.action == "review":
            if args.profile != "pilot":
                parser.error("Review bundles are pilot-only")
            from hm_recsys.colab.review import publish_review

            records = [
                record
                for stage in stages_for(profile)
                if (record := store.record(stage)) is not None
            ]
            result = publish_review(
                local, persistent, identity, result, store.view(records)
            )

    else:
        if args.action == "resume" and not (persistent / "identity.json").exists():
            parser.error("No persistent run identity to resume")
        result = execute(
            ROOT,
            args.data_root,
            local,
            persistent,
            profile_name=args.profile,
            until="preflight" if args.action == "preflight" else args.until,
            allow_full=args.allow_full,
            allow_pilot=args.allow_pilot,
            device=args.device,
            final_test=args.final_test,
            recover_lock=args.recover_lock,
            mount_root=args.mount_root,
        )
    # A plan is evidence, not authorization or a fabricated completion status.
    if args.action == "plan":
        print(json.dumps(result, indent=2))
    else:
        atomic_json(local / "last_command_result.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
