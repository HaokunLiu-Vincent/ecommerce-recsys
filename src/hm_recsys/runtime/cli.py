"""Expose canonical stages, explicit permissions and immutable run identities.

CLI inputs are paths, a smoke/full profile and optional verified Git provenance.
Outputs are aggregate JSON reports and checksummed stage records. Numerical work
runs in independent workers. Missing dependencies never trigger hidden training;
Final Test requires a separate stage and two explicit approvals.
"""

import argparse
import json
import os
import re
from pathlib import Path

from hm_recsys.core.contracts import ROOT
from hm_recsys.runtime.files import atomic_json, stage_inputs
from hm_recsys.runtime.modular import (
    ALL_STAGES,
    ORDER,
    Session,
    resolved_profile,
    run_all,
    run_one,
    safe_report,
)
from hm_recsys.runtime.profiles import execution_gate, is_colab_runtime, plan


def main() -> None:
    """Validate permissions before dispatching only the requested work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "plan",
            "preflight",
            "stage-inputs",
            "stage",
            "run",
            "resume",
            "status",
            "review",
        ),
        nargs="?",
        default="plan",
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--run-id", default="single_source_smoke_v1")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--persistent-root", type=Path)
    parser.add_argument("--mount-root", type=Path)
    parser.add_argument("--until", choices=ORDER, default="reload")
    parser.add_argument("--stage", choices=ALL_STAGES)
    parser.add_argument(
        "--stage-action", choices=("run", "resume", "status"), default="run"
    )
    parser.add_argument("--allow-full", action="store_true")
    parser.add_argument("--run-final-test", action="store_true")
    parser.add_argument("--approve-final-test", action="store_true")
    parser.add_argument("--locked-signature")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--performance", choices=("reference", "optimized"), default="optimized"
    )
    parser.add_argument("--reverify", action="store_true")
    parser.add_argument("--recover-lock", action="store_true")
    parser.add_argument("--git-remote")
    parser.add_argument("--git-ref")
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("Run ID must be a simple name")
    if args.action == "stage-inputs":
        if args.source_root is None:
            parser.error("stage-inputs requires --source-root")
        result = stage_inputs(args.source_root, args.data_root, reverify=args.reverify)
        print(
            json.dumps(
                {"complete": result["complete"], "input_digest": result["input_digest"]}
            )
        )
        return
    if args.persistent_root is None:
        parser.error("Provide --persistent-root explicitly")
    if args.action == "stage" and args.stage is None:
        parser.error("stage requires --stage")
    if (
        args.run_final_test or args.approve_final_test or args.locked_signature
    ) and not (args.action == "stage" and args.stage == "final_test"):
        parser.error("Final Test options belong only to stage --stage final_test")
    git_source = None
    provenance = (args.git_remote, args.git_ref, args.git_commit)
    if any(provenance):
        if not all(provenance):
            parser.error(
                "Git execution requires --git-remote, --git-ref and --git-commit"
            )
        from hm_recsys.runtime.git_checkout import inspect_checkout

        git_source = inspect_checkout(
            ROOT, repo_url=args.git_remote, ref=args.git_ref, commit=args.git_commit
        )
    if is_colab_runtime() and args.action != "plan":
        if args.mount_root is None or not os.path.ismount(args.mount_root):
            parser.error("Colab persistence requires a verified --mount-root")
        if not args.persistent_root.resolve().is_relative_to(args.mount_root.resolve()):
            parser.error("Persistence must be beneath the mounted root")
        if args.data_root.resolve().is_relative_to(args.mount_root.resolve()):
            parser.error("Training inputs must be local, outside Drive")
    local = args.data_root / "runs/colab" / args.run_id
    persistent = args.persistent_root / args.run_id
    if args.action == "plan":
        result = plan(
            ROOT,
            args.profile,
            args.data_root,
            local,
            persistent,
            device=args.device,
            performance=args.performance,
        )
        if git_source is not None:
            result["source_control"] = git_source
    else:
        profile = resolved_profile(
            ROOT, args.profile, device=args.device, performance=args.performance
        )
        executing = args.action in ("run", "resume", "preflight") or (
            args.action == "stage" and args.stage_action != "status"
        )
        if executing:
            execution_gate(profile, allow_full=args.allow_full, final_test=False)
        if args.action == "resume" and not (persistent / "identity.json").is_file():
            parser.error("No persistent run identity to resume")
        options = dict(
            profile=profile,
            mount_root=args.mount_root,
            recover_lock=args.recover_lock,
            git_source=git_source,
        )
        if args.action == "stage":
            result = run_one(
                ROOT,
                args.data_root,
                local,
                persistent,
                stage=args.stage,
                action=args.stage_action,
                run_final_test=args.run_final_test,
                approve_final_test=args.approve_final_test,
                locked_signature=args.locked_signature,
                **options,
            )
        elif args.action in ("status", "review"):
            with Session(ROOT, args.data_root, local, persistent, **options) as session:
                result = session.store.status()
                if args.action == "review":
                    from hm_recsys.runtime.review import publish_review

                    result = publish_review(
                        local, persistent, session.identity, result, session._view()
                    )
        else:
            result = run_all(
                ROOT,
                args.data_root,
                local,
                persistent,
                until="prepare" if args.action == "preflight" else args.until,
                **options,
            )
        atomic_json(local / "last_command_result.json", safe_report(result))
    print(json.dumps(safe_report(result), indent=2))


if __name__ == "__main__":
    main()
