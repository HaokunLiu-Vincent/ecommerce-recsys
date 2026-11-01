"""Opt-in local smoke workflow with immutable run directories and CPU guards."""

import argparse
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from hm_recsys.contracts import ROOT, Config, digest, read_json, write_json
from hm_recsys.resources import run_stage


def execute(mode: str, run: Path, config: Config, source: Path) -> None:
    """Run guarded stages sequentially; record blocked status before propagating."""
    manifest = {
        "is_smoke": True,
        "is_benchmark": False,
        "architecture": "v0.3.2",
        "protocol": "v2.0",
        "mode": mode,
        "source_root": str(source),
        "seed": 42,
        "device": "cpu",
        "status": "running",
        "git_revision": "unavailable_no_git_repository",
        "code_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for folder in (ROOT / "src/hm_recsys", ROOT / "scripts", ROOT / "configs")
            for p in folder.glob("*")
            if p.suffix in (".py", ".json")
        },
        "smoke_overrides": config.values,
        "synthetic_contract_tests": "separate_pytest_result",
        "sampled_real_end_to_end": "not_run",
        "fresh_process_reload_parity": "not_run",
        "quality_checks": "separate_commands",
    }
    write_json(run / "run_manifest.json", manifest)
    write_json(run / "config.resolved.json", config.values)
    for folder in ("models", "indexes", "tmp"):
        (run / folder).mkdir()
    with (run / "commands.log").open("a") as handle:
        handle.write(" ".join(sys.argv) + "\n")
    stages = (["fixture"] if mode == "synthetic" else []) + ["preflight"]
    if mode != "preflight":
        stages += [
            "sample",
            "wals",
            "towers",
            "index_towers",
            "encode",
            "preprocessing",
            "features",
            "ranker",
            "reload_encode",
            "reload_features",
            "reload",
        ]
    try:
        for stage in stages:
            print(f"Starting guarded stage: {stage}", flush=True)
            run_stage(stage, run, config)
    except (RuntimeError, ValueError, OSError) as error:
        manifest = read_json(run / "run_manifest.json")
        manifest.update(status="blocked", error=str(error))
        if mode == "real-smoke":
            manifest["sampled_real_end_to_end"] = "blocked"
        write_json(run / "run_manifest.json", manifest)
        raise
    manifest = read_json(run / "run_manifest.json")
    manifest["status"] = "passed"
    if mode != "preflight":
        manifest["fresh_process_reload_parity"] = "passed"
        manifest["sampled_real_end_to_end"] = (
            "passed" if mode == "real-smoke" else "not_run_synthetic_only"
        )
        results = read_json(run / "smoke_results.json")
        results["status"] = "passed"
        results["sample"] = {
            key: read_json(run / "sample_manifest.json")[key]
            for key in ("users", "catalog", "transactions")
        }
        results["wals"] = {
            key: read_json(run / "models/wals.json")[key]
            for key in ("iterations", "factors", "nnz")
        }
        results["twotower"] = {
            key: read_json(run / "models/twotower.json")[key]
            for key in (
                "steps",
                "epochs",
                "selected_targets",
                "processed_targets",
                "projection_max_parameter_change",
                "internal_daily_recall50",
            )
        }
        write_json(run / "smoke_results.json", results)
    write_json(run / "run_manifest.json", manifest)


def main() -> None:
    """Default to no-training preflight; real fitting requires real-smoke mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("preflight", "synthetic", "real-smoke", "reload"),
        nargs="?",
        default="preflight",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        parser.error("Run ID must be a simple name, not a path")
    run = ROOT / "data/runs/smoke" / run_id
    if args.mode == "reload":
        if not args.run_id or not run.is_dir():
            parser.error("reload requires an explicit existing smoke run ID")
        config = Config.load(run / "config.resolved.json")
        replay = run / "reloads" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        replay.mkdir(parents=True, exist_ok=False)
        for folder in ("models", "indexes", "inference_input"):
            shutil.copytree(run / folder, replay / folder)
        for name in (
            "run_manifest.json",
            "config.resolved.json",
            "feature_schema.json",
            "split_manifest.json",
            "queries/Validation.json",
            "features/Validation.parquet",
            "predictions/reference_scores.npy",
            "predictions/reference_top12.json",
        ):
            (replay / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / name, replay / name)
        with (replay / "commands.log").open("w") as handle:
            handle.write(" ".join(sys.argv) + "\n")
        for stage in ("reload_encode", "reload_features", "reload"):
            run_stage(stage, replay, config)
        print(f"Completed isolated reload: {replay}")
        return
    run.mkdir(parents=True, exist_ok=False)
    config = Config.load()
    source = run / "synthetic_inputs" if args.mode == "synthetic" else ROOT / "data"
    try:
        execute(args.mode, run, config, source)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Blocked: {error}; artifacts: {run}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Completed {args.mode}: {run}")


if __name__ == "__main__":
    main()
