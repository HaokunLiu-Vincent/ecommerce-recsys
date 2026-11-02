"""Thin Colab stage adapter around the existing isolated smoke workers."""

import sys
from pathlib import Path

from hm_recsys.contracts import Config, read_json


def main() -> None:
    """Reuse model implementations and expose fold/checkpoint boundaries."""
    stage, location = sys.argv[1:]
    run = Path(location)
    options = read_json(run / "colab_stage.json")
    profile_path = run / "profile.resolved.json"
    if profile_path.exists() and read_json(profile_path)["profile"] == "pilot":
        from hm_recsys.pilot.worker import run_stage

        run_stage(stage, run, read_json(profile_path), options)
        print(f"Pilot stage {stage}: passed", flush=True)
        return
    config = Config.load(run / "config.resolved.json")
    if stage.startswith("probe_"):
        from hm_recsys.colab.device import probe

        probe(run / "probe", stage.removeprefix("probe_"))
    elif stage.startswith("features_"):
        import faiss

        from hm_recsys.pipeline import candidates_and_features

        faiss.omp_set_num_threads(config.limit("threads"))
        partition = stage.removeprefix("features_")
        if partition not in ("F1", "F2", "F3", "F4", "Validation"):
            raise ValueError("Test and unknown feature partitions are disabled")
        candidates_and_features(
            run, Path(options["data_root"]), config, partitions=[partition]
        )
    elif stage == "towers":
        import torch

        from hm_recsys.data import Sample
        from hm_recsys.pipeline import load_content
        from hm_recsys.towers import fit_towers

        torch.set_num_threads(config.limit("threads"))
        sample = Sample.load(run, config)
        content = load_content(run, sample, Path(options["data_root"]))
        checkpoint = Path(options["persistent_checkpoint"])
        resume = run / "resumed_checkpoint"
        if (checkpoint / "latest.json").is_file():
            from hm_recsys.colab.files import atomic_json, verified_copy

            record = read_json(checkpoint / "latest.json")
            resume.mkdir(exist_ok=True)
            verified_copy(
                checkpoint / record["file"],
                resume / record["file"],
                expected_sha256=record["sha256"],
            )
            atomic_json(resume / "latest.json", record)
        fit_towers(
            sample,
            content,
            config,
            run,
            checkpoint_folder=run / "checkpoints",
            persistent_checkpoint=checkpoint,
            persistent_mount_root=(
                Path(options["mount_root"]) if options.get("mount_root") else None
            ),
            resume_folder=resume if resume.exists() else None,
            checkpoint_every_steps=options["checkpoint_every_steps"],
        )
    else:
        from hm_recsys.worker import main as shared_main

        shared_main()
    print(f"Colab stage {stage}: passed", flush=True)


if __name__ == "__main__":
    main()
