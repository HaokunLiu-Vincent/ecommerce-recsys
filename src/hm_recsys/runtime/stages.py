"""Dispatch canonical numerical phases without duplicating model implementations.

Consumes a private run view, resolved profile and explicit worker options; emits
canonical data, retrieval, feature, model or evaluation artifacts. Imports the
eight layers lazily to isolate native runtimes. Future-label Test phases require a
separately authorized options record.
"""

from datetime import date
from pathlib import Path

from hm_recsys.core.contracts import Config, read_json, split_manifest, write_json


def cutoff_for(name: str) -> date:
    """Return the adopted inclusive cutoff without opening future labels."""
    return date.fromisoformat(
        next(s["cutoff"] for s in split_manifest() if s["partition"] == name)
    )


def run_stage(stage: str, run: Path, profile: dict, options: dict) -> None:
    """Execute exactly one isolated phase with no implicit prerequisite fits."""
    source = Path(options["data_root"])
    if stage == "preflight":
        from hm_recsys.runtime.acceptance import preflight

        preflight(run, source, Config(profile["compute"]))
    elif stage == "canonical":
        from hm_recsys.data.history import canonical

        canonical(run, source, profile)
    elif stage == "content":
        from hm_recsys.features.content import adapt_content

        adapt_content(run, source)
    elif stage == "rosters":
        from hm_recsys.data.history import rosters

        rosters(run, profile)
    elif stage == "capacity":
        from hm_recsys.runtime.capacity import feature_capacity

        feature_capacity(run, profile)
    elif stage == "prepare_towers":
        from hm_recsys.data.history import History
        from hm_recsys.data.streaming import TargetStream
        from hm_recsys.features.content import MappedContent
        from hm_recsys.retrieval.two_tower.inputs import SqlEncoder
        from hm_recsys.retrieval.two_tower.prepared import prepare

        history = History(run)
        try:
            encoder = SqlEncoder(history, MappedContent(run))
            prepare(
                run,
                encoder,
                TargetStream(history, profile["compute"]["tt_max_targets"]),
            )
        finally:
            history.close()
    elif stage == "device_probe":
        from hm_recsys.runtime.device import probe

        probe(run / "probe", profile["device"])
    elif stage == "wals_fit":
        from hm_recsys.retrieval.wals.training import fit_wals

        fit_wals(run, profile)
    elif stage == "towers_fit":
        from hm_recsys.retrieval.two_tower.training import fit

        fit(run, profile, options)
    elif stage == "index_towers":
        import numpy as np

        from hm_recsys.retrieval.vectors import BlockIndex

        state = read_json(run / "models/twotower_items.json")
        vectors = np.load(
            run / "models/twotower_items.npy", mmap_mode="r", allow_pickle=False
        )
        BlockIndex(state["article_ids"], vectors).save(run / "indexes", "twotower")
    elif stage in ("encode", "reload_encode", "encode_test"):
        from hm_recsys.retrieval.two_tower.inference import encode

        if stage == "encode_test":
            encode(
                run,
                {**profile, "folds": ["Test"], "prediction_scope": "final_test"},
                partitions=["Test"],
                output_scope="final_test",
            )
        else:
            encode(
                run,
                profile,
                reloaded=stage == "reload_encode",
                partitions=["Validation"] if stage == "reload_encode" else None,
            )
    elif stage.startswith("statistics_"):
        from hm_recsys.retrieval.snapshot import build_statistics

        name = stage.removeprefix("statistics_")
        build_statistics(run, name, cutoff_for(name))
    elif stage.startswith("itemcf_"):
        from hm_recsys.retrieval.itemcf import build

        name = stage.removeprefix("itemcf_")
        build(run, name, cutoff_for(name), optimized=True, profile=profile)
    elif stage.startswith("channel_"):
        from hm_recsys.retrieval.pipeline import run_channel

        channel = stage.removeprefix("channel_")
        reloaded = bool(options.get("reloaded"))
        partitions = (
            ["Test"]
            if options.get("final_test_active")
            else ["Validation"]
            if reloaded
            else None
        )
        report = run_channel(
            run, profile, channel, partitions=partitions, reloaded=reloaded
        )
        write_json(run / "phase_reports" / f"{stage}.json", report)
    elif stage == "fusion_build":
        from hm_recsys.fusion.pipeline import fuse

        reloaded = bool(options.get("reloaded"))
        report = fuse(
            run,
            profile,
            partitions=["Test"]
            if options.get("final_test_active")
            else ["Validation"]
            if reloaded
            else None,
            reloaded=reloaded,
        )
        write_json(run / "phase_reports" / "fusion_build.json", report)
    elif stage == "preprocessing":
        from hm_recsys.features.preprocessing import preprocessing

        preprocessing(run, profile)
    elif stage.startswith("features_"):
        from hm_recsys.features.pipeline import generate

        generate(
            run,
            profile,
            stage.removeprefix("features_"),
            reloaded=bool(options.get("reloaded")),
        )
    elif stage == "ranker_fit":
        from hm_recsys.ranking.pipeline import train

        train(run, profile)
    elif stage == "reload_score":
        from hm_recsys.ranking.pipeline import reload

        reload(run)
    elif stage == "test_roster":
        if not options.get("final_test_active"):
            raise PermissionError(
                "Test labels require separate final Test authorization"
            )
        from hm_recsys.data.history import prepare_test_roster

        prepare_test_roster(run, source, authorized=True)
    elif stage == "test_score":
        if not options.get("final_test_active"):
            raise PermissionError(
                "Test scoring requires separate final Test authorization"
            )
        from hm_recsys.ranking.pipeline import predict_partition

        predict_partition(run, profile, "Test")
    else:
        raise ValueError(f"Unknown canonical worker phase: {stage}")


def validate(stage: str, work: Path, profile: dict) -> None:
    """Require the declared functional stage's report and essential artifacts."""
    report = read_json(work / "reports" / f"{stage}.json")
    if report["stage"] != stage or report["status"] != "completed":
        raise ValueError("Stage did not publish a successful aggregate report")
    if stage == "prepare":
        required = [
            "canonical/history.duckdb",
            "population_manifest.json",
            "content/mapping.json",
            "split_manifest.json",
        ]
    elif stage == "wals":
        required = ["models/wals.json", "indexes/wals.npy"]
    elif stage == "twotower":
        required = [
            "models/twotower.pt",
            "indexes/twotower.faiss",
            "queries/query_vectors.json",
        ]
    elif stage == "features":
        required = ["feature_schema.json"] + [
            f"features/{name}/manifest.json" for name in profile["folds"]
        ]
    elif stage == "ranker":
        required = ["models/lambdamart.txt", f"{profile['profile']}_results.json"]
    elif stage == "reload":
        required = ["reload_results.json", "predictions/reloaded_top12.parquet"]
    elif stage == "final_test":
        required = [
            "final_test_results.json",
            "final_test/predictions.parquet",
            "final_test_authorization.json",
        ]
    else:
        required = []
    for relative in required:
        if not (work / relative).is_file():
            raise ValueError(f"Missing required stage artifact: {relative}")
