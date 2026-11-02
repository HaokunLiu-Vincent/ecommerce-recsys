"""Isolated pilot stage dispatch; algorithms remain reusable module functions."""

from datetime import date
from pathlib import Path

from hm_recsys.contracts import Config, read_json, split_manifest

STAGES = (
    "preflight",
    "canonical",
    "content",
    "rosters",
    "device_probe",
    "wals",
    "towers",
    "index_towers",
    "encode",
    "preprocessing",
    "stats_F1",
    "features_F1",
    "stats_Validation",
    "features_Validation",
    "ranker",
    "reload_encode",
    "reload_features",
    "reload",
)


def run_stage(stage: str, run: Path, profile: dict, options: dict) -> None:
    """Use separate Torch, FAISS and LightGBM processes, including on CPU fixtures."""
    from hm_recsys.colab.profiles import is_colab_runtime

    if not is_colab_runtime():
        if not options.get("synthetic_fixture"):
            raise ValueError(
                "Real pilot workers are prohibited on the development host"
            )
        from hm_recsys.pilot.testing import assert_fixture

        assert_fixture(Path(options["data_root"]))
    source = Path(options["data_root"])
    if stage == "preflight":
        from hm_recsys.pipeline import preflight

        preflight(run, source, Config(profile["compute"]))
    elif stage == "canonical":
        from hm_recsys.pilot.data import canonical

        canonical(run, source, profile)
    elif stage == "content":
        from hm_recsys.pilot.data import adapt_content

        adapt_content(run, source)
    elif stage == "rosters":
        from hm_recsys.pilot.data import rosters

        rosters(run, profile)
    elif stage == "device_probe":
        from hm_recsys.colab.device import probe

        probe(run / "probe", profile["device"])
    elif stage in ("wals", "preprocessing"):
        from hm_recsys.pilot.models import fit_wals, preprocessing

        (fit_wals if stage == "wals" else preprocessing)(run, profile)
    elif stage == "towers":
        from hm_recsys.pilot.towers import fit

        fit(run, profile, options)
    elif stage == "index_towers":
        import faiss
        import numpy as np

        from hm_recsys.pilot.vectors import BlockIndex

        faiss.omp_set_num_threads(2)
        BlockIndex(
            read_json(run / "models/twotower_items.json")["article_ids"],
            np.load(
                run / "models/twotower_items.npy", mmap_mode="r", allow_pickle=False
            ),
        ).save(run / "indexes", "twotower")
    elif stage in ("encode", "reload_encode"):
        from hm_recsys.pilot.towers import encode

        if stage == "reload_encode":
            from hm_recsys.pipeline import verify_frozen

            verify_frozen(run, source)
        encode(run, profile, reload=stage == "reload_encode")
    elif stage.startswith("stats_"):
        from hm_recsys.pilot.snapshot import build_snapshot

        name = stage.removeprefix("stats_")
        part = next(p for p in split_manifest() if p["partition"] == name)
        build_snapshot(run, name, date.fromisoformat(part["cutoff"]))
    elif stage.startswith("features_") or stage == "reload_features":
        import faiss

        from hm_recsys.pilot.features import generate

        faiss.omp_set_num_threads(2)
        generate(
            run,
            "Validation"
            if stage == "reload_features"
            else stage.removeprefix("features_"),
            profile,
            reload=stage == "reload_features",
        )
    elif stage == "ranker":
        from hm_recsys.pilot.ranking import train
        from hm_recsys.pipeline import seal_run

        train(run, profile)
        seal_run(run, source)
    elif stage == "reload":
        from hm_recsys.pilot.ranking import reload
        from hm_recsys.pipeline import verify_frozen

        verify_frozen(run, source)
        reload(run)
    else:
        raise ValueError("Unknown pilot stage")


def validate(stage: str, work: Path, profile: dict) -> None:
    """Require stage-specific output contracts before the existing commit protocol."""
    from hm_recsys.colab.runner import validate_stage

    required = {
        "device_probe": ["probe/device_probe.json", "probe/probe_weights.pt"],
        "canonical": [
            "canonical/history.duckdb",
            "population_manifest.json",
            "cohort_manifest.json",
        ],
        "content": [
            "content/mapping.json",
            "content/text.npy",
            "content/image.npy",
            "content/acceptance.json",
        ],
        "rosters": [f"queries/{name}.json" for name in profile["folds"]],
        "ranker": [
            "models/lambdamart.txt",
            "pilot_results.json",
            "ranker_memory_estimate.json",
        ],
        "reload_features": ["predictions/reloaded_features/manifest.json"],
    }
    if stage.startswith("stats_"):
        name = stage.removeprefix("stats_")
        required[stage] = [f"snapshots/{name}.duckdb", f"snapshots/{name}.json"]
    if stage.startswith("features_"):
        name = stage.removeprefix("features_")
        required[stage] = [f"features/{name}/manifest.json", "candidate_results.json"]
    if stage not in required:
        validate_stage(stage, work)
        return
    for name in required[stage]:
        if not (work / name).is_file():
            raise ValueError(f"Missing pilot stage output: {name}")
    if stage == "device_probe":
        result = read_json(work / "probe/device_probe.json")
        if result["status"] != "passed" or result["device"] != profile["device"]:
            raise ValueError("Device probe did not validate the requested device")
    if stage.startswith("features_") or stage == "reload_features":
        manifest = read_json(work / required[stage][0])
        if sum(s["rows"] for s in manifest["shards"]) != manifest["rows"]:
            raise ValueError("Shard row counts do not match manifest")
