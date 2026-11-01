"""Internal stage entry point with isolated macOS numerical runtimes."""

import sys
from pathlib import Path

from hm_recsys.contracts import Config, read_json


def main() -> None:
    """Execute one CPU stage without co-initializing incompatible OpenMP copies."""
    stage, run_string = sys.argv[1:]
    run = Path(run_string).resolve()
    config = Config.load(run / "config.resolved.json")
    source = Path(read_json(run / "run_manifest.json")["source_root"])
    from hm_recsys.data import Sample, extract_sample
    from hm_recsys.pipeline import (
        candidates_and_features,
        catalog_ids,
        load_content,
        preflight,
        reload_features,
        reload_inference,
        seal_run,
        verify_frozen,
    )

    if stage in ("wals", "index_towers", "features", "reload_features"):
        import faiss

        faiss.omp_set_num_threads(config.limit("threads"))
    if stage in ("towers", "encode", "reload_encode"):
        import torch

        torch.set_num_threads(config.limit("threads"))
        if stage != "towers":
            torch.set_num_interop_threads(config.limit("interop_threads"))
    if stage == "fixture":
        from hm_recsys.synthetic import create_fixture

        create_fixture(source)
    elif stage == "preflight":
        preflight(run, source, config)
    elif stage == "sample":
        extract_sample(run, source, config)
    elif stage == "wals":
        from hm_recsys.learned import WALS

        WALS.fit(Sample.load(run, config), config, run)
    elif stage in ("towers", "preprocessing", "encode", "reload_encode"):
        if stage == "reload_encode":
            from hm_recsys.content import ContentBank

            verify_frozen(run, source)
            sample = Sample.load(run / "inference_input", config)
            content = ContentBank(source, list(sample.articles), catalog_ids(source))
        else:
            sample = Sample.load(run, config)
            content = load_content(run, sample, source)
        if stage == "towers":
            from hm_recsys.towers import fit_towers

            fit_towers(sample, content, config, run)
        elif stage == "preprocessing":
            from hm_recsys.features import fit_preprocessing

            fit_preprocessing(sample, content, config, run)
        else:
            from hm_recsys.towers import encode_queries

            encode_queries(sample, content, run, reload=stage == "reload_encode")
    elif stage == "index_towers":
        from hm_recsys.learned import build_tower_index

        build_tower_index(run)
    elif stage == "features":
        candidates_and_features(run, source, config)
    elif stage == "ranker":
        from hm_recsys.ranking import train_ranker

        train_ranker(run, config)
        seal_run(run, source)
    elif stage == "reload_features":
        reload_features(run, source, config)
    elif stage == "reload":
        reload_inference(run, source, config)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    print(f"Stage {stage}: passed", flush=True)


if __name__ == "__main__":
    main()
