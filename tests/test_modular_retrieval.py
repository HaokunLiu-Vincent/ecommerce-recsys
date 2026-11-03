"""Check canonical rule stages, native fusion and immutable upstream statistics.

Synthetic purchase inputs exercise real SQL statistics/ItemCF and all three
rule algorithms. Separate hand-built source lists cover fusion math and error
contracts; they are not substitute model-training or H&M performance evidence.
All generated artifacts remain in pytest-owned temporary directories.
"""

from datetime import date
from pathlib import Path

import pytest

from hm_recsys.core.contracts import ROOT, digest, read_json
from hm_recsys.core.synthetic import create_fixture
from hm_recsys.data.history import canonical, full_rosters
from hm_recsys.fusion.native import merge
from hm_recsys.retrieval import itemcf
from hm_recsys.retrieval.pipeline import run_channel
from hm_recsys.retrieval.snapshot import build_statistics


def test_native_union_preserves_complete_provenance() -> None:
    """Use actual source ranks, deterministic IDs and exact reciprocal-rank sums."""
    outputs = {
        name: []
        for name in ("trend", "repurchase", "sibling", "itemcf", "wals", "twotower")
    }
    outputs["trend"] = [("0000000002", 3.0), ("0000000001", 2.0)]
    outputs["wals"] = [("0000000001", 0.5)]
    result = merge(outputs, {"0000000001", "0000000002"}, {"0000000002"})
    assert [c.article for c in result] == ["0000000001", "0000000002"]
    assert result[0].rrf == 1 / 62 + 1 / 61
    assert result[0].hits["trend"].rank == 2
    assert result[0].hits["wals"].score == 0.5
    assert result[1].hits["trend"].fallback
    outputs["wals"].append(("0000000001", 0.4))
    with pytest.raises(ValueError, match="uniqueness"):
        merge(outputs, {"0000000001", "0000000002"})


def test_rule_cells_and_itemcf_do_not_mutate_shared_statistics(tmp_path: Path) -> None:
    """Build real rule outputs before ItemCF, then preserve the upstream DB hash."""
    source = tmp_path / "source"
    run = tmp_path / "run"
    run.mkdir()
    create_fixture(source)
    profile = read_json(ROOT / "configs/colab_full.json")
    profile["device"] = "cpu"
    profile["performance"] = {"backend": "optimized"}
    canonical(run, source, profile)
    full_rosters(run, profile)
    build_statistics(run, "Validation", date(2020, 9, 8))
    shared = run / "snapshots/Validation.duckdb"
    before = digest(shared)
    assert not (run / "indexes/itemcf_Validation.duckdb").exists()
    for channel in ("trend", "repurchase", "sibling"):
        report = run_channel(run, profile, channel, ["Validation"])["Validation"]
        assert report["queries"] > 0
        assert report["rows"] <= report["queries"] * report["quota"]
    with pytest.raises(ValueError, match="Missing ItemCF"):
        run_channel(run, profile, "itemcf", ["Validation"])
    itemcf.build(run, "Validation", date(2020, 9, 8), optimized=True, profile=profile)
    assert digest(shared) == before
    report = run_channel(run, profile, "itemcf", ["Validation"])["Validation"]
    assert report["queries"] > 0
    assert report["labels_read"]
    assert report["quota"] == 50
    assert digest(shared) == before


def test_columnar_fusion_retains_zero_candidate_queries(tmp_path: Path) -> None:
    """Validate persisted fusion independently using explicit synthetic source hits."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hm_recsys.core.contracts import CHANNELS, write_json
    from hm_recsys.fusion.pipeline import fuse
    from hm_recsys.fusion.storage import CandidateReader
    from hm_recsys.retrieval.pipeline import SOURCE_SCHEMA

    run = tmp_path / "run"
    for folder in ("snapshots", "queries", "labels", "channels/Validation"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    cutoff = date(2020, 9, 8)
    with duckdb.connect(str(run / "snapshots/Validation.duckdb")) as con:
        con.execute(
            "CREATE TABLE active AS SELECT "
            "unnest(['0000000001','0000000002']) AS article_id"
        )
    pq.write_table(
        pa.table({"customer_id": ["fixture-a", "fixture-b"]}),
        run / "queries/Validation.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "customer_id": ["fixture-a", "fixture-b"],
                "article_id": ["0000000001", "0000000002"],
            }
        ),
        run / "labels/Validation.parquet",
    )
    write_json(
        run / "queries/Validation.json",
        {
            "partition": "Validation",
            "cutoff": str(cutoff),
            "start": "2020-09-09",
            "end": "2020-09-16",
            "file": "queries/Validation.parquet",
            "count": 2,
        },
    )
    for channel in CHANNELS:
        records = (
            [
                {
                    "customer_id": "fixture-a",
                    "cutoff_date": cutoff,
                    "article_id": item,
                    "source_rank": j + 1,
                    "raw_score": score,
                    "trend_global_backfill": False,
                }
                for j, (item, score) in enumerate(
                    [("0000000002", 3.0), ("0000000001", 2.0)]
                )
            ]
            if channel == "trend"
            else []
        )
        path = run / "channels/Validation" / f"{channel}.parquet"
        pq.write_table(pa.Table.from_pylist(records, schema=SOURCE_SCHEMA), path)
        write_json(
            path.with_suffix(".json"),
            {"queries": 2, "cutoff": str(cutoff), "sha256": digest(path)},
        )
    report = fuse(run, {"folds": ["Validation"], "feature_query_batch": 32})[
        "Validation"
    ]
    assert report["queries"] == 2
    assert report["zero_candidate_queries"] == 1
    assert report["union_recall"] == 0.5
    assert report["rrf_baseline"]["MAP@12"] == 0.25
    reader = CandidateReader(run, "Validation")
    try:
        candidates = reader.read(["fixture-a", "fixture-b"])
        assert candidates["fixture-b"] == []
        assert [c.article for c in candidates["fixture-a"]] == [
            "0000000001",
            "0000000002",
        ]
        assert candidates["fixture-a"][0].hits["trend"].rank == 2
    finally:
        reader.close()
