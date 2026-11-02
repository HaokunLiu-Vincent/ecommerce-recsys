"""Tiny deterministic fixture authorization, never a production CLI bypass."""

import copy
import tempfile
from pathlib import Path

from hm_recsys.colab.files import INPUTS
from hm_recsys.colab.profiles import load_profile
from hm_recsys.contracts import digest
from hm_recsys.synthetic import create_fixture


def create_source(root: Path) -> None:
    """Create the exact existing 42-item synthetic source plus opaque unused indexes."""
    create_fixture(root)
    for modality in ("text", "image"):
        (root / f"{modality}_features/{modality}_faiss.index").write_bytes(
            b"synthetic unused content index"
        )


def assert_fixture(source: Path) -> None:
    """Match every source byte to a generated fixture before allowing local pilot.

    Arbitrary tiny real data or a self-declared marker is not sufficient. There
    is no CLI flag disabling the production host gate.
    """
    with tempfile.TemporaryDirectory(prefix="hm-fixture-reference-") as directory:
        reference = Path(directory) / "data"
        create_source(reference)
        for folder, names in INPUTS.items():
            for name in names:
                path = source / folder / name
                expected = reference / folder / name
                if (
                    not path.is_file()
                    or path.stat().st_size != expected.stat().st_size
                    or digest(path) != digest(expected)
                ):
                    raise ValueError(
                        "Local pilot requires exact generated synthetic input bytes"
                    )


def fixture_profile(project: Path) -> dict:
    """Reduce compute only for the authenticated fixture, including two real epochs."""
    profile = copy.deepcopy(load_profile(project, "pilot"))
    profile["device"] = "cpu"
    profile["training_subset"]["base_users"] = 12
    profile["compute"].update(
        wals_factors=8,
        wals_iterations=2,
        tt_batch_size=16,
        tt_epochs=2,
        tt_max_targets=96,
        tt_max_steps=12,
        cluster_count=8,
        cluster_max_iter=10,
        max_queries_per_fold=24,
        ranker_rounds=10,
        ranker_num_leaves=7,
        ranker_min_child_samples=5,
        ranker_early_stopping=5,
    )
    profile["checkpoint_every_steps"] = 2
    profile["synthetic_fixture_only"] = True
    return profile


def execute_fixture(
    project: Path, source: Path, local: Path, persistent: Path, *, until: str = "reload"
) -> dict:
    """Exercise the actual pilot stages on byte-authenticated bounded synthetic data."""
    from hm_recsys.colab.runner import _execute_profile

    assert_fixture(source)
    return _execute_profile(
        project,
        source,
        local,
        persistent,
        profile=fixture_profile(project),
        until=until,
        synthetic_fixture=True,
    )
