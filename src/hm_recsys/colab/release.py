"""Reproducible allowlisted releases, with a stdlib-only extraction verifier."""

import os
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

from hm_recsys.colab.files import (
    atomic_json,
    canonical_hash,
    relative_path,
)
from hm_recsys.contracts import digest, read_json

DOCS = (
    "docs/proposal.md",
    "docs/archi/architecture.md",
    "docs/archi/evaluation_protocol.md",
    "docs/archi/implementation_decisions.md",
    "docs/archi/local_smoke_report.md",
    "docs/archi/codex_local_smoke_prompt.md",
    "docs/archi/codex_colab_migration_prompt.md",
    "docs/archi/colab_runbook.md",
    "docs/archi/colab_implementation_report.md",
    "docs/archi/codex_colab_pilot_prompt.md",
    "docs/archi/colab_pilot_runbook.md",
    "docs/archi/colab_pilot_implementation_report.md",
)
ROOT_FILES = ("AGENTS.md", "README.md", "pyproject.toml", "uv.lock", ".python-version")
NOTEBOOKS = (
    "notebooks/01_Colab_Training.ipynb",
    "notebooks/03_text_embeddings_extraction.ipynb",
)


def allowed(name: str) -> bool:
    """Allow source-like files explicitly, excluding data, credentials and archives."""
    path = relative_path(name)
    if any(
        part.startswith(".")
        or Path(part).stem.lower()
        in ("__pycache__", "credentials", "secrets", "tokens")
        for part in path.parts
    ):
        return name == ".python-version"
    if name in (*ROOT_FILES, *DOCS, *NOTEBOOKS):
        return True
    return (
        (name.startswith("src/hm_recsys/") and path.suffix == ".py")
        or (path.parts[0] in ("scripts", "tests") and path.suffix == ".py")
        or (path.parts[0] == "configs" and path.suffix == ".json")
    )


def release_files(project: Path) -> list[str]:
    """Return deterministic release members, refusing symlinked code inputs."""
    required = [*ROOT_FILES, *DOCS, *NOTEBOOKS]
    for name in required:
        if not (project / name).is_file():
            raise FileNotFoundError(f"Required release member missing: {name}")
    names = set(required)
    for folder in ("src/hm_recsys", "scripts", "tests", "configs"):
        for path in (project / folder).rglob("*"):
            name = path.relative_to(project).as_posix()
            if allowed(name) and path.is_file():
                names.add(name)
    for name in names:
        path = project / name
        if path.is_symlink() or any(
            parent.is_symlink() for parent in path.parents if parent != project.parent
        ):
            raise ValueError("Release source contains a symlink")
    return sorted(names)


def build_release(project: Path, destination: Path) -> dict:
    """Build a content-addressed ZIP, external manifest, and checksum file.

    The external manifest hashes the ZIP; the archive contains only source
    members, avoiding self-referential hashes. ZIP timestamps/modes are fixed.
    """
    files = {
        name: {
            "bytes": (project / name).stat().st_size,
            "sha256": digest(project / name),
        }
        for name in release_files(project)
    }
    build_id = canonical_hash(files)[:16]
    name = f"hm_recsys_{build_id}.zip"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    revision = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    metadata = {
        "version": 1,
        "build_id": build_id,
        "files": files,
        "lock_sha256": files["uv.lock"]["sha256"],
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_is_identity_authority": False,
    }
    with tempfile.NamedTemporaryFile(
        dir=destination, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for member in files:
                info = zipfile.ZipInfo(member, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, (project / member).read_bytes())
        sha = digest(temporary)
        if target.exists() and digest(target) != sha:
            raise ValueError("Existing release with same identity has different bytes")
        if not target.exists():
            os.replace(temporary, target)
        metadata.update(archive=name, archive_sha256=sha)
        manifest_path = destination / f"{target.stem}.manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != metadata:
            raise ValueError("Existing release metadata differs")
        atomic_json(manifest_path, metadata)
        (destination / f"{target.stem}.sha256").write_text(
            f"{sha}  {name}\n", encoding="utf-8"
        )
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def extract_release(archive_path: Path, manifest_path: Path, target: Path) -> dict:
    """Use the same stdlib verifier executed by the independent notebook."""
    from hm_recsys.colab.bootstrap import unpack

    manifest = read_json(manifest_path)
    for name in manifest["files"]:
        if not allowed(name):
            raise ValueError("Disallowed release member")
    unpack(archive_path, manifest, target)
    return manifest
