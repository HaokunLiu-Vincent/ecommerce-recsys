"""Stdlib-only verified bootstrap, usable before the project is extracted."""

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    """Hash a file with bounded reads before trusting any archived Python code."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_member(name: str) -> Path:
    """Reject traversal, links, hidden credentials, and data/environment roots."""
    posix = PurePosixPath(name)
    parts = name.split("/")
    if (
        posix.is_absolute()
        or "\\" in name
        or ":" in name
        or any(p in ("", ".", "..") for p in parts)
    ):
        raise ValueError("Unsafe archive member")
    if any(p.startswith(".") for p in parts) and name != ".python-version":
        raise ValueError("Hidden archive member")
    if parts[0] not in (
        "src",
        "scripts",
        "configs",
        "tests",
        "docs",
        "notebooks",
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
    ):
        raise ValueError("Disallowed archive root")
    if "achive" in parts or any(
        Path(p).stem.lower() in ("credentials", "secrets", "tokens") for p in parts
    ):
        raise ValueError("Disallowed archive content")
    return Path(*posix.parts)


def unpack(archive_path: Path, manifest: dict, target: Path) -> None:
    """Verify all source members and publish a new project directory atomically.

    A matching existing release is verified and reused. A different checkout
    requires a new operator-selected destination; this function never deletes it.
    """
    if sha256_file(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Archive checksum mismatch")
    identity = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:16]
    if identity != manifest["build_id"]:
        raise ValueError("Release identity mismatch")
    if target.is_symlink():
        raise ValueError("Symlink project target")
    if target.exists():
        marker = target / "release_manifest.json"
        if not marker.is_file() or json.loads(marker.read_text()) != manifest:
            raise ValueError("Existing checkout differs; select a new target")
        for name, expected in manifest["files"].items():
            path = target / safe_member(name)
            if path.is_symlink() or not path.resolve().is_relative_to(target.resolve()):
                raise ValueError("Unsafe existing project path")
            if (
                path.stat().st_size != expected["bytes"]
                or sha256_file(path) != expected["sha256"]
            ):
                raise ValueError("Existing checkout content changed")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    needed = sum(entry["bytes"] for entry in manifest["files"].values())
    if shutil.disk_usage(target.parent).free < needed + 1024**2:
        raise RuntimeError("Insufficient disk for release extraction")
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".release-") as temp:
        staging = Path(temp) / "project"
        staging.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
                raise ValueError("Release member list mismatch")
            for info in archive.infolist():
                name = safe_member(info.filename)
                if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir():
                    raise ValueError("Archive links/directories are prohibited")
                expected = manifest["files"][info.filename]
                if info.file_size != expected["bytes"] or info.file_size > 32 * 1024**2:
                    raise ValueError("Unexpected archive member size")
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as reader, path.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024**2)
                if sha256_file(path) != expected["sha256"]:
                    raise ValueError("Extracted member checksum mismatch")
        (staging / "release_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(staging, target)
