"""Bounded verified file transfers; completion manifests are published last."""

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from hm_recsys.contracts import digest, read_json

INPUTS = {
    "original data": ("articles.csv", "customers.csv", "transactions_train.csv"),
    "text_features": (
        "text_article_ids.parquet",
        "text_embeddings.npy",
        "text_faiss.index",
        "text_features_manifest.json",
    ),
    "image_features": (
        "image_article_ids.parquet",
        "image_embedding_failures.parquet",
        "image_embeddings.npy",
        "image_faiss.index",
        "image_features_manifest.json",
    ),
}
CHUNK = 1024 * 1024


def canonical_hash(value: object) -> str:
    """Hash a JSON identity independently of dictionary insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Replace a metadata file only after its complete bytes are flushed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_path(name: str) -> Path:
    """Reject traversal, Windows paths, empty names, and absolute members."""
    path = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or path.is_absolute():
        raise ValueError(f"Unsafe relative path: {name}")
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Unsafe relative path: {name}")
    return Path(*path.parts)


def safe_destination(root: Path, name: str) -> Path:
    """Resolve only beneath root, rejecting symlink parents as well as leaves."""
    relative = relative_path(name)
    current = root
    if current.is_symlink():
        raise ValueError("Symlink destination root")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink destination component")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("Destination escapes root")
    return current


def require_space(path: Path, required: int) -> None:
    """Check free disk without deleting any inputs or previous results."""
    existing = path
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < required:
        raise RuntimeError(f"blocked_resource: need {required} free bytes")


def file_identity(path: Path) -> dict:
    """Portable cheap change detector; this is not a cryptographic checksum."""
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verified_copy(
    source: Path,
    target: Path,
    *,
    expected_sha256: str | None = None,
    after_chunk: Callable[[int], None] | None = None,
) -> dict:
    """Copy through an incomplete file, hash both ends, and preserve source mtime.

    Existing different targets are never overwritten. Failure removes only this
    call's temporary file; a failed source or destination is not a completed copy.
    """
    if source.is_symlink() or not source.is_file():
        raise ValueError("Expected a regular source file")
    if target.is_symlink():
        raise ValueError("Symlink transfer target")
    before = file_identity(source)
    require_space(target.parent, before["bytes"] + CHUNK)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    hasher = hashlib.sha256()
    count = 0
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            for block in iter(lambda: reader.read(CHUNK), b""):
                writer.write(block)
                hasher.update(block)
                count += len(block)
                if after_chunk is not None:
                    after_chunk(count)
            writer.flush()
            os.fsync(writer.fileno())
        sha = hasher.hexdigest()
        if before != file_identity(source) or count != before["bytes"]:
            raise ValueError("Source changed during copy")
        if expected_sha256 is not None and sha != expected_sha256:
            raise ValueError("Source checksum mismatch")
        if digest(temporary) != sha:
            raise ValueError("Destination checksum mismatch")
        if target.exists():
            if target.stat().st_size != count or digest(target) != sha:
                raise ValueError("Refusing to overwrite an incompatible existing file")
        else:
            # O_EXCL reserves the final path. The manifest is the completion
            # authority; a crash before replace leaves an invalid empty file.
            descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            os.replace(temporary, target)
            os.utime(target, ns=(before["mtime_ns"], before["mtime_ns"]))
        return {"bytes": count, "sha256": sha, "source_stat": before}
    finally:
        temporary.unlink(missing_ok=True)


def stage_inputs(source: Path, local: Path, *, reverify: bool = False) -> dict:
    """Copy only the three allowlisted input directories, never old runs.

    Fast reuse compares source size/mtime plus a local SHA256. --reverify hashes
    the source too, detecting same-size changes with preserved timestamps.
    """
    if source.resolve() == local.resolve():
        raise ValueError("Source and local input roots must differ")
    required = [
        f"{folder}/{name}" for folder, names in INPUTS.items() for name in names
    ]
    for name in required:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing regular input: {name}")
    ledger_path = local / "staging_manifest.json"
    old = read_json(ledger_path) if ledger_path.exists() else {"files": {}}
    needed = sum(
        (source / n).stat().st_size for n in required if not (local / n).exists()
    )
    require_space(local, needed + 2 * CHUNK)
    result = {
        "version": 1,
        "complete": False,
        "files": {},
        "reuse_policy": (
            "source size/mtime detector plus destination digest; reverify hashes source"
        ),
    }
    for name in required:
        src = source / name
        target = safe_destination(local, name)
        previous = old["files"].get(name)
        reusable = (
            previous
            and target.is_file()
            and previous["source_stat"] == file_identity(src)
            and digest(target) == previous["sha256"]
            and (not reverify or digest(src) == previous["sha256"])
        )
        if reusable:
            result["files"][name] = previous
        else:
            # Preserve corrupted or stale local copies for inspection. Explicit
            # re-staging never destroys the old bytes or any run directory.
            if target.exists():
                quarantine = target.with_name(
                    f"{target.name}.rejected-{uuid.uuid4().hex}"
                )
                target.rename(quarantine)
            expected = _declared_hash(source, name)
            result["files"][name] = verified_copy(src, target, expected_sha256=expected)
        atomic_json(ledger_path, result)
    result["complete"] = True
    result["input_digest"] = canonical_hash(
        {n: {k: v[k] for k in ("bytes", "sha256")} for n, v in result["files"].items()}
    )
    atomic_json(ledger_path, result)
    return result


def _declared_hash(source: Path, name: str) -> str | None:
    folder, filename = name.split("/")
    if folder == "original data" or filename.endswith("manifest.json"):
        return None
    modality = folder.split("_")[0]
    manifest = read_json(source / folder / f"{modality}_features_manifest.json")
    return manifest.get("artifact_sha256", {}).get(filename) or manifest.get(
        "files", {}
    ).get(filename, {}).get("sha256")


def verify_files(root: Path, files: dict) -> None:
    """Require actual file hashes, not a directory or completion marker alone."""
    for name, identity in files.items():
        path = safe_destination(root, name)
        if not path.is_file() or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"Missing/size-mismatched artifact: {name}")
        if digest(path) != identity["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
