"""Bounded immutable input staging and verified atomic artifact transfer.

Consumes the three approved source directories and explicit destination paths;
emits hash-verified files and completion manifests. Uses pathlib, streaming hashes
and disk guards. Source embeddings, mappings and raw CSVs are never regenerated or
mutated.
"""

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from hm_recsys.core.contracts import digest, read_json

# Publish aggregate work counters without exposing source rows or changing RNG.
from hm_recsys.core.progress import emit

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


# Hash a JSON identity independently of dictionary insertion order.
def canonical_hash(value: object) -> str:
    """Hash a JSON identity independently of dictionary insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


# Replace a metadata file only after its complete bytes are flushed.
def atomic_json(path: Path, value: object) -> None:
    """Replace a metadata file only after its complete bytes are flushed."""
    # Create only the required owned directory; immutable inputs and other runs
    # remain untouched.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    # Protect this operation with explicit failure/cleanup handling so partial work
    # is never reported as complete.
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            # Serialize the explicitly selected metadata/records to this operation's
            # output; raw source inputs are not overwritten.
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            # Serialize the explicitly selected metadata/records to this operation's
            # output; raw source inputs are not overwritten.
            handle.write("\n")
            # Flush the completed buffer/stream before a later stage can publish its
            # manifest.
            handle.flush()
            # Request durable local bytes before the completion pointer can advance.
            os.fsync(handle.fileno())
        # Publish/move the owned filesystem entry at this explicit boundary after its
        # required validation.
        os.replace(temporary, path)
    finally:
        # Remove only the identified owned temporary/lease file, never a source
        # dataset or old run.
        temporary.unlink(missing_ok=True)


# Reject traversal, Windows paths, empty names, and absolute members.
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


# Check free disk without deleting any inputs or previous results.
def require_space(path: Path, required: int) -> None:
    """Check free disk without deleting any inputs or previous results."""
    existing = path
    # Continue this owned loop only under its stated condition, respecting the
    # consumed-offset/resource boundary.
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < required:
        raise RuntimeError(f"blocked_resource: need {required} free bytes")


# Portable cheap change detector; this is not a cryptographic checksum.
def file_identity(path: Path) -> dict:
    """Portable cheap change detector; this is not a cryptographic checksum."""
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# Copy through an incomplete file, hash both ends, and preserve source mtime.
def verified_copy(
    source: Path,
    target: Path,
    *,
    expected_sha256: str | None = None,
    after_chunk: Callable[[int], None] | None = None,
    measurements=None,
) -> dict:
    """Copy through an incomplete file, hash both ends, and preserve source mtime.

    Existing different targets are never overwritten. Failure removes only this
    call's temporary file; a failed source or destination is not a completed copy.
    """
    if source.is_symlink() or not source.is_file():
        raise ValueError("Expected a regular source file")
    if target.is_symlink():
        raise ValueError("Symlink transfer target")
    import time

    started = time.perf_counter()
    # Capture source file metadata before I/O so concurrent content changes
    # invalidate verification.
    before = file_identity(source)
    # Check scratch/output disk capacity without deleting any existing run or input.
    require_space(target.parent, before["bytes"] + CHUNK)
    # Create only the required owned directory; immutable inputs and other runs
    # remain untouched.
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    hasher = hashlib.sha256()
    count = 0
    # Byte totals come from the opened source stat; checksum success is separate.
    emit(
        "file_copy",
        0,
        before["bytes"],
        unit="bytes",
        status="started",
        message="Copying to a local temporary file; verification is still pending",
        force=True,
    )
    # Protect this operation with explicit failure/cleanup handling so partial work
    # is never reported as complete.
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            for block in iter(lambda: reader.read(CHUNK), b""):
                # Serialize the explicitly selected metadata/records to this
                # operation's output; raw source inputs are not overwritten.
                writer.write(block)
                hasher.update(block)
                count += len(block)
                # Count bytes actually written, not an estimate from elapsed time.
                emit(
                    "file_copy",
                    count,
                    before["bytes"],
                    unit="bytes",
                    message="Bytes written; source and destination checks pending",
                )
                if after_chunk is not None:
                    after_chunk(count)
            # Flush the completed buffer/stream before a later stage can publish its
            # manifest.
            writer.flush()
            # Request durable local bytes before the completion pointer can advance.
            os.fsync(writer.fileno())
        if measurements:
            measurements.wall["checkpoint_copy_and_source_hash"] += (
                time.perf_counter() - started
            )
            measurements.calls["checkpoint_copy_and_source_hash"] += 1
        verify_started = time.perf_counter()
        # Hash verification has no honest byte callback in the current digest API.
        emit(
            "file_verify",
            0,
            None,
            unit="checks",
            status="started",
            message="Verifying source stability and destination SHA256",
            force=True,
        )
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
            # Release this owned file/database/writer handle at the stage boundary.
            os.close(descriptor)
            # Publish/move the owned filesystem entry at this explicit boundary after
            # its required validation.
            os.replace(temporary, target)
            os.utime(target, ns=(before["mtime_ns"], before["mtime_ns"]))
        if measurements:
            measurements.wall["checkpoint_remote_verify_and_file_publish"] += (
                time.perf_counter() - verify_started
            )
            measurements.calls["checkpoint_remote_verify_and_file_publish"] += 1
        emit(
            "file_copy",
            count,
            before["bytes"],
            unit="bytes",
            status="completed",
            message="Copied bytes and published file verified",
            force=True,
        )
        # This is one completed verification, distinct from the copied-byte counter.
        emit(
            "file_verify",
            1,
            1,
            unit="checks",
            status="completed",
            message="Source and destination integrity verified",
            force=True,
        )
        return {"bytes": count, "sha256": sha, "source_stat": before}
    # A copy/checksum failure must remain visible and must retain its original error.
    except BaseException as error:
        # Report only the error class; paths/source contents do not enter progress.
        emit(
            "file_copy",
            count,
            before["bytes"],
            unit="bytes",
            status="failed",
            message="Copy or verification failed; no successful transfer recorded",
            metrics={"error_type": type(error).__name__},
            force=True,
        )
        # Preserve interruption and checksum exception propagation to the caller.
        raise
    finally:
        # Remove only the identified owned temporary/lease file, never a source
        # dataset or old run.
        temporary.unlink(missing_ok=True)


# Copy only the three allowlisted input directories, never old runs.
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
    # Check scratch/output disk capacity without deleting any existing run or input.
    require_space(local, needed + 2 * CHUNK)
    result = {
        "version": 1,
        "complete": False,
        "files": {},
        "reuse_policy": (
            "source size/mtime detector plus destination digest; reverify hashes source"
        ),
    }
    for index, name in enumerate(required):
        src = source / name
        target = safe_destination(local, name)
        previous = old["files"].get(name)
        # A fixed whitelist filename is safe to show; customer records are never logged.
        emit(
            "input_files",
            index,
            len(required),
            unit="files",
            status="running",
            message="Checking input copy/reuse",
            metrics={"file": name},
            force=True,
        )
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
                # Publish/move the owned filesystem entry at this explicit boundary
                # after its required validation.
                target.rename(quarantine)
            expected = _declared_hash(source, name)
            result["files"][name] = verified_copy(src, target, expected_sha256=expected)
        atomic_json(ledger_path, result)
        # Only a verified reused or copied file advances the completed file count.
        emit(
            "input_files",
            index + 1,
            len(required),
            unit="files",
            message="Input file verified",
            metrics={
                "file": name,
                "copy_status": "verified_reuse" if reusable else "copied_and_verified",
            },
            force=True,
        )
    result["complete"] = True
    result["input_digest"] = canonical_hash(
        {n: {k: v[k] for k in ("bytes", "sha256")} for n, v in result["files"].items()}
    )
    atomic_json(ledger_path, result)
    # The aggregate input stage is complete only after its final ledger is published.
    emit(
        "input_files",
        len(required),
        len(required),
        unit="files",
        status="completed",
        message="All twelve allowlisted inputs verified",
        force=True,
    )
    return result


#  declared hash.
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


class DigestCache:
    """Reuse verified bytes only within one coordinator and unchanged file metadata.

    Every fresh process hashes all files again. Device/inode, mode, size, mtime
    and ctime changes invalidate a receipt; no path-existence-only trust and no
    persistent checksum cache are used. This assumes the normal local filesystem
    change contract, not an adversary able to forge all kernel metadata.
    """

    #   init  .
    def __init__(self) -> None:
        self.receipts = {}
        self.hits = self.misses = self.bytes_read = 0

    # Hash changed files and reject concurrent changes while reading bytes.
    def digest(self, path: Path) -> str:
        """Hash changed files and reject concurrent changes while reading bytes."""
        path = path.resolve(strict=True)
        # Capture source file metadata before I/O so concurrent content changes
        # invalidate verification.
        before = path.stat()
        signature = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        old = self.receipts.get(path)
        if old is not None and old[0] == signature:
            self.hits += 1
            return old[1]
        value = digest(path)
        after = path.stat()
        if signature != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Artifact changed during checksum verification")
        self.receipts[path] = (signature, value)
        self.misses += 1
        self.bytes_read += before.st_size
        return value

    # Check every expected file; an altered stat forces cryptographic reread.
    def verify(self, root: Path, files: dict) -> None:
        """Check every expected file; an altered stat forces cryptographic reread."""
        for name, identity in files.items():
            path = safe_destination(root, name)
            if not path.is_file() or path.stat().st_size != identity["bytes"]:
                raise ValueError(f"Missing/size-mismatched artifact: {name}")
            if self.digest(path) != identity["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {name}")
