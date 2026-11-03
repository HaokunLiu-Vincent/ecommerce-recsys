"""Checksummed dependency-aware stage publication and recovery.

Consumes immutable run identity, declared dependency hashes and isolated worker
output; emits atomic completed-stage records and verified restored views. Depends
on file integrity helpers and writer leases. Directory existence is insufficient;
corruption, identity drift and incomplete dependencies fail closed.
"""

import contextlib
import json
import os
import shutil
import socket
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

from hm_recsys.core.contracts import read_json

# Persistence milestones use the real stage name even outside the Notebook.
from hm_recsys.core.progress import emit, scope
from hm_recsys.runtime.files import (
    DigestCache,
    atomic_json,
    canonical_hash,
    relative_path,
    safe_destination,
    verified_copy,
)

STAGES = (
    "prepare",
    "trend",
    "repurchase",
    "sibling",
    "itemcf",
    "wals",
    "twotower",
    "fusion",
    "features",
    "ranker",
    "validation",
    "reload",
)


# Reject concurrent writers; stale remote leases require explicit recovery.
@contextlib.contextmanager
def writer_lease(root: Path, *, recover: bool = False) -> Iterator[None]:
    """Reject concurrent writers; stale remote leases require explicit recovery."""
    # Create only the required owned directory; immutable inputs and other runs
    # remain untouched.
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "writer.lock"
    if recover and lock.exists():
        # Publish/move the owned filesystem entry at this explicit boundary after its
        # required validation.
        lock.rename(root / f"writer.recovered-{uuid.uuid4().hex}.json")
    token = uuid.uuid4().hex
    # Protect this operation with explicit failure/cleanup handling so partial work
    # is never reported as complete.
    try:
        with lock.open("x", encoding="utf-8") as handle:
            # Serialize the explicitly selected metadata/records to this operation's
            # output; raw source inputs are not overwritten.
            json.dump(
                {"token": token, "host": socket.gethostname(), "pid": os.getpid()},
                handle,
            )
    # Handle the declared exception type here and preserve the surrounding
    # failure/recovery contract.
    except FileExistsError as error:
        raise RuntimeError(
            "Run has another writer or stale lease; "
            "verify it stopped before --recover-lock"
        ) from error
    # Protect this operation with explicit failure/cleanup handling so partial work
    # is never reported as complete.
    try:
        # Yield this ordered result/context to the consumer without eagerly
        # materializing subsequent input rows.
        yield
    finally:
        if lock.exists() and read_json(lock).get("token") == token:
            # Remove only the identified owned temporary/lease file, never a source
            # dataset or old run.
            lock.unlink()


# Immutable stage deltas and hash-verified recovery, not directory inference.
class StageStore:
    """Immutable stage deltas and hash-verified recovery, not directory inference."""

    #   init  .
    def __init__(
        self,
        local: Path,
        persistent: Path,
        identity: dict,
        *,
        mount_root: Path | None = None,
        stages: tuple[str, ...] = STAGES,
        dependency_graph: dict[str, tuple[str, ...]] | None = None,
        optional_stages: tuple[str, ...] = (),
    ) -> None:
        self.hashes = DigestCache()
        self.stages = stages
        self.dependency_graph = dependency_graph
        self.optional_stages = optional_stages
        self.mount_root = mount_root
        self.check_mount()
        self.local = local
        self.persistent = persistent
        self.identity = identity
        self.signature = canonical_hash(identity)
        if not persistent.parent.is_dir():
            raise FileNotFoundError("Persistent run parent is unavailable")
        for root in (local, persistent):
            path = root / "identity.json"
            if path.exists():
                if read_json(path) != identity:
                    raise ValueError(
                        "Incompatible run: code/config/input/schema/cutoff/runtime "
                        "identity differs"
                    )
            else:
                # Create only the required owned directory; immutable inputs and
                # other runs remain untouched.
                root.mkdir(parents=True, exist_ok=True)
                atomic_json(path, identity)

    # Refuse writes into a disconnected mount's replacement directory.
    def check_mount(self) -> None:
        """Refuse writes into a disconnected mount's replacement directory."""
        if self.mount_root is not None and not os.path.ismount(self.mount_root):
            raise OSError("Persistence mount is unavailable")

    # Load a committed record and validate its self-digest and job identity.
    def record(self, stage: str, *, persistent: bool = False) -> dict | None:
        """Load a committed record and validate its self-digest and job identity."""
        if persistent:
            self.check_mount()
        root = self.persistent if persistent else self.local
        path = root / "stages" / stage / "complete.json"
        if not path.exists():
            return None
        record = read_json(path)
        payload = {
            key: value for key, value in record.items() if key != "manifest_sha256"
        }
        if (
            canonical_hash(payload) != record["manifest_sha256"]
            or record["run_signature"] != self.signature
        ):
            raise ValueError(f"Corrupt or incompatible stage manifest: {stage}")
        if record["stage"] != stage:
            raise ValueError("Stage identity mismatch")
        relative_path(record["attempt"])
        self.hashes.verify(
            root / "stages" / stage / record["attempt"] / "files", record["outputs"]
        )
        return record

    # Restore a verified stage from persistence after local runtime loss.
    def restore(self, stage: str, dependencies: dict[str, str]) -> dict | None:
        """Restore a verified stage from persistence after local runtime loss."""
        local = self.record(stage)
        remote = self.record(stage, persistent=True)
        record = local or remote
        if record is None:
            return None
        if record["dependencies"] != dependencies:
            raise ValueError(f"Upstream stage identity changed: {stage}")
        if local and remote and local != remote:
            raise ValueError("Local and persistent completion disagree")
        if local is None and remote:
            source = self.persistent / "stages" / stage / remote["attempt"] / "files"
            target = self.local / "stages" / stage / remote["attempt"] / "files"
            for name, entry in remote["outputs"].items():
                # Transfer through a temporary file and verify both ends before
                # publishing completion.
                verified_copy(
                    source / name,
                    safe_destination(target, name),
                    expected_sha256=entry["sha256"],
                )
            atomic_json(self.local / "stages" / stage / "complete.json", remote)
        if remote is None:
            self.persist(record)
        return record

    # Transfer only this stage's delta; remote completion is written last.
    def persist(self, record: dict) -> None:
        """Transfer only this stage's delta; remote completion is written last."""
        self.check_mount()
        if not self.persistent.parent.is_dir():
            raise FileNotFoundError("Persistence destination became unavailable")
        stage, attempt = record["stage"], record["attempt"]
        # Attribute parent-side file copies to their real stage, keeping live writes
        # local.
        with scope(
            self.local / "live_progress.jsonl",
            run_id=self.local.name,
            profile=self.identity.get("profile", {}).get("profile", "unknown"),
            stage=stage,
        ):
            # Count only bytes from this stage delta, not all upstream dependencies.
            byte_total = sum(v["bytes"] for v in record["outputs"].values())
            # Remote publication is a separate visible task from model fitting.
            emit(
                "persistence",
                0,
                byte_total,
                unit="bytes",
                status="started",
                stage=stage,
                force=True,
            )
            source = self.local / "stages" / stage / attempt / "files"
            target = self.persistent / "stages" / stage / attempt / "files"
            self.hashes.verify(source, record["outputs"])
            for name, entry in record["outputs"].items():
                # Transfer through a temporary file and verify both ends before
                # publishing completion.
                verified_copy(
                    source / name,
                    safe_destination(target, name),
                    expected_sha256=entry["sha256"],
                )
            self.hashes.verify(target, record["outputs"])
            atomic_json(self.persistent / "stages" / stage / "complete.json", record)
            # Completion follows verified destination files and the final atomic marker.
            emit(
                "persistence",
                byte_total,
                byte_total,
                unit="bytes",
                status="completed",
                stage=stage,
                force=True,
            )

    # Run a fresh attempt; failed files never become committed outputs.
    def execute(
        self,
        stage: str,
        dependencies: dict[str, str],
        view: dict[str, Path],
        worker: Callable[[Path], None],
        validator: Callable[[Path], None],
    ) -> dict:
        """Run a fresh attempt; failed files never become committed outputs."""
        attempt = uuid.uuid4().hex
        folder = self.local / "stages" / stage / attempt
        work = folder / "work"
        # Create only the required owned directory; immutable inputs and other runs
        # remain untouched.
        work.mkdir(parents=True, exist_ok=False)
        for name, source in view.items():
            destination = safe_destination(work, name)
            # Create only the required owned directory; immutable inputs and other
            # runs remain untouched.
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix in (".json", ".log"):
                shutil.copy2(source, destination)
            else:
                # Read-only dependency links avoid copying large immutable arrays
                # between local stages. Only new/changed regular files are published.
                if source.stat().st_mode & 0o222:
                    # Remove write permission from a linked dependency before
                    # invoking the stage worker.
                    source.chmod(source.stat().st_mode & ~0o222)
                # Expose an immutable upstream dependency without duplicating its
                # large array/table bytes.
                destination.symlink_to(source.resolve())
        # Retain the verified upstream content hashes before the worker runs, for
        # mutation detection afterward.
        before = {name: self.hashes.digest(source) for name, source in view.items()}
        atomic_json(
            folder / "attempt.json",
            {"status": "running", "stage": stage, "signature": self.signature},
        )
        # Protect this operation with explicit failure/cleanup handling so partial
        # work is never reported as complete.
        try:
            worker(work)
            validator(work)
            outputs = {}
            for path in work.rglob("*"):
                name = path.relative_to(work).as_posix()
                if not path.is_file() or path.is_symlink() or name.startswith("tmp/"):
                    # Skip this explicitly inapplicable/empty record while retaining
                    # the order of all remaining valid inputs.
                    continue
                sha = self.hashes.digest(path)
                if before.get(name) == sha:
                    # Skip this explicitly inapplicable/empty record while retaining
                    # the order of all remaining valid inputs.
                    continue
                entry = {"bytes": path.stat().st_size, "sha256": sha}
                # Transfer through a temporary file and verify both ends before
                # publishing completion.
                verified_copy(
                    path, safe_destination(folder / "files", name), expected_sha256=sha
                )
                outputs[name] = entry
            if not outputs:
                raise ValueError("Stage produced no validated outputs")
            # Dependencies must remain immutable even when a library receives a
            # symlink; detect writes rather than publishing corrupted lineage.
            if any(
                self.hashes.digest(path) != before[name] for name, path in view.items()
            ):
                raise ValueError("Worker mutated an upstream artifact")
            record = {
                "stage": stage,
                "attempt": attempt,
                "run_signature": self.signature,
                "dependencies": dependencies,
                "outputs": outputs,
                "validation": "passed",
                "local_complete": True,
            }
            record["manifest_sha256"] = canonical_hash(record)
            atomic_json(self.local / "stages" / stage / "complete.json", record)
            self.persist(record)
            atomic_json(
                folder / "attempt.json",
                {"status": "persisted_complete", "stage": stage},
            )
            return record
        # Handle the declared exception type here and preserve the surrounding
        # failure/recovery contract.
        except BaseException as error:
            # Persist safe failure metadata, never an incomplete success marker.
            # Emit only the failure type; detailed private traceback is in the stage
            # log.
            emit(
                "stage",
                0,
                1,
                unit="stage",
                status="failed",
                stage=stage,
                message=type(error).__name__,
                force=True,
            )
            # Record safe failure metadata without publishing a completed stage.
            failure: dict = {
                "status": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failure["resource_estimates"] = {
                path.name: read_json(path)
                for path in sorted((work / "capacity").glob("*.json"))
            }
            usage = work / "resource_usage.json"
            if usage.is_file():
                failure["resource_measurements"] = [
                    entry
                    for entry in read_json(usage).get("stages", [])
                    if entry.get("stage") == stage
                ]
            atomic_json(folder / "attempt.json", failure)
            mounted = self.mount_root is None or os.path.ismount(self.mount_root)
            if mounted and self.persistent.parent.is_dir():
                atomic_json(
                    self.persistent / "failures" / f"{stage}-{attempt}.json", failure
                )
            raise

    def view(self, records: list[dict]) -> dict[str, Path]:
        """Resolve the latest committed version of each relative artifact."""
        view: dict[str, Path] = {}
        for record in records:
            folder = (
                self.local / "stages" / record["stage"] / record["attempt"] / "files"
            )
            view.update({name: folder / name for name in record["outputs"]})
        return view

    # Distinguish local and persisted completion, with real digest checks.
    def status(self) -> dict:
        """Distinguish local and persisted completion, with real digest checks."""
        states = {}
        dependencies: dict[str, str] = {}
        for stage in self.stages:
            local = self.record(stage)
            remote = self.record(stage, persistent=True)
            if local and remote and local != remote:
                raise ValueError("Local and persistent completion disagree")
            record = local or remote
            if record is not None:
                if self.dependency_graph is not None and any(
                    name not in dependencies for name in self.dependency_graph[stage]
                ):
                    raise ValueError(
                        "Completed stage is missing an upstream dependency"
                    )
                expected = (
                    {name: dependencies[name] for name in self.dependency_graph[stage]}
                    if self.dependency_graph is not None
                    else dependencies
                )
                if record["dependencies"] != expected:
                    raise ValueError("Upstream stage identity changed")
                dependencies[stage] = record["manifest_sha256"]
            states[stage] = {
                "local_complete": local is not None,
                "persisted_complete": remote is not None,
            }
        return {
            "signature": self.signature,
            "stages": states,
            "complete": all(
                state["persisted_complete"]
                for name, state in states.items()
                if name not in self.optional_stages
            ),
        }
