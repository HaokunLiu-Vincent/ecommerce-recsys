"""Dependency-aware stage commits with independent local/persistent evidence."""

import contextlib
import json
import os
import shutil
import socket
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

from hm_recsys.colab.files import (
    atomic_json,
    canonical_hash,
    relative_path,
    safe_destination,
    verified_copy,
    verify_files,
)
from hm_recsys.contracts import digest, read_json

STAGES = (
    "preflight",
    "sample",
    "wals",
    "towers",
    "index_towers",
    "encode",
    "preprocessing",
    "features_F1",
    "features_F2",
    "features_F3",
    "features_F4",
    "features_Validation",
    "ranker",
    "reload_encode",
    "reload_features",
    "reload",
)


@contextlib.contextmanager
def writer_lease(root: Path, *, recover: bool = False) -> Iterator[None]:
    """Reject concurrent writers; stale remote leases require explicit recovery."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "writer.lock"
    if recover and lock.exists():
        lock.rename(root / f"writer.recovered-{uuid.uuid4().hex}.json")
    token = uuid.uuid4().hex
    try:
        with lock.open("x", encoding="utf-8") as handle:
            json.dump(
                {"token": token, "host": socket.gethostname(), "pid": os.getpid()},
                handle,
            )
    except FileExistsError as error:
        raise RuntimeError(
            "Run has another writer or stale lease; "
            "verify it stopped before --recover-lock"
        ) from error
    try:
        yield
    finally:
        if lock.exists() and read_json(lock).get("token") == token:
            lock.unlink()


class StageStore:
    """Immutable stage deltas and hash-verified recovery, not directory inference."""

    def __init__(
        self,
        local: Path,
        persistent: Path,
        identity: dict,
        *,
        mount_root: Path | None = None,
        stages: tuple[str, ...] = STAGES,
    ) -> None:
        self.stages = stages
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
                root.mkdir(parents=True, exist_ok=True)
                atomic_json(path, identity)

    def check_mount(self) -> None:
        """Refuse writes into a disconnected mount's replacement directory."""
        if self.mount_root is not None and not os.path.ismount(self.mount_root):
            raise OSError("Persistence mount is unavailable")

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
        verify_files(
            root / "stages" / stage / record["attempt"] / "files", record["outputs"]
        )
        return record

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
                verified_copy(
                    source / name,
                    safe_destination(target, name),
                    expected_sha256=entry["sha256"],
                )
            atomic_json(self.local / "stages" / stage / "complete.json", remote)
        if remote is None:
            self.persist(record)
        return record

    def persist(self, record: dict) -> None:
        """Transfer only this stage's delta; remote completion is written last."""
        self.check_mount()
        if not self.persistent.parent.is_dir():
            raise FileNotFoundError("Persistence destination became unavailable")
        stage, attempt = record["stage"], record["attempt"]
        source = self.local / "stages" / stage / attempt / "files"
        target = self.persistent / "stages" / stage / attempt / "files"
        verify_files(source, record["outputs"])
        for name, entry in record["outputs"].items():
            verified_copy(
                source / name,
                safe_destination(target, name),
                expected_sha256=entry["sha256"],
            )
        verify_files(target, record["outputs"])
        atomic_json(self.persistent / "stages" / stage / "complete.json", record)

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
        work.mkdir(parents=True, exist_ok=False)
        for name, source in view.items():
            destination = safe_destination(work, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix in (".json", ".log"):
                shutil.copy2(source, destination)
            else:
                # Read-only dependency links avoid copying large immutable arrays
                # between local stages. Only new/changed regular files are published.
                source.chmod(source.stat().st_mode & ~0o222)
                destination.symlink_to(source.resolve())
        before = {name: digest(source) for name, source in view.items()}
        atomic_json(
            folder / "attempt.json",
            {"status": "running", "stage": stage, "signature": self.signature},
        )
        try:
            worker(work)
            validator(work)
            outputs = {}
            for path in work.rglob("*"):
                name = path.relative_to(work).as_posix()
                if not path.is_file() or path.is_symlink() or name.startswith("tmp/"):
                    continue
                sha = digest(path)
                if before.get(name) == sha:
                    continue
                entry = {"bytes": path.stat().st_size, "sha256": sha}
                verified_copy(
                    path, safe_destination(folder / "files", name), expected_sha256=sha
                )
                outputs[name] = entry
            if not outputs:
                raise ValueError("Stage produced no validated outputs")
            # Dependencies must remain immutable even when a library receives a
            # symlink; detect writes rather than publishing corrupted lineage.
            if any(digest(path) != before[name] for name, path in view.items()):
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
        except BaseException as error:
            # Persist safe failure metadata, never an incomplete success marker.
            failure: dict = {
                "status": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
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
                if record["dependencies"] != dependencies:
                    raise ValueError("Upstream stage identity changed")
                dependencies[stage] = record["manifest_sha256"]
            states[stage] = {
                "local_complete": local is not None,
                "persisted_complete": remote is not None,
            }
        return {
            "signature": self.signature,
            "stages": states,
            "complete": all(s["persisted_complete"] for s in states.values()),
        }
