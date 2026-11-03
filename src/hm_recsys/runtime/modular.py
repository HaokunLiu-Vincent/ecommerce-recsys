"""Execute the dependency-checked, resumable single-source stage graph.

Inputs are immutable staged data, a resolved smoke/full profile, source identity
and explicit permissions. Each functional cell executes only its own numerical
phases in fresh workers and publishes a checksummed StageStore delta. Completed
prerequisites may be restored, never silently trained. Final Test is a separate
optional stage requiring two approvals and the frozen completed-run signature.
"""

from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

from hm_recsys.core.contracts import CHANNELS, Config, digest, read_json, split_manifest
from hm_recsys.core.progress import emit, scope
from hm_recsys.runtime.files import atomic_json, require_space
from hm_recsys.runtime.profiles import load_profile, resource_plan
from hm_recsys.runtime.resources import run_stage
from hm_recsys.runtime.state import StageStore, writer_lease

ORDER = ("prepare", *CHANNELS, "fusion", "features", "ranker", "validation", "reload")
ALL_STAGES = (*ORDER, "final_test")
DIRECT = {
    "prepare": (),
    **{c: ("prepare",) for c in CHANNELS},
    "fusion": CHANNELS,
    "features": ("fusion",),
    "ranker": ("features",),
    "validation": ("ranker",),
    "reload": ("validation",),
    "final_test": ("reload",),
}


def prerequisites(stage: str) -> tuple[str, ...]:
    """Return transitive ancestors in stable execution order, excluding the stage."""
    if stage not in DIRECT:
        raise ValueError(f"Unknown stage: {stage}")
    needed = set(DIRECT[stage])
    for item in tuple(needed):
        needed.update(prerequisites(item))
    return tuple(name for name in ALL_STAGES if name in needed)


GRAPH = {name: prerequisites(name) for name in ALL_STAGES}


def resolved_profile(
    project: Path,
    name: str,
    *,
    device: str | None = None,
    performance: str = "optimized",
) -> dict:
    """Resolve the existing compute budget without changing model hyperparameters."""
    from hm_recsys.runtime.settings import apply

    if name not in ("smoke", "full"):
        raise ValueError("Only smoke and full are active execution profiles")
    profile = apply(load_profile(project, name), project, performance)
    if device is not None:
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be explicitly cpu or cuda")
        profile["device"] = device
    return profile


def _safe_value(value: object) -> object:
    """Bound recursively nested diagnostics without publishing customer arrays."""
    if isinstance(value, dict):
        return safe_report(value)
    if isinstance(value, list):
        return [_safe_value(v) for v in value[:50]]
    return value


def safe_report(value: dict) -> dict:
    """Keep aggregate diagnostics, excluding customer-level record arrays."""
    private = {
        "users",
        "items",
        "global_vocabulary",
        "maps",
        "mapping",
        "groups",
        "query_order",
        "selected_users",
        "selected_customer_ids",
        "shards",
        "files",
        "losses",
    }
    return {key: _safe_value(item) for key, item in value.items() if key not in private}


def stage_summary(local: Path, stage: str) -> dict:
    """Load the aggregate report copied from a verified completed stage."""
    return read_json(local / "reports" / f"{stage}.json")


class Session:
    """Own one source-verified stage session and its process-scoped digest cache."""

    def __init__(
        self,
        project: Path,
        data_root: Path,
        local: Path,
        persistent: Path,
        *,
        profile: dict,
        mount_root: Path | None = None,
        recover_lock: bool = False,
        git_source: dict | None = None,
        synthetic_fixture: bool = False,
    ) -> None:
        from hm_recsys.runtime.runner import execution_identity

        self.project, self.data_root = project, data_root
        self.local, self.persistent, self.profile = local, persistent, profile
        self.mount_root, self.recover_lock = mount_root, recover_lock
        self.synthetic_fixture = synthetic_fixture
        if synthetic_fixture:
            from hm_recsys.runtime.testing import assert_fixture

            assert_fixture(data_root, full=profile["profile"] == "full")
        if (
            local.resolve().is_relative_to(persistent.resolve())
            or data_root.resolve() == persistent.resolve()
        ):
            raise ValueError("Training and persistent roots must be distinct")
        self.identity = execution_identity(
            project, profile, data_root, git_source=git_source
        )
        self.context = ExitStack()

    def __enter__(self):
        try:
            self.context.enter_context(
                writer_lease(self.persistent, recover=self.recover_lock)
            )
            self.context.enter_context(
                writer_lease(self.local, recover=self.recover_lock)
            )
            self.store = StageStore(
                self.local,
                self.persistent,
                self.identity,
                mount_root=self.mount_root,
                stages=ALL_STAGES,
                dependency_graph=GRAPH,
                optional_stages=("final_test",),
            )
            current = resource_plan(self.profile)
            old = self.persistent / "resources.resolved.json"
            self.resources = read_json(old) if old.exists() else current
            self.resources["rss_abort_gib"] = min(
                current["rss_abort_gib"], self.resources["rss_abort_gib"]
            )
            self.resources["available_floor_gib"] = max(
                current["available_floor_gib"], self.resources["available_floor_gib"]
            )
            self.resources["rss_target_gib"] = self.resources["rss_abort_gib"] * 0.8
            atomic_json(self.local / "resources.resolved.json", self.resources)
            if not old.exists():
                atomic_json(old, self.resources)
            self.policy = Config({**self.profile["compute"], **self.resources})
            return self
        except BaseException:
            self.context.close()
            raise

    def __exit__(self, *args):
        return self.context.__exit__(*args)

    def _records(self, stage: str) -> list[dict]:
        records = []
        hashes = {}
        for name in GRAPH[stage]:
            expected = {k: hashes[k] for k in GRAPH[name]}
            record = self.store.restore(name, expected)
            if record is None:
                raise ValueError(
                    f"Missing prerequisite {name} for {stage}; "
                    "run its dedicated cell first"
                )
            records.append(record)
            hashes[name] = record["manifest_sha256"]
        return records

    def run(
        self,
        stage: str,
        *,
        action: str = "run",
        run_final_test: bool = False,
        approve_final_test: bool = False,
        locked_signature: str | None = None,
    ) -> dict:
        if action not in ("run", "resume", "status") or stage not in ALL_STAGES:
            raise ValueError("Invalid stage/action")
        if stage == "final_test" and action != "status":
            if (
                self.profile["profile"] != "full"
                or not run_final_test
                or not approve_final_test
            ):
                raise PermissionError(
                    "Final Test requires full profile and two explicit approvals"
                )
            if locked_signature != self.store.signature:
                raise ValueError(
                    "Final Test requires the original frozen run signature"
                )
        with scope(
            self.local / "live_progress.jsonl",
            run_id=self.local.name,
            profile=self.profile["profile"],
            stage=stage,
        ):
            records = self._records(stage)
            dependencies = {d["stage"]: d["manifest_sha256"] for d in records}
            previous = self.store.restore(stage, dependencies)
            if action == "status":
                return {
                    "stage": stage,
                    "status": "completed" if previous else "not_run",
                    "signature": self.store.signature,
                }
            reused = previous is not None
            if previous is None:
                if stage == "final_test":
                    self._publish_index()
                    lock = self.local / "validation_lock.json"
                    if (
                        not lock.exists()
                        or read_json(lock)["signature"] != self.store.signature
                    ):
                        raise ValueError(
                            "Final Test requires completed Validation and reload lock"
                        )
                    bound = read_json(lock)
                    verified = self._view()
                    if bound["model_sha256"] != digest(
                        verified["models/lambdamart.txt"]
                    ) or bound["schema_sha256"] != digest(
                        verified["feature_schema.json"]
                    ):
                        raise ValueError("Final Test frozen model/schema mismatch")
                view = self.store.view(records)
                emit("stage", 0, 1, unit="stage", status="started", force=True)

                def worker(work: Path) -> None:
                    self._execute(stage, work, final_test_active=stage == "final_test")

                from hm_recsys.runtime.stages import validate

                previous = self.store.execute(
                    stage,
                    dependencies,
                    view,
                    worker,
                    lambda work: validate(stage, work, self.profile),
                )
            base = self.local / "stages" / stage / previous["attempt"] / "files"
            report = read_json(base / "reports" / f"{stage}.json")
            result = {**report, "reused": reused, "signature": self.store.signature}
            atomic_json(self.local / "reports" / f"{stage}.json", result)
            records.append(previous)
            self._publish_index()
            emit(
                "stage",
                1,
                1,
                unit="stage",
                status="restored" if reused else "completed",
                force=True,
            )
            if stage == "reload":
                lock = {
                    "signature": self.store.signature,
                    "model_sha256": digest(self._view()["models/lambdamart.txt"]),
                    "schema_sha256": digest(self._view()["feature_schema.json"]),
                }
                atomic_json(self.local / "validation_lock.json", lock)
                atomic_json(self.persistent / "validation_lock.json", lock)
            return result

    def _view(self) -> dict[str, Path]:
        records = [
            r for name in ALL_STAGES if (r := self.store.record(name)) is not None
        ]
        return self.store.view(records)

    def _publish_index(self) -> None:
        view = self._view()
        locations = {
            name: str(path.relative_to(self.local)) for name, path in view.items()
        }
        status = self.store.status()
        for root in (self.local, self.persistent):
            atomic_json(root / "artifacts.json", locations)
            atomic_json(root / "status.json", status)
        result_name = f"{self.profile['profile']}_results.json"
        if result_name in view and "reload_results.json" in view:
            summary = read_json(view[result_name])
            summary.update(
                status="passed",
                run_signature=self.store.signature,
                reload=read_json(view["reload_results.json"]),
                sample=read_json(view["population_manifest.json"]),
                wals=read_json(view["models/wals.json"]),
                twotower={
                    k: read_json(view["models/twotower.json"])[k]
                    for k in ("steps", "epochs", "processed_targets")
                },
            )
            for root in (self.local, self.persistent):
                atomic_json(root / "run_summary.json", summary)
        lock = self.persistent / "validation_lock.json"
        if lock.exists() and not (self.local / "validation_lock.json").exists():
            atomic_json(self.local / "validation_lock.json", read_json(lock))

    def _execute(self, stage: str, work: Path, *, final_test_active: bool) -> None:
        require_space(work, self.resources["disk_reserve_bytes"])
        for directory in ("tmp", "models", "indexes", "predictions"):
            (work / directory).mkdir(parents=True, exist_ok=True)
        atomic_json(work / "profile.resolved.json", self.profile)
        atomic_json(work / "config.resolved.json", self.profile["compute"])
        atomic_json(
            work / "run_manifest.json",
            {
                "source_root": str(self.data_root.resolve()),
                "profile": self.profile["profile"],
                "is_smoke": self.profile["profile"] == "smoke",
                "is_benchmark": False,
                "portable_inputs": True,
                "migration_signature": self.store.signature,
                "code_sha256": self.identity["code"],
                "synthetic_fixture": self.synthetic_fixture,
            },
        )
        (self.persistent / "checkpoints/towers").mkdir(parents=True, exist_ok=True)
        (self.local / "checkpoints/towers").mkdir(parents=True, exist_ok=True)
        options = {
            "data_root": str(self.data_root.resolve()),
            "synthetic_fixture": self.synthetic_fixture,
            "persistent_checkpoint": str(
                (self.persistent / "checkpoints/towers").resolve()
            ),
            "local_checkpoint": str((self.local / "checkpoints/towers").resolve()),
            "checkpoint_every_steps": self.profile["checkpoint_every_steps"],
            "mount_root": str(self.mount_root) if self.mount_root else None,
            "final_test_active": final_test_active,
            "reloaded": False,
        }
        start = perf_counter()

        def phase(name: str, *, reloaded: bool = False) -> None:
            atomic_json(work / "colab_stage.json", {**options, "reloaded": reloaded})
            prefix = "test_" if final_test_active else "reload_" if reloaded else ""
            scoped = (
                prefix + name
                if prefix and name.startswith(("channel_", "fusion_", "features_"))
                else name
            )
            run_stage(
                scoped,
                work,
                self.policy,
                module="hm_recsys.runtime.worker",
                cwd=self.project,
            )

        report: dict
        if stage == "prepare":
            for name in ("preflight", "canonical", "content", "rosters", "capacity"):
                phase(name)
            for name in self.profile["folds"]:
                phase("statistics_" + name)
            report = {
                "splits": split_manifest(),
                "population": safe_report(read_json(work / "population_manifest.json")),
                "rosters": {
                    name: read_json(work / "queries" / f"{name}.json")
                    for name in self.profile["folds"]
                },
                "test": {"status": "metadata_only", "labels_read": False},
                "resources": self.resources,
            }
        elif stage in CHANNELS:
            if stage == "itemcf":
                for name in self.profile["folds"]:
                    phase("itemcf_" + name)
            elif stage == "wals":
                phase("wals_fit")
            elif stage == "twotower":
                for name in (
                    "device_probe",
                    "prepare_towers",
                    "towers_fit",
                    "index_towers",
                    "encode",
                ):
                    phase(name)
            phase("channel_" + stage)
            report = {
                "partitions": read_json(
                    work / "phase_reports" / f"channel_{stage}.json"
                )
            }
            if stage == "wals":
                report["model"] = safe_report(read_json(work / "models/wals.json"))
            if stage == "twotower":
                meta = read_json(work / "models/twotower.json")
                report["model"] = safe_report(meta)
                report["last_loss"] = meta["losses"][-1] if meta["losses"] else None
                report["performance"] = read_json(work / "tower_performance.json")
        elif stage == "fusion":
            phase("fusion_build")
            report = {"partitions": read_json(work / "phase_reports/fusion_build.json")}
        elif stage == "features":
            phase("preprocessing")
            for name in self.profile["folds"]:
                phase("features_" + name)
            report = {
                "schema": {"names": read_json(work / "feature_schema.json")["names"]},
                "partitions": {
                    name: safe_report(
                        read_json(work / f"features/{name}/manifest.json")
                    )
                    for name in self.profile["folds"]
                },
            }
        elif stage == "ranker":
            phase("ranker_fit")
            report = safe_report(
                read_json(work / f"{self.profile['profile']}_results.json")
            )
        elif stage == "validation":
            result = read_json(work / f"{self.profile['profile']}_results.json")
            report = {
                "native_metrics": result["validation"],
                "ranker": safe_report(result["ranker"]),
                "baselines": read_json(work / "fusion/Validation/manifest.json")[
                    "rrf_baseline"
                ],
                "retrieval": {
                    c: read_json(work / f"channels/Validation/{c}.json")
                    for c in CHANNELS
                },
                "test": "not_read_or_evaluated",
            }
        elif stage == "reload":
            phase("reload_encode")
            for channel in CHANNELS:
                phase("channel_" + channel, reloaded=True)
            phase("fusion_build", reloaded=True)
            phase("features_Validation", reloaded=True)
            phase("reload_score")
            report = read_json(work / "reload_results.json")
        elif stage == "final_test":
            atomic_json(
                work / "final_test_authorization.json",
                {
                    "approved": True,
                    "signature": self.store.signature,
                    "model_sha256": digest(work / "models/lambdamart.txt"),
                    "schema_sha256": digest(work / "feature_schema.json"),
                },
            )
            phase("test_roster")
            phase("statistics_Test")
            phase("itemcf_Test")
            phase("encode_test")
            for channel in CHANNELS:
                phase("channel_" + channel)
            phase("fusion_build")
            phase("features_Test")
            phase("test_score")
            report = read_json(work / "final_test_results.json")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        final = {
            **safe_report(report),
            "stage": stage,
            "status": "completed",
            "seconds": perf_counter() - start,
            "phases": safe_report(read_json(work / "resource_usage.json")),
        }
        atomic_json(work / "reports" / f"{stage}.json", final)


def run_all(
    project: Path,
    data_root: Path,
    local: Path,
    persistent: Path,
    *,
    profile: dict,
    until: str = "reload",
    mount_root: Path | None = None,
    recover_lock: bool = False,
    git_source: dict | None = None,
    synthetic_fixture: bool = False,
) -> dict:
    """Execute each required canonical stage once; final Test is never included."""
    if until not in ORDER:
        raise ValueError("Unknown default-run stage; Test is separately gated")
    with Session(
        project,
        data_root,
        local,
        persistent,
        profile=profile,
        mount_root=mount_root,
        recover_lock=recover_lock,
        git_source=git_source,
        synthetic_fixture=synthetic_fixture,
    ) as session:
        for stage in ORDER[: ORDER.index(until) + 1]:
            session.run(stage)
        return session.store.status()


def run_one(
    project: Path,
    data_root: Path,
    local: Path,
    persistent: Path,
    *,
    profile: dict,
    stage: str,
    action: str = "run",
    mount_root: Path | None = None,
    recover_lock: bool = False,
    git_source: dict | None = None,
    synthetic_fixture: bool = False,
    run_final_test: bool = False,
    approve_final_test: bool = False,
    locked_signature: str | None = None,
) -> dict:
    """Run only one requested cell after explicit prerequisite and Test checks."""
    if (
        stage == "final_test"
        and action != "status"
        and (
            not run_final_test or not approve_final_test or profile["profile"] != "full"
        )
    ):
        raise PermissionError("Final Test requires full and two separate approvals")
    with Session(
        project,
        data_root,
        local,
        persistent,
        profile=profile,
        mount_root=mount_root,
        recover_lock=recover_lock,
        git_source=git_source,
        synthetic_fixture=synthetic_fixture,
    ) as session:
        return session.run(
            stage,
            action=action,
            run_final_test=run_final_test,
            approve_final_test=approve_final_test,
            locked_signature=locked_signature,
        )
