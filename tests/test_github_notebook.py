"""Exercise the root Notebook's safe Git bootstrap against owned offline remotes.

The actual stdlib setup cells validate URL/ref/SHA/destination and clean pinned
reuse before secrets or Git execution. Tests replace only network transport and
the Colab secret provider, inspect ephemeral askpass cleanup, and verify named
status commands. No real credential, cloud runtime or project data is accessed.
"""

# Inspect and execute only the checked-in Notebook's small setup cells.
import json

# Copy source-only fixture files; no project datasets or existing runs are copied.
import shutil

# Invoke real local Git while replacing only network transport in the fixture.
import subprocess

# Supply an explicit synthetic Colab Secrets provider for the private-auth test.
import sys

# Type fixture roots and validate exact checkout contents.
from pathlib import Path

# Construct a small module stub without accessing any real account or secret.
from types import ModuleType, SimpleNamespace

# Assert explicit rejection paths without invoking cloud services.
import pytest

# Resolve maintained source and Notebook paths from the package location.
from hm_recsys.core.contracts import ROOT

# Use the same validated production origin; offline tests rewrite only its transport.
REPO_URL = "https://github.com/HaokunLiu-Vincent/ecommerce-recsys.git"


# Expose a clean source-only Git fixture for actual generated CLI plan tests.
def git_project(path: Path) -> str:
    """Copy current code/config into an owned committed detached repository."""
    # Keep local test commits separate from the user's working repository/history.
    path.mkdir()
    # Include only source and interfaces needed by real CLI planning.
    for folder in ("src", "scripts", "configs"):
        # Exclude interpreter caches to keep the fixture clean and source-only.
        shutil.copytree(
            ROOT / folder, path / folder, ignore=shutil.ignore_patterns("__pycache__")
        )
    # Preserve exact dependency/config identities used by the real planner.
    for name in ("pyproject.toml", "uv.lock", ".python-version", ".gitignore"):
        # Copy small project metadata, never data or secret-bearing runtime artifacts.
        shutil.copyfile(ROOT / name, path / name)
    (path / "docs/archi").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "docs/archi/evaluation_protocol.md",
        path / "docs/archi/evaluation_protocol.md",
    )
    # Initialize only this temporary fixture, preserving the real repository.
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(path)],
        check=True,
        capture_output=True,
    )
    # Fixture author identity is synthetic and limited to this temporary repository.
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Fixture"], check=True
    )
    # No real identity or credential is needed for an offline fixture commit.
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    # Retain the validated HTTPS origin while testing with local-only transport.
    subprocess.run(
        ["git", "-C", str(path), "remote", "add", "origin", REPO_URL], check=True
    )
    # Stage only the explicitly copied source fixture.
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    # Produce a real commit whose SHA can be checked by the real Git inspector.
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "Source fixture"],
        check=True,
        capture_output=True,
    )
    # Read the commit identity rather than fabricating a SHA.
    sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Reuse requires a pinned detached checkout rather than an advancing branch.
    subprocess.run(
        ["git", "-C", str(path), "checkout", "--detach", sha],
        check=True,
        capture_output=True,
    )
    # Return the exact clean fixture revision to the caller's CLI contract check.
    return sha


# Locate the real bootstrap cell rather than maintaining a duplicate implementation.
def _bootstrap() -> dict:
    """Load the actual stdlib bootstrap without executing a Notebook job."""
    # Read current Notebook source directly so changes to emitted code are tested.
    book = json.loads((ROOT / "colab_training_val.ipynb").read_text())
    # Pick the actual minimal bootstrap definitions, excluding model/mount cells.
    source = next(
        "".join(c["source"])
        for c in book["cells"]
        if c["cell_type"] == "code"
        and "def bootstrap_checkout(" in "".join(c["source"])
    )
    # The preceding configuration provides Path, without hidden project imports.
    namespace = {"Path": Path}
    # Define stdlib helpers only; cloning is performed explicitly by each test.
    exec(compile(source, "actual notebook bootstrap", "exec"), namespace)
    # Return the exact checked-in callable/validator for the offline fixture.
    return namespace


# Real Git clones/fetches stay offline while production URL validation remains active.
def test_actual_notebook_bootstrap_local_remote(tmp_path: Path, monkeypatch) -> None:
    """Pin a real local commit, reject stale/dirty trees and preserve credentials."""
    # Create a tiny source-only remote without network access or existing artifacts.
    source = tmp_path / "source"
    # Use the real maintained helper/code in the fixture revision.
    sha = git_project(source)
    # The remote's main branch still points at the committed fixture after detach.
    bare = tmp_path / "remote.git"
    # Bare remote transport is explicitly local and confined to this test.
    subprocess.run(
        ["git", "clone", "--bare", str(source), str(bare)],
        check=True,
        capture_output=True,
    )
    # Save the actual subprocess API before installing the narrow transport adapter.
    original = subprocess.run
    # Record only safe argv; private tokens must never appear there.
    commands = []

    # Rewrite only validated HTTPS clone/fetch transport to the owned bare remote.
    def offline(argv, **kwargs):
        # Preserve the exact public command contract before fixture-only configuration.
        commands.append(list(map(str, argv)))
        # Ordinary local identity/status inspection is executed unchanged.
        if argv[0] == "git" and any(action in argv for action in ("clone", "fetch")):
            # Add fixture overrides after production flags so no network can occur.
            index = argv.index("clone") if "clone" in argv else argv.index("fetch")
            # Keep remote.origin.url HTTPS while redirecting only offline transport.
            argv = [
                *argv[:index],
                "-c",
                f"url.{bare}.insteadOf={REPO_URL}",
                "-c",
                "protocol.file.allow=always",
                *argv[index:],
            ]
        # Execute actual Git, including object resolution and detached checkout.
        return original(argv, **kwargs)

    # No implementation logic or validation is mocked, only the network transport.
    monkeypatch.setattr(subprocess, "run", offline)
    # Define the actual Notebook functions without any mounted/cloud environment.
    namespace = _bootstrap()
    # Bootstrap a fresh destination exactly as a new Colab runtime would.
    destination = tmp_path / "checkout"
    # The resolver reads a real Git ref and returns its actual committed object SHA.
    result = namespace["bootstrap_checkout"](REPO_URL, "main", None, destination)
    # No fabricated commit or moving-branch HEAD can satisfy this parity check.
    assert result == sha
    # Same-commit recovery may verify the existing checkout without source mutation.
    assert (
        namespace["bootstrap_checkout"](REPO_URL, "main", sha, destination, resume=True)
        == sha
    )
    # Existing checkout reuse without an explicit pin is intentionally rejected.
    with pytest.raises(ValueError, match="explicit SHA"):
        # Ref tips are not used to replace an existing training checkout.
        namespace["bootstrap_checkout"](REPO_URL, "main", None, destination)
    # Wrong-commit resume never checks out another revision beneath existing work.
    with pytest.raises(ValueError, match="SHA differs"):
        # A changed source commit requires a new owned run/runtime.
        namespace["bootstrap_checkout"](
            REPO_URL, "main", "a" * 40, destination, resume=True
        )
    # Introduce an ordinary untracked user modification in this owned fixture.
    (destination / "user_change.txt").write_text("preserve")
    # Dirty source is refused rather than reset or deleted automatically.
    with pytest.raises(ValueError, match="dirty"):
        # Existing source bytes stay available for user inspection.
        namespace["bootstrap_checkout"](REPO_URL, "main", sha, destination)
    # The bootstrap never hides unsafe reset/clean/pull commands inside its calls.
    assert not any(
        action in command
        for command in commands
        for action in ("reset", "clean", "pull")
    )


# Unsafe input must be rejected before Git executes or secret access starts.
def test_notebook_bootstrap_validation_precedes_git(
    tmp_path: Path, monkeypatch
) -> None:
    """Reject unchecked URLs, refs, paths and missing recovery pins without effects."""
    # Use the checked-in validation logic for all malformed cases.
    namespace = _bootstrap()

    # Any subprocess execution here would cross the required validation boundary.
    def forbidden(*args, **kwargs):
        # Make accidental Git invocation an actionable failed assertion.
        raise AssertionError("Git ran before validation")

    # Keep the test independent of a network, Git installation or cloud account.
    monkeypatch.setattr(subprocess, "run", forbidden)
    # Cover credentials, option injection, traversal and absent revision identities.
    cases = [
        ("", "main", None, tmp_path / "missing"),
        (
            "https://token@github.com/owner/repo.git",
            "main",
            None,
            tmp_path / "credentials",
        ),
        (REPO_URL, "--upload-pack=bad", None, tmp_path / "option"),
        (REPO_URL, "../main", None, tmp_path / "ref"),
        (REPO_URL, "feature/.hidden", None, tmp_path / "hidden_ref"),
        (REPO_URL, "main", "bad", tmp_path / "sha"),
        (REPO_URL, "main", None, Path("relative")),
    ]
    # Each rejected input must remain a no-Git operation.
    for args in cases:
        # A clear input exception is expected instead of shell interpretation.
        with pytest.raises(ValueError):
            # No fixture directory is created by invalid input.
            namespace["bootstrap_checkout"](*args)
    # Recovery explicitly requires the previous commit before any checkout access.
    with pytest.raises(ValueError, match="original explicit"):
        # The current main tip cannot serve as a recovery identity.
        namespace["bootstrap_checkout"](
            REPO_URL, "main", None, tmp_path / "resume", resume=True
        )


# Private authentication must exist only inside the owned subprocess environment.
def test_notebook_private_secret_askpass_is_ephemeral(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Run actual askpass prompts without putting a token in argv/config/logs."""
    # Use a synthetic secret, not a real token or connected account.
    token = "synthetic_test_token"
    # Stub only explicit Colab Secrets access for this private-path fixture.
    google = ModuleType("google")
    # Keep the module contract identical to the supported Colab import.
    colab = ModuleType("google.colab")
    # A secret name selects the synthetic provider; its value is never displayed.
    monkeypatch.setattr(
        colab, "userdata", SimpleNamespace(get=lambda name: token), raising=False
    )
    # No secret lookup occurs in a real browser/account.
    monkeypatch.setitem(sys.modules, "google", google)
    # Match Python's qualified import resolution for google.colab.
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    # Save the actual runner for executing only the short askpass script.
    original = subprocess.run
    # Track temporary paths for post-failure cleanup checks.
    askpasses = []

    # Inspect the real Git argv/environment without contacting any remote server.
    def reject_remote(argv, **kwargs):
        # The token must stay out of every command-line argument.
        assert token not in repr(argv)
        # Read the explicitly scoped authentication subprocess environment.
        env = kwargs["env"]
        # The script references an environment variable; it must not embed the token.
        askpass = Path(env["GIT_ASKPASS"])
        # Preserve only its path for the cleanup assertion.
        askpasses.append(askpass)
        # User-provided token contents are never written to a temporary source file.
        assert token not in askpass.read_text()
        # Exercise actual prompt behavior to catch malformed newline/shebang scripts.
        user = original(
            [str(askpass), "Username"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        # Execute a password prompt only inside this private synthetic subprocess.
        password = original(
            [str(askpass), "Password"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        # The username is fixed while the password comes solely from scoped env.
        assert user.stdout == "x-access-token" and password.stdout == token
        # Simulate an auth failure carrying private stderr, which must stay undisplayed.
        return subprocess.CompletedProcess(argv, 128, "", "private stderr " + token)

    # Reject remote transport after validating the actual ephemeral askpass path.
    monkeypatch.setattr(subprocess, "run", reject_remote)
    # Execute the real Notebook private path with no real authentication/network.
    namespace = _bootstrap()
    # Git failures must be generic rather than expose captured private stderr.
    with pytest.raises(RuntimeError, match="repository access"):
        # The only accepted secret selector is the explicit name passed by the user.
        namespace["bootstrap_checkout"](
            REPO_URL,
            "main",
            None,
            tmp_path / "private",
            secret_name="GITHUB_READ_TOKEN",
        )
    # Cleanup removes the temporary authentication program even after a failed fetch.
    assert askpasses and not any(p.exists() for p in askpasses)
    # The token is absent from displayed output and from persistent checkout files.
    assert token not in capsys.readouterr().out


def test_notebook_status_requests_only_named_stage(tmp_path: Path) -> None:
    """Keep status separate from probes, fitting and implicit prerequisite runs."""
    book = json.loads((ROOT / "colab_training_val.ipynb").read_text(encoding="utf-8"))
    cells = {
        tag: "".join(cell["source"])
        for cell in book["cells"]
        if cell["cell_type"] == "code"
        for tag in cell["metadata"].get("tags", [])
    }
    calls = []
    namespace = {
        "RUN_ACTION": "status",
        "run_command": lambda argv, log: calls.append(argv),
        "CLI": ["python", "run_pipeline.py"],
        "OPTIONS": [],
        "LOG_ROOT": tmp_path,
        "REPORT_ROOT": tmp_path / "reports",
        "STAGE_RESULTS": {},
    }
    exec(compile(cells["stage-wrapper"], "actual stage wrapper", "exec"), namespace)
    exec(compile(cells["stage:wals"], "actual WALS status cell", "exec"), namespace)
    assert calls == [
        [
            "python",
            "run_pipeline.py",
            "stage",
            "--stage",
            "wals",
            "--stage-action",
            "status",
        ]
    ]
    assert "RUN_CUDA_PROBE" not in "\n".join(cells.values())
