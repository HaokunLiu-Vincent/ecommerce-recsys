"""Offline Git bootstrap, immutable resume and credential-boundary tests."""

# Configure subprocess fixtures without touching the user's Git settings.
import os

# Exercise the actual standalone script with the current project interpreter.
import subprocess

# Spawn the standalone helper with the exact interpreter running the tests.
import sys

# Use only pytest-owned paths for repositories, commits and checkouts.
from pathlib import Path

# Assert precise failures instead of silently repairing source repositories.
import pytest

# Locate the standalone script without depending on the process working directory.
from hm_recsys.core.contracts import ROOT

# Reuse the same stdlib API called after the Notebook's pinned clone bootstrap.
from hm_recsys.runtime.git_checkout import (
    checkout,
    development_identity,
    git_environment,
    inspect_checkout,
    repository_url,
    validate_commit,
    validate_ref,
)


# Run real local Git commands exclusively inside a pytest fixture repository.
def git(directory: Path, *args: str) -> str:
    """Return actual Git stdout while keeping all fixture identities synthetic."""
    # Explicit author metadata prevents accidental use of the user's identity.
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Fixture",
        GIT_COMMITTER_NAME="Fixture",
        GIT_AUTHOR_EMAIL="fixture@example.invalid",
        GIT_COMMITTER_EMAIL="fixture@example.invalid",
    )
    # Argv and cwd prevent any fixture command from addressing the real project.
    result = subprocess.run(
        ["git", *args],
        cwd=directory,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    # The fixture uses only tiny known source files, never user data or network.
    return result.stdout.strip()


# Create a mutable seed and a real temporary bare remote for clone/fetch tests.
@pytest.fixture
def local_remote(tmp_path: Path) -> tuple[Path, Path, str]:
    """Build a synthetic main branch and publish it only to an owned local bare repo."""
    # The two directories are independent of the shared working checkout.
    seed, remote = tmp_path / "seed", tmp_path / "remote.git"
    # Establish an ordinary seed worktree with a known branch.
    seed.mkdir()
    # Git initialization is authorized here only for temporary test repositories.
    git(seed, "init", "--initial-branch=main")
    # One small source file suffices to verify commit/checkpoint identity behavior.
    (seed / "README.md").write_text("first fixture commit\n", encoding="utf-8")
    # Stage only the explicitly authored synthetic file.
    git(seed, "add", "README.md")
    # Create the first source revision in the isolated seed.
    git(seed, "commit", "-m", "first fixture")
    # A bare local remote exercises actual Git transports without a GitHub service.
    git(tmp_path, "init", "--bare", str(remote))
    # This push never leaves tmp_path and changes no real remote repository.
    git(seed, "push", str(remote), "main")
    # Record the full immutable commit that the checkout must resolve.
    first = git(seed, "rev-parse", "HEAD")
    # Keep the seed available for a later branch-tip update.
    return seed, remote, first


# Real clones/fetches must pin code and keep older runtime trees untouched.
def test_local_clone_new_commit_and_exact_resume(local_remote, tmp_path: Path) -> None:
    """Resolve main, detach, preserve original-SHA resume and use a fresh new run."""
    # All remote operations in this test stay within its synthetic bare repository.
    seed, remote, first = local_remote
    # The initial run resolves the explicitly configured branch once.
    project = tmp_path / "runtime"
    # Local paths require the test-only opt-in; production accepts HTTPS GitHub only.
    identity = checkout(str(remote), "main", None, project, allow_local_remote=True)
    # Actual HEAD must equal the resolved full SHA and be detached/clean.
    assert identity == {
        "kind": "git",
        "remote_url": str(remote.resolve()),
        "requested_ref": "main",
        "resolved_commit": first,
        "clean": True,
    }
    # Resume validation must not mutate the original checkout.
    assert (
        inspect_checkout(
            project,
            repo_url=str(remote),
            ref="main",
            commit=first,
            allow_local_remote=True,
        )
        == identity
    )
    # Publish a second real fixture commit only to the temporary local remote.
    (seed / "README.md").write_text("second fixture commit\n", encoding="utf-8")
    # Keep the second revision's staged population explicit.
    git(seed, "add", "README.md")
    # A new commit simulates main moving while an earlier run remains resumable.
    git(seed, "commit", "-m", "second fixture")
    # No external network or user repository is contacted.
    git(seed, "push", str(remote), "main")
    # Preserve the distinct second immutable identity.
    second = git(seed, "rev-parse", "HEAD")
    # Recovery uses first even though main now points at second.
    assert (
        checkout(
            str(remote), "main", first, project, resume=True, allow_local_remote=True
        )
        == identity
    )
    # A different commit cannot rewrite an already existing runtime's source.
    with pytest.raises(ValueError, match="commit differs"):
        # Refuse source mutation rather than reset/pull the old runtime.
        checkout(str(remote), "main", second, project, allow_local_remote=True)
    # The rejected operation must retain the old source bytes.
    assert (project / "README.md").read_text(
        encoding="utf-8"
    ) == "first fixture commit\n"
    # A new destination legitimately selects the new branch tip for a new run.
    newer = checkout(
        str(remote), "main", None, tmp_path / "new_runtime", allow_local_remote=True
    )
    # This is a new source identity, not a renamed old checkpoint/run.
    assert newer["resolved_commit"] == second != first


# Existing dirty/mismatched trees must be preserved on every rejection path.
def test_existing_checkout_refusals(local_remote, tmp_path: Path) -> None:
    """Reject edits, different origins, attached branches and missing resume pins."""
    # Construct a known detached checkout through the actual production API.
    _, remote, first = local_remote
    # Use an owned source tree rather than the real project.
    project = tmp_path / "runtime"
    # Enable filesystem transport explicitly for this offline test only.
    checkout(str(remote), "main", first, project, allow_local_remote=True)
    # An unpinned resume must fail before any branch lookup or source mutation.
    with pytest.raises(ValueError, match="original explicit"):
        # Never recover an experiment using today's branch tip.
        checkout(
            str(remote), "main", None, project, resume=True, allow_local_remote=True
        )
    # Keep the edit after failure so the operator can inspect/preserve it.
    (project / "README.md").write_text("user fixture edit\n", encoding="utf-8")
    # An existing dirty tree cannot be silently cleaned by bootstrap.
    with pytest.raises(ValueError, match="dirty"):
        # This check includes tracked edits as well as nonignored untracked files.
        checkout(str(remote), "main", first, project, allow_local_remote=True)
    # The caller's bytes are untouched by the refusal.
    assert (project / "README.md").read_text(encoding="utf-8") == "user fixture edit\n"
    # Restore only our synthetic fixture bytes explicitly inside the test.
    (project / "README.md").write_text("first fixture commit\n", encoding="utf-8")
    # A different existing remote directory is not automatically substituted.
    other = tmp_path / "other.git"
    # This second bare repository remains entirely inside the fixture.
    git(tmp_path, "init", "--bare", str(other))
    # Changing origin is an explicit test action, never performed by the helper.
    git(project, "remote", "set-url", "origin", str(other))
    # The requested repository and actual origin must agree.
    with pytest.raises(ValueError, match="different origin"):
        # Reject before any fetch or checkout.
        inspect_checkout(
            project,
            repo_url=str(remote),
            ref="main",
            commit=first,
            allow_local_remote=True,
        )
    # Restore the fixture origin to isolate the detached-state check.
    git(project, "remote", "set-url", "origin", str(remote))
    # A branch-bound worktree is not the pinned detached runtime contract.
    git(project, "checkout", "-b", "fixture_branch")
    # Bootstrap never detaches an existing potentially active source tree itself.
    with pytest.raises(ValueError, match="detached"):
        # The operator can select a fresh runtime instead.
        inspect_checkout(
            project,
            repo_url=str(remote),
            ref="main",
            commit=first,
            allow_local_remote=True,
        )


# Validate external values without contacting GitHub or embedding any credential.
@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://token@github.com/owner/repo",
        "http://github.com/o/r",
        "https://example.com/o/r",
        "https://github.com/o/r?token=x",
        "https://github.com/o/../r",
        "/tmp/local.git",
    ],
)
def test_production_repository_url_rejections(url: str) -> None:
    """Only a credential-free owner/repository GitHub URL is a production source."""
    # A filesystem path is never an implicit production fallback.
    with pytest.raises(ValueError):
        # No Git process or network request is needed to reject malformed input.
        repository_url(url)


# Restricted refs/commits/paths are validated before command invocation.
def test_ref_commit_path_and_actual_git_failures(local_remote, tmp_path: Path) -> None:
    """Test real clone/fetch rejection and invalid pins without source replacement."""
    # URL canonicalization is stable for equivalent explicit GitHub inputs.
    assert (
        repository_url("https://github.com/Owner/Repo")
        == "https://github.com/owner/repo.git"
    )
    # Disallow option/refspec/revision syntax and malformed component names.
    for ref in ("--upload-pack=bad", "main:other", "a..b", "a.lock", "a/../b", "a//b"):
        # Invalid values never reach a Git subprocess argument.
        with pytest.raises(ValueError):
            # Validation does not query a mutable branch tip.
            validate_ref(ref)
    # A shortened SHA cannot establish deterministic recovery.
    with pytest.raises(ValueError, match="forty-character"):
        # Full hashes are required even if the local repository could disambiguate.
        validate_commit("abc123")
    # Use only the real fixture's local bare remote from here onward.
    _, remote, _ = local_remote
    # Refuse a relative destination before Git can address the process cwd.
    with pytest.raises(ValueError, match="absolute"):
        # The API never falls back to a personal/default project path.
        checkout(str(remote), "main", None, Path("relative"), allow_local_remote=True)
    # Clone an actual non-repository directory to exercise Git's real error path.
    empty = tmp_path / "not_a_repo"
    # This is not a mocked Git response.
    empty.mkdir()
    # Failed source cloning must remain a nonzero operation.
    with pytest.raises(RuntimeError, match="Git operation failed"):
        # The helper neither invents source files nor retries a different remote.
        checkout(
            str(empty), "main", None, tmp_path / "failed_clone", allow_local_remote=True
        )
    # A nonexistent ref fails during a real local fetch after clone.
    with pytest.raises(RuntimeError, match="Git operation failed"):
        # The incomplete owned destination remains for inspection, never false success.
        checkout(
            str(remote),
            "missing_branch",
            None,
            tmp_path / "failed_ref",
            allow_local_remote=True,
        )
    # A nonexistent object cannot silently select main instead of the requested pin.
    with pytest.raises(RuntimeError, match="Git operation failed"):
        # This fetch also uses actual Git against the local bare fixture.
        checkout(
            str(remote),
            "main",
            "0" * 40,
            tmp_path / "failed_commit",
            allow_local_remote=True,
        )


# The askpass file contains only a fixed program and is removed after its scope.
def test_ephemeral_auth_and_redacted_failure(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Exercise real askpass and a simulated network authorization failure."""
    # Missing credentials do not trigger an interactive prompt or implicit fallback.
    with pytest.raises(ValueError, match="missing"):
        # Select a deliberately absent explicit secret variable.
        with git_environment("HM_TEST_MISSING_SECRET"):
            # The body must never be reached with absent authorization.
            pytest.fail("Missing auth unexpectedly entered its scope")
    # Use a synthetic secret; no account credential is needed or accessed.
    monkeypatch.setenv("HM_TEST_SECRET", "synthetic-private-token")
    # Enter the same scoped helper used by actual Git clone/fetch operations.
    with git_environment("HM_TEST_SECRET") as env:
        # The helper is a real executable token-free file.
        helper = Path(env["GIT_ASKPASS"])
        # A token must never be persisted into this temporary program's bytes.
        assert "synthetic-private-token" not in helper.read_text(encoding="utf-8")
        # Git's username and password prompts are handled without shell interpolation.
        username = subprocess.run(
            [str(helper), "Username"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # The synthetic password is returned only to the spawned process/test memory.
        password = subprocess.run(
            [str(helper), "Password"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # Both prompt branches are exercised by a real child executable.
        assert username == "x-access-token" and password == "synthetic-private-token"
    # Context cleanup must remove the owned helper even when the caller retains its
    # name.
    assert not helper.exists()

    # Simulate a GitHub authorization rejection at the external-process boundary only.
    def deny(command, **kwargs):
        # An external tool may echo arbitrary secret text; the API must not surface it.
        return subprocess.CompletedProcess(
            command, 128, "", "Auth rejected synthetic-private-token"
        )

    # Real GitHub/network interoperability remains explicitly not run.
    monkeypatch.setattr(subprocess, "run", deny)
    # The subprocess diagnostic must be replaced with an actionable redacted message.
    with pytest.raises(RuntimeError) as error:
        # No network request occurs because only this external boundary is simulated.
        checkout(
            "https://github.com/owner/private",
            "main",
            None,
            tmp_path / "private",
            token_env="HM_TEST_SECRET",
        )
    # Neither exception text nor captured stdout exposes the synthetic password.
    assert "synthetic-private-token" not in str(error.value)
    # No credential is displayed by the helper.
    assert "synthetic-private-token" not in capsys.readouterr().out


# Verify direct execution before any project-package imports or environment install.
def test_standalone_cli_and_development_identity(local_remote, tmp_path: Path) -> None:
    """Exercise the actual Python script with a real local remote and dirty metadata."""
    # Create a standalone source checkout through the executable file, not PYTHONPATH.
    seed, remote, first = local_remote
    # This invocation has no dependency on the source archive bootstrap.
    command = [
        sys.executable,
        str(ROOT / "src/hm_recsys/runtime/git_checkout.py"),
        "--repo-url",
        str(remote),
        "--ref",
        "main",
        "--commit",
        first,
        "--destination",
        str(tmp_path / "cli"),
        "--allow-local-remote",
    ]
    # Run a real child Python and inspect its actual exit/output contract.
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    # The stable JSON provenance identifies the actual pin without token values.
    assert result.returncode == 0 and first in result.stdout
    # A local seed has real HEAD metadata but no publishable GitHub origin yet.
    identity = development_identity(seed)
    # A clean committed tree can be identified without inventing a remote.
    assert identity["resolved_commit"] == first and identity["clean"] is True
    # An empty data subfolder is not itself a checkout even inside a real worktree.
    nested = seed / "ordinary_data_folder"
    # Creating this empty fixture directory does not change tracked source files.
    nested.mkdir()
    # Parent HEAD must never be misreported as this plain folder's code identity.
    assert development_identity(nested)["kind"] == "local"
    # User development edits remain allowed while being marked nonreproducible.
    (seed / "README.md").write_text("dirty local development\n", encoding="utf-8")
    # The real commit is retained but dirty work cannot pretend to be that exact code.
    assert development_identity(seed)["reproducible"] is False
    # An ordinary nonrepository directory is explicitly unversioned.
    plain = tmp_path / "plain"
    # Create no .git metadata for this final absence test.
    plain.mkdir()
    # Unversioned development must not fabricate a usable commit SHA.
    assert development_identity(plain)["kind"] == "local"
