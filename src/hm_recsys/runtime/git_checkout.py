"""Pinned Git checkout and source provenance without automatic code updates.

Consumes a repository URL, requested ref and immutable commit; emits a detached
verified checkout or identity. Depends only on standard-library subprocess and
filesystem APIs. Rejects dirty, mismatched and credential-bearing requests; never
commits or pushes.
"""

# Parse a standalone bootstrap command before the project environment exists.
import argparse

# Keep credentials in child environment variables and a token-free askpass file.
import contextlib

# Serialize only public checkout provenance to stdout.
import json

# Isolate Git from inherited working-tree and credential settings.
import os

# Reject option-like refs and malformed commit hashes before invoking Git.
import re
import subprocess

# Create and remove only an owned temporary authentication helper.
import tempfile

# Type the scoped environment and portable absolute checkout paths.
from collections.abc import Iterator
from pathlib import Path

# Validate URL components without accepting credentials or alternate transports.
from urllib.parse import urlsplit

# A full SHA prevents branch-tip movement from silently changing a recovered run.
_SHA = re.compile(r"[0-9a-fA-F]{40}")
# Restricted GitHub path components cannot encode credentials, options or traversal.
_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
# Refs remain arguments, but validation also prevents refspecs and revision expressions.
_REF = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*")
# The askpass program contains no token; only its Git child receives this variable.
_ASKPASS = """#!/bin/sh
case "$1" in
  *sername*) printf '%s\\n' 'x-access-token' ;;
  *) printf '%s\\n' "$HM_GIT_AUTH_TOKEN" ;;
esac
"""


# Reject credential-bearing URLs before they can reach Git, a command log or identity.
def repository_url(value: str, *, allow_local_remote: bool = False) -> str:
    """Return a canonical HTTPS GitHub URL, or an explicit offline-test path."""
    # The public API rejects invalid types without exposing their contents to Git.
    if not isinstance(value, str):
        # Configuration must identify an explicit repository URL.
        raise ValueError("Provide the actual HTTPS GitHub repository URL")
    # Local test remotes never become an implicit fallback for malformed GitHub URLs.
    if allow_local_remote and Path(value).is_absolute():
        # An existing directory supports real offline clone/fetch failure fixtures.
        if not Path(value).is_dir():
            # No path is created or repaired by repository validation.
            raise ValueError("Offline test remote must be an existing directory")
        return str(Path(value).resolve())
    # A missing URL should fail with a useful message before a subprocess is launched.
    if not value:
        # Never invent a repository or select a remote implicitly.
        raise ValueError("Provide the actual HTTPS GitHub repository URL")
    # Parse URL authority fields before considering the owner/repository path.
    parsed = urlsplit(value)
    # Only this exact public transport/host is accepted in production.
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        # Do not echo an invalid value, which might contain a credential.
        raise ValueError(
            "Repository must be credential-free HTTPS github.com/owner/repo"
        )
    # A single optional terminal slash is harmless; internal empty segments are
    # rejected.
    pieces = parsed.path.removesuffix("/").split("/")
    # GitHub repository URLs must identify exactly one owner and repository.
    if len(pieces) != 3 or pieces[0] != "":
        # Do not accept arbitrary URL paths as Git command input.
        raise ValueError("Repository URL must identify one owner and repository")
    # Normalize the optional clone suffix rather than saving multiple remote identities.
    owner, repo = pieces[1], pieces[2].removesuffix(".git")
    # Reject traversal, URL escapes and empty repository components.
    if any(not _COMPONENT.fullmatch(p) or p in (".", "..") for p in (owner, repo)):
        # An invalid component cannot be treated as a local filesystem remote.
        raise ValueError("Invalid GitHub owner or repository component")
    # GitHub owner/repository lookups are case-insensitive; provenance stays canonical.
    return f"https://github.com/{owner.lower()}/{repo.lower()}.git"


# Ref validation is deliberately independent of the current local branch state.
def validate_ref(ref: str) -> str:
    """Accept an ordinary branch/tag name, not a revision expression or refspec."""
    # Git forbids empty components, dot-leading components and lock suffixes.
    # External Notebook configuration can be malformed before argparse is involved.
    if not isinstance(ref, str):
        # Keep invalid values out of subprocess arguments and error logs.
        raise ValueError("Invalid branch/tag ref; use a plain ref name")
    # Split only validated text when enforcing Git component rules.
    parts = ref.split("/")
    # Reject shell/option characters and Git's ambiguous ref grammar explicitly.
    if (
        not _REF.fullmatch(ref)
        or ".." in ref
        or ref.endswith(".")
        or any(not p or p.startswith(".") or p.endswith(".lock") for p in parts)
    ):
        # Keep the error generic so an invalid value is not replayed into logs.
        raise ValueError("Invalid branch/tag ref; use a plain ref name")
    # Preserve the ref as informational provenance even when a SHA is pinned.
    return ref


def validate_commit(commit: str) -> str:
    """Normalize a full SHA1 commit; abbreviated hashes are not reproducible pins."""
    # GitHub currently exposes full forty-character commit identifiers in this workflow.
    if not isinstance(commit, str) or not _SHA.fullmatch(commit):
        # Never guess which object an abbreviated hash intended.
        raise ValueError("Commit must be a full forty-character hexadecimal SHA")
    # Case differences must not create incompatible run identities.
    return commit.lower()


# Authentication is explicit and temporary; ordinary public clones need no secret.
@contextlib.contextmanager
def git_environment(token_env: str | None = None) -> Iterator[dict[str, str]]:
    """Yield a child-only environment with optional token-free ephemeral askpass."""
    # Copy caller state without mutating its credentials or development configuration.
    env = os.environ.copy()
    # Git control/trace variables must not redirect this operation or log
    # authentication.
    for key in list(env):
        # User/global credentials, traces and alternate repositories are not
        # implicit inputs.
        if key in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_CONFIG",
            "GIT_CONFIG_PARAMETERS",
            "GIT_CONFIG_COUNT",
            "GIT_ASKPASS",
            "SSH_ASKPASS",
            "GIT_CURL_VERBOSE",
            "HM_GIT_AUTH_TOKEN",
        ) or key.startswith("GIT_TRACE"):
            # Remove only the copied environment's entry.
            env.pop(key, None)
    # Prevent a private-repo prompt from hanging an unattended Colab cell.
    env.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
    )
    # Public repositories run without an authentication helper.
    if token_env is None:
        # The scoped copy is discarded after this operation.
        yield env
        # Do not fall through into private authentication setup.
        return
    # Secret configuration accepts a variable name, never a token in command arguments.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env):
        # Reject malformed configuration before reading any secret.
        raise ValueError(
            "Token environment setting must be an environment variable name"
        )
    token = os.environ.get(token_env)
    # Missing access must fail before clone rather than using an unrelated
    # credential store.
    if not token or "\n" in token or "\r" in token:
        # Never include the token or its contents in a diagnostic.
        raise ValueError("Configured GitHub secret is missing or invalid")
    # Only the askpass executable is written; it contains no credential bytes.
    with tempfile.TemporaryDirectory(prefix="hm-git-askpass-") as temporary:
        helper = Path(temporary) / "askpass.sh"
        # The fixed shell program reads its password from the child-only environment.
        helper.write_text(_ASKPASS, encoding="utf-8")
        # Only this user can execute/read the temporary authentication program.
        helper.chmod(0o700)
        # Do not persist token values in URLs, Git config, files or output.
        env.update(GIT_ASKPASS=str(helper), HM_GIT_AUTH_TOKEN=token)
        # The directory is removed on success, failure or interruption.
        yield env


# Keep every Git invocation noninteractive and never expose raw remote errors.
def _git(
    args: list[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    optional: bool = False,
    allow_local_remote: bool = False,
) -> str:
    """Run Git argv with bounded time; return stdout or a redacted failure."""
    # Clear helper configuration even for a pre-existing checkout with local config.
    command = [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        "core.hooksPath=" + os.devnull,
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"protocol.file.allow={'always' if allow_local_remote else 'never'}",
        *args,
    ]
    # A command failure must not leak stderr containing a URL or echoed authorization.
    try:
        # No shell interprets owner/ref/path strings and no secret appears in argv.
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    # Missing Git and timeouts still receive an operator-actionable, credential-free
    # error.
    except (OSError, subprocess.TimeoutExpired):
        # Avoid chaining a subprocess exception with potentially sensitive diagnostics.
        raise RuntimeError(
            "Git unavailable or timed out; verify Git and repository access"
        ) from None
    # Some read-only probes intentionally return no value outside a Git checkout.
    if result.returncode and optional:
        # Do not return stderr or a partially valid revision.
        return ""
    # Clone/fetch/auth failures are never treated as a successful checkout.
    if result.returncode:
        # This covers authorization/ref/object errors without echoing tokens or
        # arbitrary URLs.
        raise RuntimeError(
            "Git operation failed; verify repository access, ref/SHA and authorization"
        )
    # Callers validate the specific root, remote and commit returned by Git.
    return result.stdout.strip()


# Filesystem validation happens before any clone can create directories.
def _destination(value: Path) -> Path:
    """Require a dedicated absolute destination without replacing a symlink/root."""
    # Preserve the literal path long enough to reject traversal and symlink leaves.
    path = Path(value)
    # These targets cannot safely serve as a dedicated source checkout.
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        # No directory is removed, reset or repaired automatically.
        raise ValueError("Checkout destination must be an absolute non-symlink path")
    # Canonicalize parent aliases for consistent worktree-root checks.
    path = path.resolve()
    # Avoid commands directed at the filesystem root or the user's entire home
    # directory.
    if path == Path(path.anchor) or path == Path.home().resolve():
        # A dedicated project path is required.
        raise ValueError("Choose a dedicated project checkout directory")
    # Creation occurs only after URL/ref/commit validation also succeeds.
    return path


# The identity verifier is shared by Notebook bootstrap and the pipeline's Git path.
def inspect_checkout(
    project: Path,
    *,
    repo_url: str,
    ref: str,
    commit: str,
    allow_local_remote: bool = False,
) -> dict:
    """Verify an existing exact clean checkout without fetching or changing code."""
    # Validate all externally supplied components before invoking Git.
    remote, requested, pinned = (
        repository_url(repo_url, allow_local_remote=allow_local_remote),
        validate_ref(ref),
        validate_commit(commit),
    )
    project = _destination(project)
    # Inspection is read-only and never authenticates against a network service.
    with git_environment() as env:
        # Reject nested directories that merely happen to inherit a parent repository.
        top = _git(["rev-parse", "--show-toplevel"], cwd=project, env=env)
        # The source root must be exactly the requested project directory.
        if Path(top).resolve() != project:
            # Do not identify another checkout as this project's source.
            raise ValueError("Destination is not the requested Git worktree root")
        actual_remote = _git(["remote", "get-url", "origin"], cwd=project, env=env)
        # Reject credentials/alternate remotes instead of rewriting origin
        # automatically.
        if (
            repository_url(actual_remote, allow_local_remote=allow_local_remote)
            != remote
        ):
            # A new destination/runtime is required for a different repository.
            raise ValueError("Existing checkout has a different origin")
        # Include untracked nonignored files; .gitignore does not hide tracked
        # modifications.
        dirty = _git(
            ["status", "--porcelain", "--untracked-files=normal"], cwd=project, env=env
        )
        # Existing edits are preserved rather than reset, stashed or deleted.
        if dirty:
            # Never print private filenames from porcelain output.
            raise ValueError(
                "Existing checkout is dirty; preserve changes and use a fresh runtime"
            )
        head = _git(["rev-parse", "HEAD^{commit}"], cwd=project, env=env)
        # Resume and new-run reuse must both point at exactly the configured commit.
        if head != pinned:
            # No checkout/pull/reset is performed on existing different code.
            raise ValueError(
                "Checkout commit differs; use original SHA or a fresh runtime"
            )
        # A detached checkout prevents incidental branch commands from advancing the
        # run.
        if _git(
            ["symbolic-ref", "--quiet", "HEAD"], cwd=project, env=env, optional=True
        ):
            # Detachment is done only while constructing a new checkout.
            raise ValueError(
                "Existing checkout must already be detached at the pinned commit"
            )
    # No timestamp/path/secret enters the stable source-control identity.
    return {
        "kind": "git",
        "remote_url": remote,
        "requested_ref": requested,
        "resolved_commit": pinned,
        "clean": True,
    }


# Create a new pinned checkout, or validate an identical existing checkout only.
def checkout(
    repo_url: str,
    ref: str,
    commit: str | None,
    destination: Path,
    resume: bool = False,
    *,
    token_env: str | None = None,
    allow_local_remote: bool = False,
) -> dict:
    """Clone/fetch and detach a new source tree; preserve all existing work.

    Args:
        repo_url: Credential-free HTTPS GitHub owner/repository URL.
        ref: Informational branch/tag and new-run resolution source when unpinned.
        commit: Full SHA; mandatory for resume and preferred for new runs.
        destination: Dedicated absolute local project root.
        resume: Require the original explicit commit, never today's branch tip.
        token_env: Optional explicitly selected token variable, consumed in memory.
        allow_local_remote: Offline tests only; allow an absolute filesystem remote.

    Returns:
        Stable, credential-free source-control provenance after verification.
    """
    # Normalize user inputs before creating a directory or starting Git.
    remote, requested = (
        repository_url(repo_url, allow_local_remote=allow_local_remote),
        validate_ref(ref),
    )
    # An unpinned recovery is nondeterministic even when main currently looks unchanged.
    if resume and commit is None:
        # Do not silently resolve the current branch tip for old artifacts.
        raise ValueError("Resume requires the original explicit commit SHA")
    # Validate a provided SHA before it is used as a Git argument.
    pinned = validate_commit(commit) if commit is not None else None
    # This check also rejects a path aimed at a symlink or filesystem root.
    destination = _destination(destination)
    # Reusing an existing tree must never fetch, pull, reset or replace its code.
    if destination.exists():
        # Existing runtime reuse is meaningful only with an explicit original pin.
        if pinned is None:
            # An operator can select a fresh runtime for a new unpinned experiment.
            raise ValueError("Existing checkout reuse requires an explicit commit SHA")
        # Inspection validates origin, cleanliness, detached state and exact HEAD.
        return inspect_checkout(
            destination,
            repo_url=remote,
            ref=requested,
            commit=pinned,
            allow_local_remote=allow_local_remote,
        )
    # Parent directories are created only for a fresh destination, never removed
    # afterward.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Any private credential is scoped to these clone/fetch children only.
    with git_environment(token_env) as env:
        # Clone code without checking out a moving branch or invoking model commands.
        _git(
            ["clone", "--no-checkout", "--", remote, str(destination)],
            env=env,
            allow_local_remote=allow_local_remote,
        )
        # Fetch the exact SHA when given, otherwise the explicitly chosen new-run ref.
        _git(
            ["fetch", "--no-tags", "--", "origin", pinned or requested],
            cwd=destination,
            env=env,
            allow_local_remote=allow_local_remote,
        )
        # The fetched object must be a commit, including when the source is an
        # annotated tag.
        resolved = _git(["rev-parse", "FETCH_HEAD^{commit}"], cwd=destination, env=env)
        # A server/ref mismatch cannot silently select a different tree.
        if pinned is not None and resolved != pinned:
            # Leave the incomplete new directory available for operator inspection.
            raise ValueError("Fetched object does not match the requested commit")
        # Validate Git's output before it can become provenance or a checkout argument.
        pinned = validate_commit(resolved)
        # New destinations alone can be detached; existing edits are never rewritten.
        _git(["checkout", "--detach", pinned], cwd=destination, env=env)
    return inspect_checkout(
        destination,
        repo_url=remote,
        ref=requested,
        commit=pinned,
        allow_local_remote=allow_local_remote,
    )


# Local development may be dirty, but its report must not pretend to be a pinned run.
def development_identity(project: Path) -> dict:
    """Record actual local HEAD/cleanliness and only a safe GitHub remote if present."""
    # Local metadata inspection never invokes a network or token helper.
    with git_environment() as env:
        # A subdirectory may inherit its parent's Git repository without owning one.
        top = _git(
            ["rev-parse", "--show-toplevel"], cwd=project, env=env, optional=True
        )
        # Only the exact project worktree root can supply reproducible project metadata.
        if not top or Path(top).resolve() != project.resolve():
            # Do not attribute an ordinary data/test folder to its parent's commit.
            return {
                "kind": "local",
                "resolved_commit": None,
                "clean": False,
                "reproducible": False,
                "remote_url": None,
            }
        # Non-Git source snapshots remain usable for bounded local development.
        head = _git(["rev-parse", "HEAD^{commit}"], cwd=project, env=env, optional=True)
        # Explicitly distinguish an unversioned directory from a reproducible commit.
        if not head:
            # No fabricated revision string is presented as a Git identifier.
            return {
                "kind": "local",
                "resolved_commit": None,
                "clean": False,
                "reproducible": False,
                "remote_url": None,
            }
        # Count tracked and untracked edits without persisting their private filenames.
        dirty = bool(
            _git(
                ["status", "--porcelain", "--untracked-files=normal"],
                cwd=project,
                env=env,
            )
        )
        # An unconfigured origin is legitimate before the user's first approved push.
        raw = _git(["remote", "get-url", "origin"], cwd=project, env=env, optional=True)
        # Unsafe or local development remotes are not published into provenance.
        try:
            # Only a credential-free GitHub identity is acceptable in the public report.
            remote = repository_url(raw) if raw else None
        # A local/credential-bearing remote remains undisclosed, not automatically
        # changed.
        except ValueError:
            # Missing public remote information is explicit.
            remote = None
    # A dirty tree can carry real HEAD metadata while remaining nonreproducible.
    return {
        "kind": "git",
        "resolved_commit": head,
        "remote_url": remote,
        "clean": not dirty,
        "reproducible": not dirty,
    }


# This source can be invoked directly before importing/installing the project package.
def main() -> None:
    """Parse credential-free bootstrap arguments and print verified provenance."""
    # Do not accept credentials as arguments that could appear in Notebook command logs.
    parser = argparse.ArgumentParser(description=__doc__)
    # The repository URL is required; no unexpected remote is selected by default.
    parser.add_argument("--repo-url", required=True)
    # A branch/tag is only an explicit new-run resolution source.
    parser.add_argument("--ref", default="main")
    # A complete SHA is required by checkout when recovering a previous run.
    parser.add_argument("--commit")
    # The destination must be explicit and absolute.
    parser.add_argument("--destination", required=True, type=Path)
    # Recovery cannot silently follow the current main branch.
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--token-env")
    # This opt-in exists only for bounded offline integration fixtures.
    parser.add_argument("--allow-local-remote", action="store_true")
    # Reject malformed command syntax before executing source-control operations.
    args = parser.parse_args()
    # Convert a Git/access rejection into a concise error without printing
    # stderr/tokens.
    try:
        # Shared API and standalone CLI use identical validation and pinning rules.
        result = checkout(
            args.repo_url,
            args.ref,
            args.commit,
            args.destination,
            args.resume,
            token_env=args.token_env,
            allow_local_remote=args.allow_local_remote,
        )
    # Invalid external configuration and Git failures are both explicit nonzero exits.
    except (ValueError, RuntimeError) as error:
        # The API messages contain no user-supplied URL, token or private filename.
        parser.error(str(error))
    # Print only the canonical public identity after all verification succeeds.
    print(json.dumps(result, sort_keys=True))


# Importing the module does not clone, fetch or modify any checkout.
if __name__ == "__main__":
    # Execute the thin standalone command only when explicitly invoked.
    main()
