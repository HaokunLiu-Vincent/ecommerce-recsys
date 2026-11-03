"""Read-only Git privacy auditing for source, artifacts and credentials.

Consumes a repository checkout and bounded Git metadata; emits path/content risk
findings with identifiers redacted. Uses standard-library Git subprocesses. Ignore
rules do not exempt tracked files or historical data; the audit never rewrites
history.
"""

# Keep content parsing limited to JSON notebooks and bounded byte strings.
import json

# Git output uses filesystem encoding; optional locks remain disabled for inspection.
import os

# Match credential indicators without emitting matched values or surrounding content.
import re

# Execute argument lists directly; no network or shell interpolation is needed.
import subprocess
from pathlib import Path

# Detect URL-embedded credentials without printing the configured URL.
from urllib.parse import urlsplit

# Large blobs are metadata-audited but never read in full by the content scanner.
CONTENT_LIMIT = 1024 * 1024
# The audit is deliberately bounded on the development machine.
MAX_COMMITS = 1000
# Paths identify artifact families; they do not authorize automatic untracking.
ARTIFACT_SUFFIXES = {
    ".csv",
    ".tsv",
    ".jsonl",
    ".ndjson",
    ".npy",
    ".npz",
    ".parquet",
    ".arrow",
    ".feather",
    ".ipc",
    ".faiss",
    ".index",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
    ".duckdb",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pkl",
    ".pickle",
    ".joblib",
    ".h5",
    ".hdf5",
    ".bin",
    ".zip",
    ".tar",
    ".tgz",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tiff",
}
# Exact path components avoid confusing source packages with runtime artifact trees.
PRIVATE_DIRS = {
    "achive",
    "data",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    ".aws",
    ".ssh",
    ".kaggle",
    "checkpoints",
    "runs",
    "artifacts",
    "text_features",
    "image_features",
    "embeddings",
    "mlruns",
    "wandb",
    "models",
    "outputs",
    "logs",
    ".cache",
    ".uv",
    ".coverage",
    "htmlcov",
    "build",
    "dist",
}
# Indicators are intentionally labelled for manual review, not proof of a live key.
SECRET_PATTERNS = {
    "private_key_header": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "github_token_shape": (
        rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"
    ),
    "aws_access_key_shape": rb"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "openai_key_shape": rb"sk-(?:proj-)?[A-Za-z0-9_-]{32,}",
    "credential_assignment": (
        rb"(?i)(?:api[_-]?key|access[_-]?token|password|secret[_-]?key)"
        rb"\s*[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_+/=-]{20,}[\"']"
    ),
    "credential_url": rb"https?://[^\s/\"<>]+:[^\s/\"<>]+@",
}


# Keep all Git operations local and suppress optional index refresh writes.
def _git(repo: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    """Return bounded local metadata or raise without echoing private stderr."""
    # This environment does not change the user's persistent Git configuration.
    environment = dict(
        os.environ,
        GIT_OPTIONAL_LOCKS="0",
        GIT_TERMINAL_PROMPT="0",
        GIT_NO_LAZY_FETCH="1",
        GIT_LFS_SKIP_SMUDGE="1",
    )
    # A caller's unrelated alternate index/repository must not redirect this audit.
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        # Only the child environment is sanitized.
        environment.pop(key, None)
    # check-ignore exits one when none of the supplied paths match.
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(repo), *arguments],
        input=input_bytes,
        capture_output=True,
        env=environment,
        timeout=60,
    )
    # Git errors may contain private remote URLs, so emit only the command family.
    if result.returncode not in (0, 1) or (
        result.returncode == 1 and arguments[0] != "check-ignore"
    ):
        # Never convert an incomplete audit into a clean result.
        raise ValueError(f"Local Git audit command failed: {arguments[0]}")
    # Reject unreasonable metadata rather than scanning an unbounded repository.
    if len(result.stdout) > 16 * 1024**2:
        # The content of the oversized output stays out of the error message.
        raise ValueError("Git audit metadata exceeds the 16 MiB inspection budget")
    return result.stdout


# A path is useful for remediation, but long opaque user identifiers are not public
# logs.
def _display_path(value: str) -> str:
    """Redact long identifier-like path components in reports."""
    # Customer IDs and credential-shaped filenames need not be disclosed to identify
    # risk.
    return re.sub(r"[A-Za-z0-9_-]{48,}", "<redacted-identifier>", value)


# Path rules are independent of .gitignore because tracked files ignore ignore rules.
def path_indicators(value: str) -> list[str]:
    """Classify forbidden artifact/credential paths without opening their contents."""
    # Git repository paths always use forward slashes.
    path = Path(value)
    # Case-insensitive checks also cover accidental platform-specific capitalization.
    parts = {part.lower() for part in path.parts}
    # The canonical data package contains reviewed Python source, not runtime data.
    # Extension/content checks below still apply; no CSV/vector exception is made.
    if path.parts[:3] == ("src", "hm_recsys", "data") and path.suffix == ".py":
        parts.discard("data")
    # Retain all reasons so reviewers can distinguish paths from content indicators.
    reasons = []
    # A runtime data/artifact component is forbidden anywhere in tracked source.
    if parts & PRIVATE_DIRS:
        # Report the policy family rather than any private values.
        reasons.append("private_or_runtime_directory")
    # Authentication tools can store OAuth material in ordinary YAML/JSON files.
    normalized = value.lower().replace("\\", "/")
    # Match known private config trees even when nested beneath a home-directory copy.
    if any(
        fragment in "/" + normalized
        for fragment in ("/.config/gh/", "/.config/git/", "/.config/gcloud/")
    ):
        # Merely ignoring the path does not protect an already tracked authentication
        # file.
        reasons.append("authentication_config_directory")
    # Build metadata directories and coverage caches are not source inputs.
    if any(part.endswith(".egg-info") for part in parts):
        # A custom package name must not bypass the general generated-metadata rule.
        reasons.append("generated_package_metadata")
    # Models, tables, source images and opaque archives require explicit review.
    if path.suffix.lower() in ARTIFACT_SUFFIXES or value.lower().endswith(".tar.gz"):
        # A small model remains private even when it does not exceed the size threshold.
        reasons.append("artifact_extension")
    # Credentials often have ordinary text/JSON extensions, so inspect names separately.
    name = path.name.lower()
    # Example environment templates are still content-scanned for accidentally real
    # keys.
    if (
        name == ".env"
        or (name.startswith(".env.") and name != ".env.example")
        or name
        in ("kaggle.json", ".netrc", ".git-credentials", "model.txt", "lambdamart.txt")
        or name.startswith(
            (
                "id_rsa",
                "id_ed25519",
                "credentials",
                "client_secret",
                "service-account",
                "service_account",
            )
        )
        or path.suffix.lower() in (".pem", ".key", ".p12", ".pfx")
    ):
        # This classification does not include the file's actual credential payload.
        reasons.append("credential_or_model_filename")
    # Credential stores use several text formats; Python source named token_helpers
    # stays valid.
    if name.startswith(("token", "oauth", "hosts.")) and path.suffix.lower() in (
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
    ):
        # These names require review even when no recognizable secret value appears yet.
        reasons.append("authentication_store_filename")
    return reasons


# Scan a bounded object once and retain only categorical findings.
def content_indicators(value: bytes, *, notebook: bool = False) -> list[str]:
    """Return secret-shape and persisted-notebook-output indicators."""
    # Pattern matches are evidence for review, not exported secret strings.
    reasons = [
        name for name, pattern in SECRET_PATTERNS.items() if re.search(pattern, value)
    ]
    # Notebook outputs can contain private histories even when no credential pattern
    # matches.
    if notebook:
        # Malformed notebooks are a review blocker rather than silently accepted source.
        try:
            # Parse only the already bounded blob; no cells are executed.
            book = json.loads(value)
        # Error details could contain private content, so report a fixed category.
        except (ValueError, UnicodeDecodeError):
            # Preserve an actionable failure without printing the document bytes.
            reasons.append("invalid_notebook_json")
        else:
            # Reject malformed cell containers rather than crashing or assuming safety.
            cells = book.get("cells") if isinstance(book, dict) else None
            # An incomplete or scalar JSON document is not a valid source notebook.
            if not isinstance(cells, list) or any(
                not isinstance(cell, dict) for cell in cells
            ):
                # Emit a fixed category without exporting malformed private bytes.
                reasons.append("invalid_notebook_json")
            # Persisted cell traces or outputs need explicit publication review.
            elif any(
                cell.get("outputs") or cell.get("execution_count") is not None
                for cell in cells
                if cell.get("cell_type") == "code"
            ):
                # The report does not include output text, images or counts per
                # customer.
                reasons.append("persisted_notebook_outputs")
    # Leave interpretation to the explicit review gate; never rewrite files.
    return reasons


# Tree entries provide historical paths and blob identities without checkout or fetch.
def _tree(repo: Path, revision: str) -> list[tuple[str, str]]:
    """Return regular historical blob coordinates from one local commit."""
    # NUL delimiters preserve literal spaces and newlines in tracked paths.
    records = _git(repo, "ls-tree", "-r", "-z", revision).split(b"\0")
    result = []
    # Each tree entry contains mode, type, object ID, tab, then the literal path.
    for record in records:
        # The trailing NUL produces an empty final entry.
        if not record:
            # It is not a file or an indication of missing history.
            continue
        # Separate metadata from the filename before decoding it for policy
        # classification.
        metadata, path = record.split(b"\t", 1)
        # Submodule trees are not recursively fetched or opened by an offline audit.
        if metadata.split()[1] == b"blob":
            # Keep opaque object IDs separate from the human-readable path.
            result.append((metadata.split()[2].decode("ascii"), os.fsdecode(path)))
    # A caller may deduplicate identical blobs across commits without dropping path
    # checks.
    return result


# The audit reports all risk and dirty state; it never stages, untracks or publishes
# code.
def audit_repository(repo: Path, *, large_file_bytes: int = 10 * 1024**2) -> dict:
    """Audit worktree, index, HEAD and reachable local history without network.

    Content checks read at most one MiB per blob. Oversized content is explicitly
    unscanned and blocks a clean audit. Git history is bounded to 1,000 commits.
    No source or matched secret contents are included in the returned report.
    """
    repo = repo.resolve(strict=True)
    # Reject invalid size thresholds rather than allowing a hidden no-op gate.
    if large_file_bytes <= 0:
        # A negative limit does not have a meaningful safety interpretation.
        raise ValueError("Large-file audit threshold must be positive")
    # Confirm Git ownership without initializing or altering repository state.
    root = Path(
        os.fsdecode(_git(repo, "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    # The caller must name the repository root, not an unrelated nested directory.
    if root != repo:
        # This prevents silently auditing the wrong checkout's tracked files.
        raise ValueError("Audit path must be the Git repository root")
    # Keep status as aggregate counts; raw filenames are reported only for findings.
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=normal")
    # Both tracked and currently addable source files need bounded content checks.
    tracked = [os.fsdecode(p) for p in _git(repo, "ls-files", "-z").split(b"\0") if p]
    # Ignored datasets are not recursively read merely to perform a code audit.
    untracked = [
        os.fsdecode(p)
        for p in _git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(
            b"\0"
        )
        if p
    ]
    ignored = _git(
        repo,
        "check-ignore",
        "--no-index",
        "--stdin",
        "-z",
        input_bytes=b"\0".join(os.fsencode(p) for p in tracked)
        + (b"\0" if tracked else b""),
    )
    # An ignore rule is not protection for any path in this set.
    ignored_tracked = {os.fsdecode(p) for p in ignored.split(b"\0") if p}
    # Findings store policy categories and redacted paths, never matching bytes.
    findings = []
    # Deduplicate the same object/path/reason within one audit scope.
    seen_findings = set()

    # Centralize safe finding construction so no caller accidentally exports contents.
    def add(scope: str, path: str, reason: str) -> None:
        # Store only stable labels and a minimally identifying relative path.
        key = (scope, _display_path(path), reason)
        # Duplicate findings add no useful remediation information.
        if key not in seen_findings:
            # Mark this report coordinate as already emitted.
            seen_findings.add(key)
            # Do not attach file snippets, regex matches or raw Git stderr.
            findings.append({"scope": scope, "path": key[1], "indicator": reason})

    # Inspect bounded bytes only when a file fits the content budget.
    def inspect(
        scope: str,
        path: str,
        size: int,
        data: bytes | None,
        cached_reasons: list[str] | None = None,
    ) -> None:
        # File size is independent of path/secret classification.
        if size >= large_file_bytes:
            # A large source-like file also warrants an explicit commit review.
            add(scope, path, "large_file")
        # Classify every actual path, including paths hidden by an ignore rule.
        for reason in path_indicators(path):
            # This is metadata evidence, not a scan of ignored private datasets.
            add(scope, path, reason)
        # Missing content coverage cannot be represented as a completely safe scan.
        if cached_reasons is not None:
            # The blob cache holds labels only, never all historical payloads.
            for reason in cached_reasons:
                add(scope, path, reason)
        # Oversized payloads have metadata coverage only and remain review blockers.
        elif data is None:
            # The artifact remains untouched; reviewers must resolve its publication
            # policy.
            add(scope, path, "content_not_scanned_over_budget")
        else:
            # Inspect ordinary text and notebook structure without executing anything.
            for reason in content_indicators(
                data, notebook=path.lower().endswith(".ipynb")
            ):
                # Only the indicator label enters the report.
                add(scope, path, reason)

    # Check current files independently from the staged snapshot and committed version.
    for name in sorted(set(tracked + untracked)):
        # A tracked ignore match remains tracked until the user explicitly changes
        # the index.
        if name in ignored_tracked:
            # This is the key distinction a .gitignore-only check would miss.
            add("index", name, "ignored_but_tracked")
        # Worktree symlinks can point outside the checkout and must not be followed.
        path = repo / name
        # Treat links as a review issue rather than opening external content.
        if path.is_symlink():
            # Submodule/linked content needs a separate explicit policy.
            add("worktree", name, "symlink_not_content_scanned")
            # Never resolve or read the external target.
            continue
        # A tracked deletion is still covered by the index and historical blob scans.
        if not path.resolve().is_relative_to(repo):
            # Do not follow a linked parent directory outside the audited checkout.
            add("worktree", name, "external_path_not_content_scanned")
            # External bytes are never needed to establish that review is blocked.
            continue
        # A deleted tracked path is still audited from its index/history object.
        if not path.is_file():
            # Missing worktree bytes are not evidence that old private bytes
            # disappeared.
            continue
        # The stat gives the bound before any content allocation.
        size = path.stat().st_size
        data = path.read_bytes() if size <= CONTENT_LIMIT else None
        # Current untracked source is reviewed before a future git add operation.
        inspect("worktree", name, size, data)
    # Index objects can differ from both HEAD and the current working file.
    index_entries = _git(repo, "ls-files", "--stage", "-z").split(b"\0")
    # Cache content reasons per object without retaining full historical blobs.
    blob_cache = {}

    def inspect_blob(scope: str, oid: str, path: str) -> None:
        # JSON structure checks depend on the path's extension, not the blob alone.
        key = (oid, path.lower().endswith(".ipynb"))
        # Repeated historical versions retain only sizes and categorical findings.
        if key not in blob_cache:
            # Exact object size is known before any historical content allocation.
            size = int(_git(repo, "cat-file", "-s", oid))
            # A large object is classified from metadata without decompression.
            if size > CONTENT_LIMIT:
                # Preserve the explicit content coverage limitation.
                reasons = ["content_not_scanned_over_budget"]
            else:
                # Hold at most one small blob while extracting nonsecret category
                # labels.
                data = _git(repo, "cat-file", "blob", oid)
                # Matched values and snippets never leave this local scan.
                reasons = content_indicators(data, notebook=key[1])
            # Cache no raw bytes, keeping memory proportional to small metadata labels.
            blob_cache[key] = (size, reasons)
        # Each historical filename still receives its own path/size policy checks.
        size, reasons = blob_cache[key]
        # Ordinary source renames cannot hide forbidden earlier paths.
        inspect(scope, path, size, None, reasons)

    # Audit staged/unstaged index entries without requiring a commit to exist.
    for entry in index_entries:
        # The final NUL adds no index record.
        if not entry:
            # Continue to the next real tracked entry.
            continue
        # Filenames can contain literal spaces, so split only at Git's tab separator.
        metadata, name = entry.split(b"\t", 1)
        # Non-blob submodule entries are deliberately not fetched.
        if metadata.split()[0] != b"160000":
            # The staged blob remains risky even if the working file was already
            # cleaned.
            inspect_blob(
                "index", metadata.split()[1].decode("ascii"), os.fsdecode(name)
            )
        else:
            # An offline audit cannot attest to a separately hosted submodule tree.
            add("index", os.fsdecode(name), "submodule_not_scanned")
    # A bounded reachable-commit list includes branches/tags without contacting remotes.
    commits = _git(
        repo, "rev-list", "--all", f"--max-count={MAX_COMMITS + 1}"
    ).splitlines()
    # Exceeding the cap must stop the clean gate, not silently truncate history.
    history_complete = len(commits) <= MAX_COMMITS
    # The report records this scope limitation explicitly.
    if not history_complete:
        # No sensitive commit content is printed with this warning.
        add("history", "<repository>", "history_commit_budget_exceeded")
    head = (
        _git(repo, "rev-parse", "--verify", "--quiet", "HEAD").strip()
        if commits
        else b""
    )
    # HEAD findings remain distinguishable from earlier deleted files.
    if head:
        # Tree metadata is read without changing the checkout.
        for oid, path in _tree(repo, head.decode("ascii")):
            # Current committed source is independently checked from the index.
            inspect_blob("head", oid, path)
    # Avoid emitting identical historical coordinates for every unchanged commit.
    history_paths = set()
    # Walk only the declared bounded local history.
    for commit in commits[:MAX_COMMITS]:
        # No checkout, fetch, reset or history rewrite occurs here.
        for oid, path in _tree(repo, commit.decode("ascii")):
            # Distinct content and renames both matter to the publication audit.
            if (oid, path) not in history_paths:
                # Preserve this path/version coordinate for report completeness.
                history_paths.add((oid, path))
                # Detect sensitive blobs even when removed from HEAD or ignored today.
                inspect_blob("history", oid, path)
    # Do not persist bounded content buffers after the audit returns.
    blob_cache.clear()
    remote_names = _git(repo, "remote").splitlines()
    # Missing origin is a setup requirement, not permission to invent a URL.
    origin_configured = b"origin" in remote_names
    # Embedded URL credentials are a privacy risk even if source files are clean.
    if origin_configured:
        # Keep the actual URL confined to this short-lived local variable.
        remote = os.fsdecode(_git(repo, "remote", "get-url", "origin")).strip()
        # SSH's git@host syntax is not an embedded HTTPS token.
        parsed = (
            urlsplit(remote) if remote.startswith(("https://", "http://")) else None
        )
        # Never echo username/password/query-bearing authentication values.
        if parsed and (
            parsed.username is not None or parsed.password is not None or parsed.query
        ):
            # The report identifies only the configuration entry needing review.
            add("configuration", "remote.origin.url", "credential_bearing_remote_url")
    # Dirty work is valid local development but not a reproducible pushed commit.
    dirty = bool(status)
    # Source snapshots remain distinguishable from completed publication authorization.
    return {
        "version": 1,
        "network_accessed": False,
        "repository_initialized": False,
        "tracked_files": len(tracked),
        "untracked_addable_files": len(untracked),
        "index_entries": sum(bool(x) for x in index_entries),
        "head": head.decode("ascii") if head else None,
        "history_commits_scanned": min(len(commits), MAX_COMMITS),
        "history_complete": history_complete,
        "historical_path_versions": len(history_paths),
        "ignored_but_tracked_count": len(ignored_tracked),
        "content_limit_bytes": CONTENT_LIMIT,
        "large_file_threshold_bytes": large_file_bytes,
        "origin_configured": origin_configured,
        "worktree_dirty": dirty,
        "findings": findings,
        "blocking_findings": len(findings),
        "push_gate": "blocked"
        if findings
        else "needs_review_commit"
        if dirty
        else "clear_local_audit",
        "actions_taken": "read_only_no_staging_untracking_commit_or_push",
        "limitations": (
            "Pattern indicators require review; no audit proves absence of "
            "every possible secret."
        ),
    }
