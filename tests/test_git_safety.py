"""Audit publication safely using commits in isolated temporary repositories."""

# Construct clean and output-bearing notebooks without executing their cells.
import json

# Run local Git fixtures directly with explicit arguments, never remote URLs.
import subprocess

# Invoke the thin audit entry point with the existing interpreter.
import sys

# Keep test files and repositories inside the fixture's owned directory.
from pathlib import Path

# Require explicit rejection of invalid repository/audit limits.
import pytest

# Locate the actual entry point and ignore policy without a cwd assumption.
from hm_recsys.core.contracts import ROOT

# Inspect the same auditor used by the operator's CLI.
from hm_recsys.runtime import git_safety


# Local fixture commands never run against the user's working repository.
def _git(repo: Path, *args: str) -> str:
    """Execute local fixture setup and return only its nonsecret metadata."""
    # Assertions contain no dataset, credentials or external authentication material.
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    # Fixture commits and object IDs are intentionally local and temporary.
    return result.stdout.strip()


# Initialize only a fresh temporary repository, which the task explicitly permits.
def _repo(tmp_path: Path) -> Path:
    """Create a tiny committed source checkout without network or real inputs."""
    # The fixture path is distinct from the user's actual source repository.
    repo = tmp_path / "repo"
    # A fresh directory prevents accidental modification of another fixture/run.
    repo.mkdir()
    # The current project's Git metadata is never initialized or replaced.
    _git(repo, "init")
    # Identity is scoped to this temporary repository rather than global configuration.
    _git(repo, "config", "user.name", "Offline Fixture")
    # This deliberately invalid-domain address has no real authentication role.
    _git(repo, "config", "user.email", "fixture@example.invalid")
    # Test the actual corrected ignore policy rather than a weaker custom substitute.
    (repo / ".gitignore").write_text(
        (ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    # Only small authored source metadata enters the first fixture commit.
    (repo / "README.md").write_text("# Synthetic source fixture\n", encoding="utf-8")
    # Stage only fixture source, never any files from the user's checkout.
    _git(repo, "add", ".gitignore", "README.md")
    # A fixture commit supplies real HEAD/history metadata for the audit.
    _git(repo, "commit", "-m", "synthetic source")
    # Return the owned root so each test controls its own history.
    return repo


# A clean local audit is evidence only about local source, never a GitHub push.
def test_clean_repo_and_unique_cli_reports(tmp_path: Path) -> None:
    """Keep clean and dirty-review states distinct with unique default reports."""
    # The fixture contains only safely publishable synthetic text.
    repo = _repo(tmp_path)
    # Inspect actual index, HEAD and history without contacting a remote.
    result = git_safety.audit_repository(repo)
    # Every committed fixture path is bounded and content-scanned.
    assert result["push_gate"] == "clear_local_audit" and not result["findings"]
    # No GitHub interoperability or remote publication is inferred from local success.
    assert (
        result["network_accessed"] is False and result["history_commits_scanned"] == 1
    )
    # Exercise the actual script twice, as before/after-stage operator instructions do.
    for _ in range(2):
        # The default output belongs to the fixture's ignored data tree.
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/audit_git.py"), "--repo", str(repo)],
            capture_output=True,
            text=True,
        )
        # A safe privacy result exits zero even though no actual push is performed.
        assert completed.returncode == 0, completed.stdout
    # Each invocation preserved its earlier report using a distinct UTC filename.
    assert len(list((repo / "data/runs/git_audit").glob("*.json"))) == 2
    # Uncommitted source can be developed locally but is not a reproducible commit.
    (repo / "README.md").write_text("# Changed synthetic source\n", encoding="utf-8")
    # The gate now requires a reviewed commit rather than claiming clean pinned code.
    assert git_safety.audit_repository(repo)["push_gate"] == "needs_review_commit"


# A current ignore rule cannot erase either the index or previously committed contents.
def test_tracked_ignore_staged_secret_and_deleted_history(tmp_path: Path) -> None:
    """Detect current/staged/history risk without exporting private matches."""
    # Set up an isolated source-only repository before deliberately adding unsafe
    # fixtures.
    repo = _repo(tmp_path)
    # The data file is fake but tests the exact tracked-ignore failure mode.
    (repo / "data").mkdir()
    # No real customer record or purchase history is present in the fixture.
    (repo / "data/private.csv").write_text(
        "synthetic_column\nsynthetic_value\n", encoding="utf-8"
    )
    # Construct a recognizable fake token only at runtime, so test source itself has
    # no token.
    fake_token = "ghp_" + "a" * 36
    # Deliberately force-add ignored fixtures only in this temporary repository.
    (repo / ".env").write_text(fake_token, encoding="utf-8")
    # Ordinary notebook output text must never appear in the audit's returned JSON.
    private_output = "synthetic-output-must-not-be-exported"
    # A real persisted notebook cell exercises structural output detection.
    book = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "outputs": [{"output_type": "stream", "text": private_output}],
                "source": "1",
            }
        ]
    }
    # Keep the fixture's notebook small and independent of current production notebooks.
    (repo / "demo.ipynb").write_text(json.dumps(book), encoding="utf-8")
    # Forced adds demonstrate why adding ignore patterns is insufficient.
    _git(repo, "add", "-f", "data/private.csv", ".env", "demo.ipynb")
    # Commit unsafe synthetic files so deleted historical secrets can be detected later.
    _git(repo, "commit", "-m", "synthetic risky history")
    # Removing a file from HEAD does not remove its bytes from reachable history.
    _git(repo, "rm", ".env")
    # This cleanup is confined to the owned fixture and deliberately leaves old history.
    _git(repo, "commit", "-m", "synthetic deletion")
    # Stage a different fake assignment then clean only its current working bytes.
    staged_secret = "b" * 32
    # The content pattern is assembled at runtime and is not a committed real key.
    (repo / "config.txt").write_text(
        'api_key="' + staged_secret + '"', encoding="utf-8"
    )
    # The index must be inspected separately from the current working file.
    _git(repo, "add", "config.txt")
    # The staged secret stays detectable despite a clean-looking worktree version.
    (repo / "config.txt").write_text("clean synthetic working copy\n", encoding="utf-8")
    # Run the actual complete offline metadata/content audit.
    report = git_safety.audit_repository(repo)
    # Risks block publication without automatically untracking or rewriting anything.
    assert report["push_gate"] == "blocked" and report["ignored_but_tracked_count"] == 1
    # A current ignored-but-tracked path needs explicit index remediation.
    assert any(x["indicator"] == "ignored_but_tracked" for x in report["findings"])
    # Deleted committed credentials remain a history finding.
    assert any(
        x["scope"] == "history" and x["indicator"] == "github_token_shape"
        for x in report["findings"]
    )
    # The staged snapshot is audited independently of the cleaned current file.
    assert any(
        x["scope"] == "index" and x["indicator"] == "credential_assignment"
        for x in report["findings"]
    )
    # Saved notebook outputs are blockers even without recognizable token syntax.
    assert any(
        x["indicator"] == "persisted_notebook_outputs" for x in report["findings"]
    )
    # Reports contain only category labels/paths, never matching credential or output
    # content.
    encoded = json.dumps(report)
    # Even deliberately synthetic secrets must not be echoed by the audit report.
    assert (
        fake_token not in encoded
        and staged_secret not in encoded
        and private_output not in encoded
    )
    # Audit execution left all fixture index and history records in place.
    assert _git(repo, "rev-list", "--all", "--count") == "3"


# Large historical content is bounded before git cat-file can decompress it.
def test_large_blob_is_metadata_only(tmp_path: Path, monkeypatch) -> None:
    """Refuse clean coverage for oversized content without reading its blob bytes."""
    # Start from a safe owned fixture repository.
    repo = _repo(tmp_path)
    # A compressible fixture exceeds the content cap without using substantial memory.
    (repo / "large.txt").write_bytes(b"x" * (git_safety.CONTENT_LIMIT + 1))
    # Record a real historical object to inspect through Git's size metadata.
    _git(repo, "add", "large.txt")
    # Only the temporary repository gains this fixture commit.
    _git(repo, "commit", "-m", "oversized synthetic source")
    # Obtain the real object ID without loading it through the audit path.
    oid = _git(repo, "rev-parse", "HEAD:large.txt")
    # Retain the real read-only command adapter for all allowed metadata operations.
    original = git_safety._git

    # Reject any accidental full blob read while permitting exact object-size queries.
    def guarded(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
        # The auditor must inspect size before deciding whether content fits its budget.
        assert args != ("cat-file", "blob", oid)
        # Preserve every other real Git command in the test.
        return original(repo, *args, input_bytes=input_bytes)

    # The guard affects only this fixture's audit call and is restored by pytest.
    monkeypatch.setattr(git_safety, "_git", guarded)
    # The smaller publication threshold is deliberate fixture policy, not a larger
    # data budget.
    report = git_safety.audit_repository(repo, large_file_bytes=512)
    # Both size risk and missing bounded content coverage block a false clean result.
    reasons = {x["indicator"] for x in report["findings"]}
    # No private/synthetic blob content was decompressed by the audit code.
    assert {"large_file", "content_not_scanned_over_budget"} <= reasons


# Invalid inputs fail explicitly rather than initializing Git or producing a clean gate.
def test_invalid_repository_threshold_and_cli_output(tmp_path: Path) -> None:
    """Validate repository ownership, positive budgets and local report paths."""
    # A temporary folder may inherit a parent repo; either explicit refusal is valid.
    with pytest.raises(
        ValueError, match="Local Git audit|Audit path must be the Git repository root"
    ):
        # No git init is part of the production audit implementation.
        git_safety.audit_repository(tmp_path)
    # Prepare a valid fixture for independent threshold/output validation.
    repo = _repo(tmp_path)
    # A zero threshold cannot mean that arbitrarily large blobs are safe.
    with pytest.raises(ValueError, match="positive"):
        # No source bytes need reading to reject the invalid budget.
        git_safety.audit_repository(repo, large_file_bytes=0)
    # The thin command also rejects invalid thresholds before writing any report.
    bad = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_git.py"),
            "--repo",
            str(repo),
            "--large-file-mib",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    # argparse returns two for an invalid operator request.
    assert bad.returncode == 2
    # An audit may not write outside its repository's ignored data tree.
    outside = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_git.py"),
            "--repo",
            str(repo),
            "--output",
            str(tmp_path / "outside.json"),
        ],
        capture_output=True,
        text=True,
    )
    # The external destination remains untouched.
    assert outside.returncode == 2 and not (tmp_path / "outside.json").exists()


# Auth stores need path-based detection even when their format is ordinary YAML.
def test_authentication_and_generated_path_policy() -> None:
    """Cover credential directories/names without blocking normal source modules."""
    # These locations remain risky when someone force-adds an ignored file.
    cases = (
        ".git-credentials",
        ".config/gh/hosts.yml",
        ".config/git/credentials",
        "tokens.yaml",
        "token.json",
        ".cache/model.bin",
        "build/pkg.py",
        "dist/pkg.egg-info/PKG-INFO",
    )
    # None of the path indicators requires opening an actual authentication store.
    for path in cases:
        # At least one explicit category must prevent a false clean-source claim.
        assert git_safety.path_indicators(path), path
    # Source modules discussing token handling are not credential store files.
    assert not git_safety.path_indicators("src/token_helpers.py")


def test_canonical_data_source_is_not_a_runtime_dataset() -> None:
    """Allow canonical source while retaining every runtime artifact boundary."""
    assert not git_safety.path_indicators("src/hm_recsys/data/history.py")
    assert git_safety.path_indicators("src/hm_recsys/data/customer.csv")
    assert git_safety.path_indicators("data/history.py")
    assert git_safety.path_indicators("docs/achive/source.py")
    assert git_safety.path_indicators("exported_customers.jsonl")
    assert git_safety.path_indicators("exported_histories.ndjson")
