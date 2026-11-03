"""Inspect current source/index/history privacy locally before a reviewed Git push."""

# Parse only explicit local paths; this command never takes GitHub credentials.
import argparse

# Render aggregate findings without printing matched secret values.
import json
import sys

# Unique UTC names preserve prior audit evidence across before/after-stage checks.
from datetime import UTC, datetime
from pathlib import Path

# Keep local entry-point behavior consistent with the other project scripts.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

# Import only the offline metadata auditor; no model or remote connection is started.
from hm_recsys.runtime.git_safety import audit_repository  # noqa: E402


# A nonzero result blocks publication but never removes files or rewrites history.
def main() -> None:
    """Write a bounded privacy report under the audited repository's data root."""
    # Command parsing happens before any filesystem or Git inspection.
    parser = argparse.ArgumentParser(description=__doc__)
    # Existing repositories are audited in place and never initialized here.
    parser.add_argument("--repo", type=Path, default=PROJECT)
    # Explicit report paths remain local and cannot overwrite prior evidence.
    parser.add_argument("--output", type=Path)
    # This threshold flags large source-like files independently of forbidden paths.
    parser.add_argument("--large-file-mib", type=int, default=10)
    # Reject malformed arguments before choosing an output location.
    args = parser.parse_args()
    # The actual audited root controls the output data tree.
    repo = args.repo.resolve()
    # A nonpositive threshold cannot define an actionable large-file policy.
    if args.large_file_mib <= 0:
        # Reject invalid input before any Git or content inspection begins.
        parser.error("--large-file-mib must be positive")
    # Distinct microsecond names retain both pre-stage and post-stage audit evidence.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    # Keep generated reports under the existing ignored runtime data root.
    output = args.output or repo / "data/runs/git_audit" / f"audit_{stamp}.json"
    # Do not write into personal paths or unrelated repositories.
    if not output.resolve().is_relative_to((repo / "data").resolve()):
        # A privacy audit does not grant arbitrary filesystem publication permission.
        parser.error(
            "Audit output must remain beneath the audited repository data root"
        )
    # Prior reports remain inspectable instead of silently changing their evidence.
    if output.exists():
        # The caller can select a new unused output filename for a new inspection.
        parser.error("Audit report exists; choose a new --output path")
    # Preserve all repository files/index/history regardless of the audit outcome.
    report = audit_repository(repo, large_file_bytes=args.large_file_mib * 1024**2)
    # Only the local report directory is created; no data/credential file is altered.
    output.parent.mkdir(parents=True, exist_ok=True)
    # JSON escapes unusual filenames and contains no matched content snippets.
    with output.open("x", encoding="utf-8") as handle:
        # Exclusive creation prevents concurrent audits from replacing prior evidence.
        handle.write(json.dumps(report, indent=2, ensure_ascii=True) + "\n")

    # Print only aggregate gate information and the explicitly chosen evidence location.
    print(
        json.dumps(
            {
                "report": str(output),
                "push_gate": report["push_gate"],
                "blocking_findings": report["blocking_findings"],
                "history_complete": report["history_complete"],
            },
            indent=2,
        )
    )
    # A completed audit with risks is a blocked push, not a successful safety gate.
    raise SystemExit(1 if report["blocking_findings"] else 0)


# Importing this thin entry point never runs an audit or modifies a repository.
if __name__ == "__main__":
    # Run only the caller's explicitly requested local inspection.
    main()
