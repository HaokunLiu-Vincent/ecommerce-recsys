"""Build a source-only Colab release under the repository's data/releases."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hm_recsys.colab.release import build_release  # noqa: E402


def main() -> None:
    """Use the current checkout without initializing or changing Git."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "data/releases")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(PROJECT / "data/releases"):
        parser.error("Releases must remain under data/releases")
    result = build_release(PROJECT, args.output.resolve())
    companion = args.output / (Path(result["archive"]).stem + ".pilot_runbook.md")
    text = (PROJECT / "docs/archi/colab_pilot_runbook.md").read_text(encoding="utf-8")
    text = text.replace("{{RELEASE_NAME}}", result["archive"])
    stem = Path(result["archive"]).stem
    text += (
        "\n## Resolved release files\n\n"
        f"- Notebook: `{PROJECT / 'notebooks/01_Colab_Training.ipynb'}`\n"
        f"- ZIP: `{args.output / result['archive']}`\n"
        f"- SHA256 file: `{args.output / (stem + '.sha256')}`\n"
        f"- Manifest: `{args.output / (stem + '.manifest.json')}`\n"
        f"- Archive SHA256: `{result['archive_sha256']}`\n"
    )
    if companion.exists() and companion.read_text(encoding="utf-8") != text:
        raise ValueError("Existing resolved runbook differs; preserve it")
    companion.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ("build_id", "archive", "archive_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
