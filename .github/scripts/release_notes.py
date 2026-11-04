"""Print the CHANGELOG.md notes of one version, for its GitHub Release.

    python .github/scripts/release_notes.py 0.2.0 > notes.md

Exits with an error when the CHANGELOG has no section for that version, so a release
without notes stops before anything is published.
"""
import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[2] / "CHANGELOG.md"


def notes(version: str, text: str) -> str:
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[Unreleased\]:|\Z)", text, re.S | re.M)
    if not m or not m.group(1).strip():
        raise SystemExit(f"CHANGELOG.md has no notes for {version}: add a '## [{version}] - date' section")
    return (f"{m.group(1).strip()}\n\n---\n"
            "**Install or update in one line** (no Python needed): macOS / Linux "
            "`curl -LsSf https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/install.sh | sh` · "
            "Windows `irm https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/install.ps1 | iex`\n\n"
            f"**With pip:** `pipx install aiforge-studio=={version}` · "
            f"[PyPI](https://pypi.org/project/aiforge-studio/{version}/) · "
            "[Full changelog](https://github.com/ducdinh-codehub/ai-forge/blob/main/CHANGELOG.md)")


if __name__ == "__main__":
    print(notes(sys.argv[1].removeprefix("v"), CHANGELOG.read_text(encoding="utf-8")))
