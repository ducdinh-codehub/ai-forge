"""Every release needs CHANGELOG notes: the release workflow publishes them as the GitHub Release."""
import importlib.util
import re
from pathlib import Path

import pytest

from aiforge import __version__

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_notes", ROOT / ".github/scripts/release_notes.py")
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_the_current_version_has_notes():
    notes = release_notes.notes(__version__, CHANGELOG)
    assert notes.startswith("###") and f"aiforge-studio=={__version__}" in notes


def test_notes_stop_at_the_next_version():
    versions = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", CHANGELOG, re.M)
    assert versions[0] == __version__  # the newest section is the current version
    for v in versions:
        assert "## [" not in release_notes.notes(v, CHANGELOG)


def test_missing_notes_stop_the_release():
    with pytest.raises(SystemExit, match="no notes for 9.9.9"):
        release_notes.notes("9.9.9", CHANGELOG)
