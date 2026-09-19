"""Scans the tree (minus .git, .env and ignored paths) for key-shaped
strings and the live key's value. The pre-commit hook blocks commits."""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

# "sk-" plus a long token. Deliberately broad.
KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")

EXCLUDED_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", "fixtures", "results", "logs", "replay"}
EXCLUDED_FILENAMES = {".env"}
SCANNABLE_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".html", ".cfg", ".ini"}


def _scannable_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name in EXCLUDED_FILENAMES:
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix not in SCANNABLE_SUFFIXES:
            continue
        yield path


def test_no_api_key_shaped_string_in_working_tree():
    hits = []
    for path in _scannable_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in KEY_PATTERN.finditer(text):
            hits.append(f"{path.relative_to(ROOT)}: {match.group(0)[:12]}...")
    assert hits == [], "possible API key found in working tree:\n" + "\n".join(hits)


def test_env_gitignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore


def test_live_deepseek_key_value_not_present_in_tracked_shaped_files():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key or len(key) < 8:
        return  # no key set in this environment -- nothing to check
    hits = []
    for path in _scannable_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if key in text:
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], "live DEEPSEEK_API_KEY value found in:\n" + "\n".join(hits)
