"""Secret-hygiene guard (see the project plan's "Secret hygiene" section).

Scans the working tree -- excluding .git, .env itself, and other
gitignored paths -- for anything shaped like a DeepSeek API key, and
separately checks that if DEEPSEEK_API_KEY is set in the environment
right now, its literal value doesn't appear anywhere in tracked-shaped
files either. This is the automated backstop; the pre-commit hook
(scripts/install-hooks.sh) is the one that actually blocks a commit.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

# DeepSeek (and most OpenAI-compatible) API keys are "sk-" followed by a
# long alphanumeric token. This is intentionally broad -- false positives
# here just mean double-checking a file, which is cheap; a false negative
# is the failure mode that actually matters.
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
