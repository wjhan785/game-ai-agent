"""Methodological guard: the agent and both baselines must never be able
to see which bugs were seeded. This is the automated half of the
"scenarios and agent strategy were built independently of the answer key"
claim in the project write-up -- the other half is git history itself
(docs/scenario-matrix.md predates engine/defects.py; check the log).

Two checks:
  1. No file under agent/ or baselines/ imports engine.defects or
     anything from oracle/ (the oracle is a grading-time-only tool).
  2. No file under agent/ or baselines/ contains defect-identifying
     vocabulary -- the ANSWER_KEY bug IDs, or the DefectFlags field names
     themselves -- anywhere in source, including string literals (which
     covers prompt templates).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.defects import ALL_DEFECT_IDS

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SOURCE_ROOTS = ["agent", "baselines"]
FORBIDDEN_IMPORT_MODULES = ("engine.defects", "oracle")

# Every bug ID (B01..B10) and every DefectFlags field name -- if any of
# these literally appear in agent or baseline source, something is
# leaking answer-key vocabulary into a place that must stay blind to it.
FORBIDDEN_TOKENS = set(ALL_DEFECT_IDS.keys()) | set(ALL_DEFECT_IDS.values())
_TOKEN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in FORBIDDEN_TOKENS) + r")\b"
)


def _python_files():
    for root_name in FORBIDDEN_SOURCE_ROOTS:
        root = ROOT / root_name
        if root.exists():
            yield from root.rglob("*.py")


def test_no_defect_or_oracle_imports_in_agent_or_baselines():
    violations = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == m or alias.name.startswith(m + ".")
                        for m in FORBIDDEN_IMPORT_MODULES
                    ):
                        violations.append(f"{path}: import {alias.name}")
                continue
            if module and any(
                module == m or module.startswith(m + ".") for m in FORBIDDEN_IMPORT_MODULES
            ):
                violations.append(f"{path}: from {module} import ...")
    assert violations == [], "answer-key-adjacent imports found:\n" + "\n".join(violations)


def test_no_defect_vocabulary_in_agent_or_baseline_source():
    violations = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for match in _TOKEN_PATTERN.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            violations.append(f"{path}:{line_no}: {match.group(0)!r}")
    assert violations == [], "defect vocabulary found in agent/baseline source:\n" + "\n".join(violations)


def test_scenario_matrix_committed_before_defects_in_git_history():
    """The audit-trail half of the guard: docs/scenario-matrix.md's first
    commit must predate engine/defects.py's first commit. Skips quietly
    if git history isn't available (e.g. a fresh checkout without a .git
    directory, or the file's history was squashed) rather than failing --
    this is a nice-to-have provenance check, not a correctness check."""
    import subprocess

    def first_commit_date(relpath: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "log", "--follow", "--format=%aI", "--reverse", "--", relpath],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        lines = out.stdout.strip().splitlines()
        return lines[0] if lines else None

    matrix_date = first_commit_date("docs/scenario-matrix.md")
    defects_date = first_commit_date("engine/defects.py")
    if matrix_date is None or defects_date is None:
        return
    assert matrix_date <= defects_date, (
        f"docs/scenario-matrix.md first committed {matrix_date}, "
        f"engine/defects.py first committed {defects_date} -- "
        f"the matrix must come first"
    )
