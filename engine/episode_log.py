"""Episode JSONL logging: serialize a completed (or in-progress) episode
to a JSON-Lines file, and read it back.

One file, three kinds of line, one JSON object per line:
  1. exactly one "header" line -- run metadata (method, scenario_id, seed,
     defects enabled) plus the full initial BattleState (so ability
     tooltips, element, cooldowns, etc. are available to a reader without
     re-running the engine).
  2. one "turn" line per TurnRecord in engine.log, in order.
  3. exactly one "footer" line -- whether the battle finished and how.

This format, not a single JSON document, is deliberate: a crash mid-sweep
leaves every completed line intact and readable, and eval/harness.py can
append lines as an episode runs rather than buffering the whole thing in
memory. Consumers (replay/generate.py, eval/harness.py) read this back as
plain dicts, not re-parsed into TurnRecord/BattleState -- a reader may
only need a handful of fields and shouldn't have to satisfy every
required field of those models against an older or hand-trimmed log.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.engine import BattleEngine
from engine.models import BattleState


def write_episode_jsonl(
    path: str | Path,
    engine: BattleEngine,
    initial_state: BattleState,
    method: str,
    extra_by_index: dict[int, dict] | None = None,
) -> None:
    """Write `engine`'s full log (and `initial_state`, captured right
    after reset() -- BEFORE any turns ran) to `path` as episode JSONL.

    `extra_by_index` merges caller-supplied fields into specific turn
    lines by their position in `engine.log` -- e.g. agent/runner.py
    attaches the agent's own stated reasoning for each decision this way,
    without this module (shared by any method, including non-agent
    baselines) needing to know that concept exists.
    """
    extra_by_index = extra_by_index or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        header = {
            "type": "header",
            "method": method,
            "scenario_id": engine.state.scenario_id,
            "seed": engine.state.seed,
            "defects_enabled": engine.defects.enabled(),
            "turn_cap": engine.turn_cap,
            "initial_state": initial_state.model_dump(mode="json"),
        }
        f.write(json.dumps(header) + "\n")
        for i, record in enumerate(engine.log):
            line = {"type": "turn", "index": i, **record.model_dump(mode="json"), **extra_by_index.get(i, {})}
            f.write(json.dumps(line) + "\n")
        footer = {
            "type": "footer",
            "finished": engine.state.finished,
            "outcome": engine.state.outcome,
            "step_count": engine.state.step_count,
        }
        f.write(json.dumps(footer) + "\n")


def read_episode_jsonl(path: str | Path) -> tuple[dict, list[dict], dict]:
    """Returns (header, turn_lines, footer) as plain dicts."""
    header: dict | None = None
    footer: dict | None = None
    turns: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            obj = json.loads(raw_line)
            kind = obj.get("type")
            if kind == "header":
                header = obj
            elif kind == "turn":
                turns.append(obj)
            elif kind == "footer":
                footer = obj
    if header is None:
        raise ValueError(f"{path}: missing header line")
    if footer is None:
        raise ValueError(f"{path}: missing footer line (episode log incomplete?)")
    return header, turns, footer
