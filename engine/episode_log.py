"""Episode JSONL: one header line (run metadata + initial state), one line
per turn, one footer line. Read back as plain dicts."""

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
    """Write the engine's log to `path`. `extra_by_index` adds fields to
    turn lines by their index in engine.log (e.g. the agent's reasoning)."""
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
