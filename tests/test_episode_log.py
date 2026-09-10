"""Round-trip test for engine/episode_log.py: write a scripted episode to
JSONL, read it back, and confirm the header/turns/footer reflect what the
engine actually did."""
from __future__ import annotations

from engine.defects import DefectFlags
from engine.engine import BattleEngine
from engine.episode_log import read_episode_jsonl, write_episode_jsonl


def _run_first_legal(scenario_id: str, seed: int, defects: DefectFlags, turn_cap: int = 20) -> BattleEngine:
    engine = BattleEngine(scenario_id, seed, defects, turn_cap)
    initial_state = engine.state.model_copy(deep=True)
    while not engine.state.finished:
        legal = engine.legal_actions()
        if not legal:
            break
        engine.take_action(legal[0])
    return engine, initial_state


def test_write_and_read_round_trip(tmp_path):
    engine, initial_state = _run_first_legal("S1", seed=2, defects=DefectFlags())
    out_path = tmp_path / "ep_0001.jsonl"
    write_episode_jsonl(out_path, engine, initial_state, method="scripted_first_legal")

    header, turns, footer = read_episode_jsonl(out_path)

    assert header["scenario_id"] == "S1"
    assert header["seed"] == 2
    assert header["method"] == "scripted_first_legal"
    assert header["defects_enabled"] == []
    assert set(header["initial_state"]["characters"]) == set(initial_state.characters)

    assert len(turns) == len(engine.log)
    assert turns[0]["index"] == 0
    assert turns[0]["actor_id"] == engine.log[0].actor_id

    assert footer["finished"] == engine.state.finished
    assert footer["outcome"] == engine.state.outcome
    assert footer["step_count"] == engine.state.step_count


def test_missing_footer_raises(tmp_path):
    bad_path = tmp_path / "truncated.jsonl"
    bad_path.write_text('{"type": "header", "scenario_id": "S1"}\n', encoding="utf-8")
    try:
        read_episode_jsonl(bad_path)
        assert False, "expected ValueError for missing footer"
    except ValueError as exc:
        assert "footer" in str(exc)


def test_missing_header_raises(tmp_path):
    bad_path = tmp_path / "no_header.jsonl"
    bad_path.write_text('{"type": "footer", "finished": true}\n', encoding="utf-8")
    try:
        read_episode_jsonl(bad_path)
        assert False, "expected ValueError for missing header"
    except ValueError as exc:
        assert "header" in str(exc)
