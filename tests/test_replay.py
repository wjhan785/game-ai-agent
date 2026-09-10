"""replay/generate.py: run a scripted episode, write it to JSONL, render
it to HTML, and sanity-check the output contains what the episode
actually did -- this is the "read every turn against the raw JSONL by
hand" check from the project plan's verification list, automated enough
to catch a regression without a human reading it every time."""
from __future__ import annotations

from engine.defects import DefectFlags, single_flag
from engine.engine import BattleEngine
from engine.episode_log import write_episode_jsonl
from replay.generate import build_view_model, generate_replay_html
from engine.episode_log import read_episode_jsonl


def _run_first_legal(scenario_id: str, seed: int, defects: DefectFlags, turn_cap: int = 20):
    engine = BattleEngine(scenario_id, seed, defects, turn_cap)
    initial_state = engine.state.model_copy(deep=True)
    while not engine.state.finished:
        legal = engine.legal_actions()
        if not legal:
            break
        engine.take_action(legal[0])
    return engine, initial_state


def test_view_model_reflects_engine_log(tmp_path):
    engine, initial_state = _run_first_legal("S1", seed=4, defects=DefectFlags())
    jsonl_path = tmp_path / "ep.jsonl"
    write_episode_jsonl(jsonl_path, engine, initial_state, method="scripted_first_legal")

    header, turns, footer = read_episode_jsonl(jsonl_path)
    vm = build_view_model(header, turns, footer)

    assert vm["meta"]["scenario_id"] == "S1"
    assert vm["meta"]["total_turns"] == len(engine.log)
    assert len(vm["roster"]) == len(initial_state.characters)
    # Every actor named in a turn view must resolve to a real roster name,
    # not fall back to the raw character id (proves the legend lookup wired
    # correctly end to end).
    roster_names = {c["name"] for c in vm["roster"]}
    for t in vm["turns"]:
        assert t["actor_name"] in roster_names


def test_generate_replay_html_produces_a_self_contained_file(tmp_path):
    engine, initial_state = _run_first_legal("S5", seed=1, defects=single_flag("B08"), turn_cap=30)
    jsonl_path = tmp_path / "ep.jsonl"
    write_episode_jsonl(jsonl_path, engine, initial_state, method="scripted_first_legal")

    out_path = tmp_path / "ep.html"
    result_path = generate_replay_html(jsonl_path, out_path)

    assert result_path == out_path
    html = out_path.read_text(encoding="utf-8")
    assert "<html" in html
    assert "S5" in html
    assert "B08" in html  # defect badge rendered
    # Every character name from the initial roster should appear somewhere
    # (roster legend section at minimum).
    for c in initial_state.characters.values():
        assert c.name in html
    # No unresolved Jinja2 template syntax leaked into the output.
    assert "{{" not in html
    assert "{%" not in html
