"""Greedy-LLM baseline: the full agent minus planner and ledger readback."""

import agent.runner as runner
from agent.ledger import Ledger
from agent.llm import Cassette
from agent.outer_loop import run_session
from agent.runner import EpisodeConfig, run_episode
from baselines.greedy_llm import greedy_llm_config


def _session(tmp_path, **kw):
    config = greedy_llm_config("g", **{"episodes": 3, "turn_cap": 3, "allowed_scenarios": ["S1", "S2"], **kw})
    return run_session(
        config, runs_root=tmp_path, mode="replay", cassette=Cassette(dir_path=tmp_path / "empty"),
        progress=lambda _: None,
    )


def test_no_planner_calls_and_scenarios_rotate(tmp_path):
    summary = _session(tmp_path)
    assert summary["method"] == "greedy_llm"
    assert summary["episodes_completed"] == 3
    assert summary["planner_calls"] == 0 and summary["planner_fallbacks"] == 0
    assert summary["final_summary"] is None

    ledger = Ledger(tmp_path / "g" / "ledger.db")
    episodes = ledger.episodes()
    ledger.close()
    assert [e["scenario"] for e in episodes] == ["S1", "S2", "S1"]
    assert all(e["plan"]["goal"] == "" and e["plan"]["focus_actor_ids"] == [] for e in episodes)
    assert all(e["summary"]["planner_fallback"] is False for e in episodes)


def test_brief_ignores_the_ledger(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "ledger.db")
    hid = ledger.add_hypothesis("shield eats burn ticks", created_ep=None, status="confirmed")
    ep = ledger.start_episode("S1", 1, {}, str(tmp_path / "ep.jsonl"))
    seen = []
    real = runner.episode_brief
    monkeypatch.setattr(runner, "episode_brief", lambda engine, **kw: seen.append(kw) or real(engine, **kw))

    for memory in (True, False):
        run_episode(
            EpisodeConfig(scenario_id="S1", seed=1, turn_cap=1, hypothesis_ids=[hid], memory=memory),
            ledger=ledger, episode_id=ep, mode="replay", cassette=Cassette(dir_path=tmp_path / "empty"),
        )
    ledger.close()
    assert seen[0]["hypotheses"] and seen[0]["hypotheses"][0]["id"] == hid
    assert seen[1]["hypotheses"] == [] and seen[1]["confirmed"] == []
