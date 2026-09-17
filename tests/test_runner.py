"""agent/runner.py, offline: gating, ledger bookkeeping, decision
annotations for the episode log, and a clean abort when the spend cap is
hit. Replay mode with an empty cassette makes every LLM decision fall back
to the first legal action -- deterministic and free."""
from __future__ import annotations

from engine.defects import single_flag
from engine.engine import BattleEngine
from engine.episode_log import read_episode_jsonl
from agent.ledger import Ledger
from agent.llm import Cassette, SpendTracker
from agent.runner import EpisodeConfig, MemoryNovelty, decision_annotations, run_episode, write_episode
from agent.tools import ToolDispatcher


def _replay(tmp_path):
    return dict(mode="replay", cassette=Cassette(dir_path=tmp_path / "empty_cassette"))


def test_every_decision_asks_the_model_without_gating(tmp_path):
    result = run_episode(EpisodeConfig(scenario_id="S1", seed=1, turn_cap=3, gating=False), **_replay(tmp_path))
    assert result.decisions
    assert all(d["mode"] == "fallback" for d in result.decisions)


def test_gating_scripts_turns_on_already_visited_states(tmp_path):
    novelty = MemoryNovelty()
    config = EpisodeConfig(scenario_id="S1", seed=1, turn_cap=3)
    first = run_episode(config, novelty=novelty, **_replay(tmp_path))
    assert first.decisions[0]["mode"] == "fallback"  # novel state -> model asked (and fell back)
    second = run_episode(config, novelty=novelty, **_replay(tmp_path))
    assert second.decisions[0]["mode"] == "scripted"  # same opening state, already visited


def test_focus_characters_always_get_the_model(tmp_path):
    novelty = MemoryNovelty()
    run_episode(EpisodeConfig(scenario_id="S1", seed=1, turn_cap=3), novelty=novelty, **_replay(tmp_path))
    focused = run_episode(
        EpisodeConfig(scenario_id="S1", seed=1, turn_cap=3, focus_actor_ids=["p1"]), novelty=novelty, **_replay(tmp_path)
    )
    # p1 need not be the first actor (scenarios carry varied speeds) -- but
    # whichever decision is p1's must go to the model despite the state
    # already being visited, since focus overrides novelty-gating.
    p1_decision = next(d for d in focused.decisions if d["actor_id"] == "p1")
    assert p1_decision["mode"] == "fallback"


def test_scripted_policy_is_deterministic(tmp_path):
    def run():
        novelty = MemoryNovelty()
        config = EpisodeConfig(scenario_id="S2", seed=4, turn_cap=4)
        run_episode(config, novelty=novelty, **_replay(tmp_path))
        return [r.action for r in run_episode(config, novelty=novelty, **_replay(tmp_path)).engine.log]

    assert run() == run()


def test_ledger_bookkeeping_records_invariant_flags_coverage_and_visits(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    ep = ledger.start_episode("S5", 1, {}, "p")
    result = run_episode(
        EpisodeConfig(scenario_id="S5", seed=1, turn_cap=12, gating=False),
        defects=single_flag("B08"), ledger=ledger, episode_id=ep, run_dir=tmp_path, **_replay(tmp_path),
    )
    invariant = ledger.flags(source="invariant", episode=ep)
    assert any("no-progress" in f["description"] for f in invariant)
    assert ledger.visit_stats()["total_visits"] == len(result.decisions)
    assert len(ledger.coverage()) == len(result.new_coverage)  # fresh ledger: everything seen is new


def test_invariant_flags_are_deduplicated_per_kind(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    ep = ledger.start_episode("S5", 1, {}, "p")
    run_episode(
        EpisodeConfig(scenario_id="S5", seed=1, turn_cap=20, gating=False),
        defects=single_flag("B08"), ledger=ledger, episode_id=ep, run_dir=tmp_path, **_replay(tmp_path),
    )
    from agent.ledger import invariant_kind

    kinds = [invariant_kind(f["description"]) for f in ledger.flags(source="invariant", episode=ep)]
    assert len(kinds) == len(set(kinds))


def test_decision_annotations_pair_actions_with_decisions_in_order():
    engine = BattleEngine("S1", seed=1)
    dispatcher = ToolDispatcher(engine)
    decisions = []
    for i in range(3):
        opt = dispatcher.list_legal_actions()["options"][0]
        dispatcher.take_action(
            actor_id=engine.state.current_actor_id(), ability_name=opt["ability_name"],
            target_id=opt["target_id"], reasoning=f"reason-{i}",
        )
        decisions.append({"mode": "llm", "reasoning": f"reason-{i}", "flags": [{"description": "d"}] if i == 1 else []})

    extra = decision_annotations(engine, decisions)
    idx = [i for i, r in enumerate(engine.log) if r.action is not None]
    assert [extra[i]["agent_reasoning"] for i in idx] == ["reason-0", "reason-1", "reason-2"]
    assert extra[idx[1]]["agent_flags"] == [{"description": "d"}]
    assert all(extra[i]["decision_mode"] == "llm" for i in idx)


def test_decision_annotations_stop_when_decisions_run_out():
    engine = BattleEngine("S1", seed=1)
    dispatcher = ToolDispatcher(engine)
    for _ in range(2):
        opt = dispatcher.list_legal_actions()["options"][0]
        dispatcher.take_action(actor_id=engine.state.current_actor_id(), ability_name=opt["ability_name"],
                               target_id=opt["target_id"], reasoning="x")
    extra = decision_annotations(engine, [{"mode": "scripted", "reasoning": None}])
    first_action = next(i for i, r in enumerate(engine.log) if r.action is not None)
    assert extra == {first_action: {"decision_mode": "scripted"}}


def test_write_episode_carries_annotations(tmp_path):
    result = run_episode(EpisodeConfig(scenario_id="S1", seed=1, turn_cap=2, gating=False), **_replay(tmp_path))
    path = write_episode(result, tmp_path / "ep.jsonl", method="test")
    _header, turns, _footer = read_episode_jsonl(path)
    modes = [t.get("decision_mode") for t in turns if t.get("action")]
    assert modes and all(m == "fallback" for m in modes)


def test_spend_cap_aborts_cleanly_before_any_call(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    spend = SpendTracker(path=tmp_path / "spend.json", max_session_spend=0.0)
    result = run_episode(
        EpisodeConfig(scenario_id="S1", seed=1, turn_cap=3, gating=False),
        mode="live", spend=spend, allow_peak=True,
    )
    assert result.aborted_reason is not None and "cap" in result.aborted_reason
    assert result.decisions == []
    assert not result.engine.state.finished
    assert spend.total_usd == 0.0
