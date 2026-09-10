"""Tests for agent/runner.py, offline via cassette replay mode. Focuses
on reasoning_by_log_index, which pairs each action-bearing TurnRecord
with the decision that produced it so replay/generate.py can show the
agent's stated reasoning per turn -- this was silently dropped until
fixed, so it gets its own regression test.
"""
from __future__ import annotations

from engine.defects import DefectFlags
from engine.engine import BattleEngine
from agent.runner import reasoning_by_log_index
from agent.tools import ToolDispatcher


def test_reasoning_by_log_index_pairs_actions_with_decisions_in_order():
    engine = BattleEngine("S1", seed=1, defects=DefectFlags(), turn_cap=20)
    dispatcher = ToolDispatcher(engine)

    decisions = []
    for i in range(3):
        options = dispatcher.list_legal_actions()["options"]
        actor_id = dispatcher.engine.state.current_actor_id()
        chosen = options[0]
        dispatcher.take_action(
            actor_id=actor_id,
            ability_name=chosen["ability_name"],
            target_id=chosen["target_id"],
            reasoning=f"reason-{i}",
        )
        decisions.append({"reasoning": f"reason-{i}"})

    extra = reasoning_by_log_index(engine, decisions)
    action_indices = [i for i, r in enumerate(engine.log) if r.action is not None]
    assert len(action_indices) == 3
    for i, idx in enumerate(action_indices):
        assert extra[idx]["agent_reasoning"] == f"reason-{i}"


def test_reasoning_by_log_index_skips_entries_with_no_reasoning():
    engine = BattleEngine("S1", seed=1, defects=DefectFlags(), turn_cap=20)
    dispatcher = ToolDispatcher(engine)
    options = dispatcher.list_legal_actions()["options"]
    actor_id = dispatcher.engine.state.current_actor_id()
    dispatcher.take_action(
        actor_id=actor_id, ability_name=options[0]["ability_name"], target_id=options[0]["target_id"], reasoning="x"
    )
    extra = reasoning_by_log_index(engine, [{"reasoning": None}])
    assert extra == {}


def test_reasoning_by_log_index_stops_when_decisions_exhausted():
    engine = BattleEngine("S1", seed=1, defects=DefectFlags(), turn_cap=20)
    dispatcher = ToolDispatcher(engine)
    for _ in range(2):
        options = dispatcher.list_legal_actions()["options"]
        actor_id = dispatcher.engine.state.current_actor_id()
        dispatcher.take_action(
            actor_id=actor_id, ability_name=options[0]["ability_name"], target_id=options[0]["target_id"], reasoning="x"
        )
    # Only one decision recorded even though two actions were taken --
    # must not raise, must not fabricate a pairing for the second.
    extra = reasoning_by_log_index(engine, [{"reasoning": "only-one"}])
    action_indices = [i for i, r in enumerate(engine.log) if r.action is not None]
    assert extra == {action_indices[0]: {"agent_reasoning": "only-one"}}
