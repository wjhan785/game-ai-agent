"""agent/ledger.py: schema, each table's API, the lossy state signature,
coverage combos, and the planner digest -- including that grading-time
columns never reach anything the agent reads."""

import json

from engine.engine import BattleEngine
from engine.models import Action
from agent.ledger import (
    ALL_STATUS_PAIRS,
    Ledger,
    action_signature,
    invariant_kind,
    record_coverage_combos,
    state_signature,
    status_pairs_from_combos,
)
from tests.conftest import advance_to_actor


def _ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "ledger.db")


def test_episode_lifecycle_and_total_actions(tmp_path):
    ledger = _ledger(tmp_path)
    ep = ledger.start_episode("S1", 3, {"goal": "g"}, "logs/x.jsonl")
    ledger.finish_episode(
        ep, outcome="party_win", actions_used=40, llm_calls=12, tokens_in=1000,
        tokens_out=100, cache_hit_tokens=600, cost_usd=0.01, summary={"k": 1},
    )
    ep2 = ledger.start_episode("S2", 4, {}, "logs/y.jsonl")
    ledger.finish_episode(
        ep2, outcome=None, actions_used=10, llm_calls=0, tokens_in=0,
        tokens_out=0, cache_hit_tokens=0, cost_usd=0.0, summary={},
    )
    rows = ledger.episodes()
    assert [r["id"] for r in rows] == [ep, ep2]
    assert rows[0]["plan"] == {"goal": "g"}
    assert rows[0]["summary"] == {"k": 1}
    assert ledger.total_actions() == 50


def test_hypothesis_add_update_and_filter(tmp_path):
    ledger = _ledger(tmp_path)
    h = ledger.add_hypothesis("burn from two sources may not sum", created_ep=1)
    assert ledger.hypothesis_exists(h)
    assert ledger.update_hypothesis(h, "confirmed", "tick was 30, expected 11", episode=2)
    row = ledger.hypotheses()[0]
    assert row["status"] == "confirmed"
    assert row["resolved_ep"] == 2
    assert "[ep 2] tick was 30" in row["notes"]
    assert ledger.hypotheses(("open",)) == []
    assert ledger.update_hypothesis(999, "refuted", "", episode=2) is False


def test_unknown_hypothesis_status_rejected(tmp_path):
    ledger = _ledger(tmp_path)
    try:
        ledger.add_hypothesis("x", created_ep=None, status="maybe")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_flags_never_expose_grading_columns(tmp_path):
    ledger = _ledger(tmp_path)
    fid = ledger.add_flag(episode=1, turn=5, source="agent", description="d", severity="high", evidence="e")
    # Simulate a grading pass writing its verdict into the same row.
    ledger._conn.execute("UPDATE flags SET adjudication='tp', bug_id='X' WHERE id=?", (fid,))
    for row in ledger.flags():
        assert "adjudication" not in row and "bug_id" not in row
    assert "adjudication" not in json.dumps(ledger.digest())
    assert "bug_id" not in json.dumps(ledger.digest())


def test_flags_filter_by_source_and_episode(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.add_flag(episode=1, turn=1, source="agent", description="a")
    ledger.add_flag(episode=1, turn=2, source="invariant", description="p1: energy went negative (-3.0)")
    ledger.add_flag(episode=2, turn=1, source="agent", description="b")
    assert [f["description"] for f in ledger.flags(source="agent")] == ["a", "b"]
    assert len(ledger.flags(episode=1)) == 2


def test_visits_and_novelty(tmp_path):
    ledger = _ledger(tmp_path)
    assert ledger.seen("sig") is False
    ledger.record_visit("sig", "Basic Attack->e1", episode=1)
    ledger.record_visit("sig", "Basic Attack->e1", episode=2)
    assert ledger.seen("sig") is True
    assert ledger.visit_stats() == {"distinct_states": 1, "distinct_state_actions": 1, "total_visits": 2}


def test_coverage_records_first_sighting_only(tmp_path):
    ledger = _ledger(tmp_path)
    assert ledger.record_coverage("set:burn+shield", episode=1, action_idx=7) is True
    assert ledger.record_coverage("set:burn+shield", episode=2, action_idx=90) is False
    row = ledger.coverage()[0]
    assert (row["first_seen_ep"], row["first_seen_action_idx"]) == (1, 7)


def test_state_signature_is_deterministic_and_lossy():
    a = BattleEngine("S1", seed=1)
    b = BattleEngine("S1", seed=1)
    assert state_signature(a.state) == state_signature(b.state)
    # A 1-HP scratch on a full-HP character stays in the same bucket.
    b.state.characters["e1"].hp -= 1.0
    assert state_signature(a.state) == state_signature(b.state)
    # Dropping below 75% moves it.
    b.state.characters["e1"].hp = b.state.characters["e1"].max_hp * 0.5
    assert state_signature(a.state) != state_signature(b.state)


def test_state_signature_distinguishes_scenarios():
    assert state_signature(BattleEngine("S1", seed=1).state) != state_signature(BattleEngine("S2", seed=1).state)


def test_action_signature():
    assert action_signature("Aegis Ward", "p1") == "Aegis Ward->p1"
    assert action_signature("Rallying Cry", None) == "Rallying Cry->-"


def test_record_coverage_combos_from_real_turn():
    eng = BattleEngine("S1", seed=0)
    advance_to_actor(eng, "p1")
    eng.take_action(Action(actor_id="p1", ability_name="Aegis Ward", target_id="p1"))
    advance_to_actor(eng, "e1")
    rec = eng.take_action(Action(actor_id="e1", ability_name="Cinder Burn", target_id="p1"))
    combos = record_coverage_combos(rec, ability_element="fire")
    assert "set:burn+shield" in combos
    assert "pair:burn|fire" in combos


def test_multiple_instances_of_a_status_are_a_coverage_item():
    eng = BattleEngine("S1", seed=0)
    advance_to_actor(eng, "p1")
    eng.take_action(Action(actor_id="p1", ability_name="Basic Attack", target_id="e1"))
    advance_to_actor(eng, "e1")
    eng.take_action(Action(actor_id="e1", ability_name="Cinder Burn", target_id="p1"))
    advance_to_actor(eng, "p2")
    eng.take_action(Action(actor_id="p2", ability_name="Basic Attack", target_id="e1"))
    advance_to_actor(eng, "e2")
    rec = eng.take_action(Action(actor_id="e2", ability_name="Flame Lash", target_id="p1"))
    assert "stack:burn" in record_coverage_combos(rec, ability_element="fire")


def test_status_pairs_from_combos():
    pairs = status_pairs_from_combos(["set:burn+poison+shield", "pair:burn|fire", "set:stun"])
    assert pairs == {("burn", "poison"), ("burn", "shield"), ("poison", "shield")}
    assert len(ALL_STATUS_PAIRS) == 21


def test_invariant_kind_normalization():
    assert invariant_kind("e1: energy went negative (-5.0)") == "energy went negative"
    assert invariant_kind("p2: used Cinder Burn while it had 1 turn(s) of cooldown remaining") == (
        "used Cinder Burn while it had turn(s) of cooldown remaining"
    )
    assert invariant_kind(
        "e1: 2 simultaneous stun entries (non-stacking status should have at most 1)"
    ) == "simultaneous stun entries"


def test_digest_shape_and_coverage_gaps(tmp_path):
    ledger = _ledger(tmp_path)
    ep = ledger.start_episode("S1", 1, {"goal": "shield vs burn"}, "p")
    ledger.record_coverage("set:burn+shield", ep, 3)
    ledger.add_hypothesis("h1", created_ep=ep)
    ledger.add_flag(episode=ep, turn=3, source="invariant", description="p1: energy went negative (-1.0)")
    ledger.add_flag(episode=ep, turn=9, source="invariant", description="p2: energy went negative (-4.0)")
    ledger.finish_episode(
        ep, outcome="turn_cap", actions_used=30, llm_calls=5, tokens_in=0, tokens_out=0,
        cache_hit_tokens=0, cost_usd=0.0, summary={"agent_flags": []},
    )
    d = ledger.digest()
    assert d["episodes_run"] == 1
    assert d["last_episode"]["plan"] == {"goal": "shield vs burn"}
    assert d["hypotheses_active"][0]["text"] == "h1"
    assert d["invariant_violation_kinds"] == [{"kind": "energy went negative", "episodes": 1, "first_episode": ep}]
    assert "burn+shield" in d["coverage"]["status_pairs_seen_together"]
    assert "burn+shield" not in d["coverage"]["status_pairs_never_seen_together"]
    assert len(d["coverage"]["status_pairs_never_seen_together"]) == 20


def test_ledger_persists_across_connections(tmp_path):
    path = tmp_path / "ledger.db"
    first = Ledger(path)
    first.add_hypothesis("persisted", created_ep=None)
    first.close()
    second = Ledger(path)
    assert second.hypotheses()[0]["text"] == "persisted"
