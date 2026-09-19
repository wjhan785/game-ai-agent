"""agent/prompts.py: the rules spec covers every mechanic, and observations
report declared vs observed numbers faithfully."""

from engine.engine import BattleEngine
from engine.models import Action, StatusType
from agent.prompts import (
    RULES_SPEC,
    compact_legal,
    compact_state,
    format_roster,
    recent_events,
    summarize_record,
)
from tests.conftest import advance_to_actor


def test_rules_spec_covers_every_status_type():
    # The spec is meant to describe every mechanic at the same depth;
    # a status the spec never mentions would be untestable by reasoning.
    for st in StatusType:
        assert f"- {st.value.capitalize()}:" in RULES_SPEC, st


def test_rules_spec_multiplier_table_matches_the_engine():
    # Every non-neutral matchup the spec spells out must agree with the
    # engine's own table -- a spec that disagreed would manufacture reports.
    from engine.elements import elemental_multiplier
    from engine.models import Element

    for attacker in (Element.FIRE, Element.ICE, Element.LIGHTNING):
        for defender in (Element.FIRE, Element.ICE, Element.LIGHTNING):
            m = elemental_multiplier(attacker, defender)
            pair = f"{attacker.value}->{defender.value}"
            if m > 1:
                assert pair in RULES_SPEC.split("advantage, x1.5:")[1].split(".")[0], pair
            elif m < 1:
                assert pair in RULES_SPEC.split("disadvantage, x(1/1.5) = x0.667:")[1].split(".")[0], pair


def test_roster_lists_declared_ability_data():
    roster = format_roster(BattleEngine("S1", seed=0).state)
    assert "Cinder Burn: fire, cost 30, cd 2, one_enemy, power 10, applies burn 6 for 3t" in roster
    assert roster.index("p1 ") < roster.index("e1 ")


def test_shielded_hit_reports_tooltip_vs_taken_vs_absorbed(shield_test_scenario):
    # Dedicated 1v1 scenario: no other character exists to spend a filler
    # turn against p1's shield before this test's own explicit hit.
    eng = BattleEngine(shield_test_scenario, seed=0)
    eng.take_action(Action(actor_id="p1", ability_name="Aegis Ward", target_id="p1"))
    rec = eng.take_action(Action(actor_id="e1", ability_name="Basic Attack", target_id="p1"))
    effect = summarize_record(rec, eng.state)["effects"][0]
    assert effect == {
        "target": "p1",
        "tooltip_damage": 8,
        "damage_taken": 0,
        "shield_absorbed": 8,
        "hp": [effect["hp"][0], effect["hp"][0]],
        "statuses_after": ["shield:22/2/p1"],
    }


def test_energy_is_decomposed_into_start_regen_spent_end():
    eng = BattleEngine("S1", seed=0)
    advance_to_actor(eng, "p1")
    rec = eng.take_action(Action(actor_id="p1", ability_name="Aegis Ward", target_id="p1"))
    energy = summarize_record(rec, eng.state)["energy"]
    assert energy["spent"] == 25
    assert abs(energy["start"] + energy["regen"] - energy["spent"] - energy["end"]) < 0.01
    # Seeding can raise max energy above the starting energy; regen is capped by the headroom.
    assert energy["regen"] <= energy["max"] - energy["start"] + 1e-9


def test_disadvantaged_hit_reports_no_absorption():
    eng = BattleEngine("S2", seed=0)
    advance_to_actor(eng, "p1")
    rec = eng.take_action(Action(actor_id="p1", ability_name="Ember Slash", target_id="e2"))
    effect = summarize_record(rec, eng.state)["effects"][0]
    assert effect["tooltip_damage"] == 12 and effect["damage_taken"] == 12
    assert "shield_absorbed" not in effect


def test_recent_events_skip_uneventful_dead_turns_and_respect_window():
    eng = BattleEngine("S1", seed=1)
    for _ in range(10):
        eng.take_action(eng.legal_actions()[0])
    events = recent_events(eng.log, eng.state, window=4)
    assert len(events) == 4
    assert events[-1]["step"] == eng.log[-1].step_count


def test_recent_events_mark_and_include_everything_new():
    eng = BattleEngine("S1", seed=1)
    for _ in range(10):
        eng.take_action(eng.legal_actions()[0])
    events = recent_events(eng.log, eng.state, window=2, audit_from=3)
    steps = [ev["step"] for ev in events]
    assert steps == list(range(3, eng.log[-1].step_count + 1))  # widened to cover all new slots
    assert all(ev.get("new") for ev in events)
    older = recent_events(eng.log, eng.state, window=6, audit_from=9)
    assert [ev.get("new", False) for ev in older] == [ev["step"] >= 9 for ev in older]


def test_compact_state_and_legal():
    eng = BattleEngine("S1", seed=0)
    state = compact_state(eng.state)
    assert set(state["characters"]) == {"p1", "p2", "p3", "e1", "e2", "e3"}
    legal = compact_legal([
        {"ability_name": "Basic Attack", "target_id": "e1"},
        {"ability_name": "Basic Attack", "target_id": "e2"},
        {"ability_name": "Rallying Cry", "target_id": None},
    ])
    assert legal == {"Basic Attack": ["e1", "e2"], "Rallying Cry": ["-"]}
