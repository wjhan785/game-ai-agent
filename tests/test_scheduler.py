"""The action-value scheduler: determinism, equal-speed order, tie-break,
forecast and the AV invariant. Uses a uniform-speed scenario where needed."""

import pytest

from engine.content import make_character
from engine.defects import DefectFlags
from engine.engine import BattleEngine
from engine.invariants import _check_action_values
from engine.models import BASE_ACTION_VALUE, CYCLE_AV, Element
from engine import scenarios as scenarios_module

OLD_FIXED_ORDER = ["p1", "e1", "p2", "e2", "p3", "e3"]
SCENARIO_IDS = ["S1", "S2", "S3", "S4", "S5", "S6"]
UNIFORM_SCENARIO_ID = "_T_UNIFORM_SPEED"


def _build_uniform_speed_scenario(seed, defects):
    party = [make_character(cid, cid, "party", Element.NEUTRAL, 100, 50, 20, []) for cid in ("p1", "p2", "p3")]
    enemy = [make_character(cid, cid, "enemy", Element.NEUTRAL, 100, 50, 20, []) for cid in ("e1", "e2", "e3")]
    return party + enemy, scenarios_module._interleave(party, enemy)


@pytest.fixture
def uniform_speed_scenario():
    """A 3v3 battle where every character is at the default speed (100) --
    the ground truth for 'ties break by declared turn order', independent
    of any real scenario's own speed balance."""
    scenarios_module.SCENARIOS[UNIFORM_SCENARIO_ID] = _build_uniform_speed_scenario
    try:
        yield UNIFORM_SCENARIO_ID
    finally:
        del scenarios_module.SCENARIOS[UNIFORM_SCENARIO_ID]


def _actor_sequence(scenario_id: str, seed: int, defects: DefectFlags, n: int) -> list[str]:
    engine = BattleEngine(scenario_id, seed, defects, turn_cap=30)
    actors = []
    for _ in range(n):
        if engine.state.finished or not engine.legal_actions():
            break
        actors.append(engine.state.current_actor_id())
        engine.take_action(engine.legal_actions()[0])
    return actors


def test_equal_speed_reproduces_the_old_fixed_interleave(uniform_speed_scenario):
    # At uniform speed the order is the declared cycle, every cycle.
    actors = _actor_sequence(uniform_speed_scenario, seed=3, defects=DefectFlags(), n=18)
    assert actors == (OLD_FIXED_ORDER * 3)[: len(actors)]


def test_replay_is_deterministic():
    a = _actor_sequence("S4", seed=11, defects=DefectFlags(), n=40)
    b = _actor_sequence("S4", seed=11, defects=DefectFlags(), n=40)
    assert a == b


def test_current_actor_id_ties_break_by_turn_order_position(uniform_speed_scenario):
    engine = BattleEngine(uniform_speed_scenario, seed=0, defects=DefectFlags())
    # All six start tied; declared order is p1,e1,p2,e2,p3,e3.
    assert engine.state.current_actor_id() == "p1"
    engine.state.characters["p1"].action_value = 999.0  # take p1 out of contention
    assert engine.state.current_actor_id() == "e1"


def test_turn_forecast_matches_actual_play_at_equal_speed(uniform_speed_scenario):
    engine = BattleEngine(uniform_speed_scenario, seed=0, defects=DefectFlags())
    forecast_ids = [cid for cid, _av in engine.state.turn_forecast(6)]
    actual = _actor_sequence(uniform_speed_scenario, seed=0, defects=DefectFlags(), n=6)
    assert forecast_ids == actual == OLD_FIXED_ORDER


def test_turn_forecast_excludes_the_dead():
    engine = BattleEngine("S1", seed=0, defects=DefectFlags())
    engine.state.characters["e1"].alive = False
    forecast_ids = [cid for cid, _av in engine.state.turn_forecast(6)]
    assert "e1" not in forecast_ids


def test_round_number_matches_old_semantics_at_equal_speed(uniform_speed_scenario):
    # Round 0 lasts until the first character comes up again.
    engine = BattleEngine(uniform_speed_scenario, seed=0, defects=DefectFlags())
    for i in range(6):
        assert engine.state.round_number == 0, i
        engine.take_action(engine.legal_actions()[0])
    assert engine.state.round_number == 1


def test_action_value_invariant_clean_across_all_scenarios():
    for scenario_id in SCENARIO_IDS:
        engine = BattleEngine(scenario_id, seed=7, defects=DefectFlags())
        for _ in range(30):
            if engine.state.finished or not engine.legal_actions():
                break
            engine.take_action(engine.legal_actions()[0])
            assert _check_action_values(engine.state) == [], scenario_id


def test_action_value_invariant_flags_a_stale_clock():
    engine = BattleEngine("S1", seed=0, defects=DefectFlags())
    engine.state.elapsed_av += 50  # clock jumped ahead without any character catching up
    violations = _check_action_values(engine.state)
    assert any("behind the scheduler clock" in v for v in violations)


def test_action_value_invariant_flags_nonpositive_speed():
    engine = BattleEngine("S1", seed=0, defects=DefectFlags())
    engine.state.characters["p1"].speed = 0
    assert any("speed is not positive" in v for v in _check_action_values(engine.state))


def test_action_value_invariant_never_encodes_which_actor_is_correct():
    # B11 (wrong tie-break) must NOT be catchable by this invariant --
    # selection conformance is the oracle's job, not this checker's.
    from engine.defects import single_flag

    engine = BattleEngine("S1", seed=0, defects=single_flag("B11"))
    assert _check_action_values(engine.state) == []


def test_cycle_constants_make_speed_100_the_reference():
    assert BASE_ACTION_VALUE / 100.0 == CYCLE_AV
