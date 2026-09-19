"""Differential oracle and single-flag attribution: a trace replays exactly
under its own flags, and attribution names the right defect."""

from engine.defects import ALL_DEFECT_IDS, DefectFlags, single_flag
from engine.engine import BattleEngine
from engine.models import Ability, Character, Element, TargetType
from engine import scenarios as scenarios_module
from oracle.attribute import attribute
from oracle.differential import replay_and_diff

TURN_CAP = 20
SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def _run_first_legal(scenario_id: str, seed: int, defects: DefectFlags, turn_cap: int = TURN_CAP):
    """Deterministic scripted policy: always take the first legal action
    (stable order from rules.list_legal_actions), run to completion."""
    engine = BattleEngine(scenario_id, seed, defects, turn_cap)
    while not engine.state.finished:
        legal = engine.legal_actions()
        if not legal:
            break
        engine.take_action(legal[0])
    return engine.log


def test_clean_trace_shows_zero_divergence_against_clean_replay():
    for scenario_id in SCENARIOS:
        log = _run_first_legal(scenario_id, seed=1, defects=DefectFlags())
        result = replay_and_diff(scenario_id, 1, TURN_CAP, log, DefectFlags())
        assert result.diverged is False, scenario_id
        assert result.divergence is None, scenario_id


def test_replay_reproduces_original_log_exactly_under_same_defects():
    for defect_id in ALL_DEFECT_IDS:
        flags = single_flag(defect_id)
        for scenario_id in ("S1", "S5"):
            log = _run_first_legal(scenario_id, seed=3, defects=flags)
            result = replay_and_diff(scenario_id, 3, TURN_CAP, log, flags)
            assert result.diverged is False, (
                f"{defect_id}/{scenario_id}: replay under its own flags must reproduce exactly "
                f"(got divergence: {result.divergence})"
            )


def _make_b06_1v1_scenario():
    def make_1v1(seed, defects):
        atk = Character(
            id="atk", name="Attacker", team="party", element=Element.NEUTRAL,
            max_hp=200, hp=200, max_energy=200, energy=200, energy_regen=20,
            abilities=[
                Ability(name="Basic Attack", element=Element.NEUTRAL, energy_cost=0,
                        cooldown=0, target_type=TargetType.ONE_ENEMY, base_power=1.0,
                        is_basic_attack=True),
                Ability(name="Big Hit", element=Element.NEUTRAL, energy_cost=10,
                        cooldown=2, target_type=TargetType.ONE_ENEMY, base_power=5.0),
            ],
        )
        df = Character(
            id="def", name="Defender", team="enemy", element=Element.NEUTRAL,
            max_hp=200, hp=200, max_energy=200, energy=200, energy_regen=20,
            abilities=[
                Ability(name="Basic Attack", element=Element.NEUTRAL, energy_cost=0,
                        cooldown=0, target_type=TargetType.ONE_ENEMY, base_power=1.0,
                        is_basic_attack=True),
            ],
        )
        return [atk, df], ["atk", "def"]

    return make_1v1


def test_attribution_names_b06_from_a_full_episode_trace():
    scenarios_module.SCENARIOS["_T_B06_ORACLE"] = _make_b06_1v1_scenario()
    try:
        defects = single_flag("B06")
        eng = BattleEngine("_T_B06_ORACLE", seed=0, defects=defects, turn_cap=30)
        atk_turns = 0
        while atk_turns < 3 and not eng.state.finished:
            legal = eng.legal_actions()
            if not legal:
                break
            actor = eng.state.current_actor_id()
            if actor == "atk":
                names = {a.ability_name for a in legal}
                action = (
                    next(a for a in legal if a.ability_name == "Big Hit")
                    if "Big Hit" in names
                    else next(a for a in legal if a.actor_id == "atk")
                )
                eng.take_action(action)
                atk_turns += 1
            else:
                eng.take_action(legal[0])

        result = attribute("_T_B06_ORACLE", seed=0, turn_cap=30, original_log=eng.log)
        assert result.triggered is True
        assert "B06" in result.matching_defects
    finally:
        del scenarios_module.SCENARIOS["_T_B06_ORACLE"]


def test_attribution_names_b08_from_a_full_episode_trace():
    log = _run_first_legal("S5", seed=1, defects=single_flag("B08"), turn_cap=30)
    result = attribute("S5", seed=1, turn_cap=30, original_log=log)
    assert result.triggered is True
    assert "B08" in result.matching_defects


def test_attribution_matching_defects_is_consistent_with_per_defect_results():
    log = _run_first_legal("S5", seed=1, defects=single_flag("B08"), turn_cap=30)
    result = attribute("S5", seed=1, turn_cap=30, original_log=log)
    for defect_id in ALL_DEFECT_IDS:
        assert (defect_id in result.matching_defects) == (not result.per_defect[defect_id].diverged)
