"""Golden tests: hand-calculated values in clean mode. If these fail, fix
the engine, not the test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.defects import DefectFlags
from engine.effects import (
    apply_energy_regen,
    apply_shield_absorption,
    apply_status,
    decrement_cooldowns,
    tick_burn,
    tick_poison,
)
from engine.elements import ADVANTAGE, DISADVANTAGE, elemental_multiplier
from engine.engine import BattleEngine
from engine.models import (
    Ability,
    Character,
    Element,
    StatusType,
    TargetType,
)

CLEAN = DefectFlags()


def make_target(**overrides) -> Character:
    defaults = dict(
        id="t1",
        name="Target",
        team="enemy",
        element=Element.NEUTRAL,
        max_hp=100.0,
        hp=100.0,
        max_energy=100.0,
        energy=50.0,
        energy_regen=20.0,
        abilities=[],
    )
    defaults.update(overrides)
    return Character(**defaults)


# --- Elemental triangle ----------------------------------------------------

def test_elemental_triangle_advantage():
    assert elemental_multiplier(Element.FIRE, Element.ICE) == ADVANTAGE
    assert elemental_multiplier(Element.ICE, Element.LIGHTNING) == ADVANTAGE
    assert elemental_multiplier(Element.LIGHTNING, Element.FIRE) == ADVANTAGE


def test_elemental_triangle_disadvantage():
    assert elemental_multiplier(Element.ICE, Element.FIRE) == DISADVANTAGE
    assert elemental_multiplier(Element.FIRE, Element.LIGHTNING) == DISADVANTAGE
    assert elemental_multiplier(Element.LIGHTNING, Element.ICE) == DISADVANTAGE


def test_elemental_neutral_is_always_one():
    assert elemental_multiplier(Element.FIRE, Element.FIRE) == 1.0
    assert elemental_multiplier(Element.NEUTRAL, Element.FIRE) == 1.0
    assert elemental_multiplier(Element.FIRE, Element.NEUTRAL) == 1.0


# --- Poison ------------------------------------------------------------

def test_poison_is_percent_of_max_hp():
    target = make_target(max_hp=100.0, hp=100.0)
    apply_status(target, StatusType.POISON, magnitude=0.1, duration=3, source_id="src", defects=CLEAN)
    damage = tick_poison(target, CLEAN)
    assert damage == 10.0
    assert target.hp == 90.0


def test_poison_ignores_weaken_in_clean_mode():
    """Golden pair for defect B04 -- see test_defects.py for the buggy side."""
    target = make_target(max_hp=100.0, hp=100.0)
    apply_status(target, StatusType.POISON, magnitude=0.1, duration=3, source_id="src", defects=CLEAN)
    apply_status(target, StatusType.WEAKEN, magnitude=0.4, duration=3, source_id="src2", defects=CLEAN)
    damage = tick_poison(target, CLEAN)
    assert damage == 10.0  # NOT 0.1 * (100 * 0.6) = 6.0


# --- Burn ----------------------------------------------------------------

def test_burn_stacks_additively_across_sources():
    """Golden pair for defect B02."""
    target = make_target(hp=100.0)
    apply_status(target, StatusType.BURN, magnitude=5.0, duration=3, source_id="a", defects=CLEAN)
    apply_status(target, StatusType.BURN, magnitude=7.0, duration=3, source_id="b", defects=CLEAN)
    damage = tick_burn(target, CLEAN)
    assert damage == 12.0  # NOT 5 * 7 = 35
    assert target.hp == 88.0


# --- Shield ----------------------------------------------------------------

def test_shield_absorbs_partial_direct_damage():
    target = make_target()
    apply_status(target, StatusType.SHIELD, magnitude=20.0, duration=3, source_id="s", defects=CLEAN)
    remaining = apply_shield_absorption(target, 15.0, CLEAN)
    assert remaining == 0.0
    shield = next(s for s in target.statuses if s.status_type == StatusType.SHIELD)
    assert shield.magnitude == 5.0


def test_shield_overflow_and_removed_at_zero():
    """Golden pair for defect B10."""
    target = make_target()
    apply_status(target, StatusType.SHIELD, magnitude=20.0, duration=3, source_id="s", defects=CLEAN)
    remaining = apply_shield_absorption(target, 25.0, CLEAN)
    assert remaining == 5.0
    assert not any(s.status_type == StatusType.SHIELD for s in target.statuses)


def test_shield_does_not_absorb_dot_ticks():
    """Golden pair for defect B01."""
    target = make_target(hp=100.0)
    apply_status(target, StatusType.SHIELD, magnitude=50.0, duration=3, source_id="s", defects=CLEAN)
    apply_status(target, StatusType.BURN, magnitude=10.0, duration=3, source_id="b", defects=CLEAN)
    tick_burn(target, CLEAN)
    assert target.hp == 90.0  # DoT went straight through
    shield = next(s for s in target.statuses if s.status_type == StatusType.SHIELD)
    assert shield.magnitude == 50.0  # untouched


# --- Chill / energy regen ---------------------------------------------------

def test_chill_floors_at_zero_energy():
    """Golden pair for defect B03."""
    target = make_target(energy=5.0, max_energy=100.0, energy_regen=20.0)
    apply_status(target, StatusType.CHILL, magnitude=1.5, duration=2, source_id="c", defects=CLEAN)
    apply_energy_regen(target, CLEAN)
    assert target.energy == 5.0  # would be -5 without the floor


def test_energy_regen_clamps_to_max():
    target = make_target(energy=95.0, max_energy=100.0, energy_regen=20.0)
    apply_energy_regen(target, CLEAN)
    assert target.energy == 100.0


# --- Stun --------------------------------------------------------------

def test_stun_refreshes_not_stacks():
    """Golden pair for defect B05."""
    target = make_target()
    apply_status(target, StatusType.STUN, magnitude=1.0, duration=1, source_id="s1", defects=CLEAN)
    apply_status(target, StatusType.STUN, magnitude=1.0, duration=3, source_id="s2", defects=CLEAN)
    stuns = [s for s in target.statuses if s.status_type == StatusType.STUN]
    assert len(stuns) == 1
    assert stuns[0].turns_remaining == 3


# --- Cooldowns ---------------------------------------------------------

def test_decrement_cooldowns_floors_and_removes_at_zero():
    c = make_target()
    c.cooldowns = {"A": 2, "B": 1}
    decrement_cooldowns(c)
    assert c.cooldowns == {"A": 1}


# --- End-to-end: a scripted battle, hand-checked --------------------------

def test_scripted_two_character_battle_matches_hand_calculation():
    """A minimal 1v1, driven by an explicit action script, with every HP
    number checked against a hand calculation. This is the "prove it by
    hand-driving a scripted battle" step from the project plan."""
    attacker = Character(
        id="atk", name="Attacker", team="party", element=Element.FIRE,
        max_hp=100, hp=100, max_energy=100, energy=100, energy_regen=20,
        abilities=[
            Ability(name="Basic Attack", element=Element.FIRE, energy_cost=0,
                    cooldown=0, target_type=TargetType.ONE_ENEMY, base_power=10.0,
                    is_basic_attack=True),
        ],
    )
    defender = Character(
        id="def", name="Defender", team="enemy", element=Element.ICE,
        max_hp=100, hp=100, max_energy=100, energy=100, energy_regen=20,
        abilities=[
            Ability(name="Basic Attack", element=Element.ICE, energy_cost=0,
                    cooldown=0, target_type=TargetType.ONE_ENEMY, base_power=10.0,
                    is_basic_attack=True),
        ],
    )

    from engine import scenarios as scenarios_module
    from engine.models import BattleState

    def _one_v_one(seed, defects):
        return [attacker.model_copy(deep=True), defender.model_copy(deep=True)], ["atk", "def"]

    scenarios_module.SCENARIOS["_TEST_1V1"] = _one_v_one
    try:
        eng = BattleEngine("_TEST_1V1", seed=0, defects=CLEAN, turn_cap=30)

        # Turn 1: attacker (Fire) hits defender (Ice) -- advantage x1.5.
        # 10 base_power * 1.5 = 15.0 damage.
        legal = eng.legal_actions()
        action = next(a for a in legal if a.actor_id == "atk")
        eng.take_action(action)
        assert eng.state.characters["def"].hp == 85.0

        # Turn 2: defender (Ice) hits attacker (Fire) -- disadvantage x1/1.5.
        # 10 * (1/1.5) = 6.666...
        legal = eng.legal_actions()
        action = next(a for a in legal if a.actor_id == "def")
        eng.take_action(action)
        assert abs(eng.state.characters["atk"].hp - (100 - 10 / 1.5)) < 1e-9

        assert eng.log[0].invariant_violations == []
        assert eng.log[1].invariant_violations == []
    finally:
        del scenarios_module.SCENARIOS["_TEST_1V1"]
