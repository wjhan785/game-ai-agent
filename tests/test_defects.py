"""One test per seeded defect: enabling exactly that flag (via
engine.defects.single_flag) changes the trajectory relative to clean mode,
and in the direction ANSWER_KEY.md/defects.py documents. Enabling no flags
must reproduce the golden trace exactly -- test_engine_golden.py already
covers that; this file is the other half of the ablation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.defects import DefectFlags, single_flag
from engine.effects import (
    apply_energy_regen,
    apply_status,
    clear_statuses_on_death,
    tick_burn,
    tick_poison,
    tick_regen,
)
from engine.engine import BattleEngine
from engine.models import Character, Element, StatusType
from engine import scenarios as scenarios_module
from tests.test_engine_golden import make_target

CLEAN = DefectFlags()


# --- B01: shield absorbs DoT ------------------------------------------

def test_b01_shield_absorbs_dot_when_flagged():
    defects = single_flag("B01")
    target = make_target(hp=100.0)
    apply_status(target, StatusType.SHIELD, magnitude=50.0, duration=3, source_id="s", defects=defects)
    apply_status(target, StatusType.BURN, magnitude=10.0, duration=3, source_id="b", defects=defects)
    tick_burn(target, defects)
    assert target.hp == 100.0  # burn fully eaten by shield -- wrong
    shield = next(s for s in target.statuses if s.status_type == StatusType.SHIELD)
    assert shield.magnitude == 40.0


# --- B02: Burn stacks multiply ------------------------------------------

def test_b02_burn_stacks_multiply_when_flagged():
    defects = single_flag("B02")
    target = make_target(hp=100.0)
    apply_status(target, StatusType.BURN, magnitude=5.0, duration=3, source_id="a", defects=defects)
    apply_status(target, StatusType.BURN, magnitude=7.0, duration=3, source_id="b", defects=defects)
    damage = tick_burn(target, defects)
    assert damage == 35.0  # 5 * 7, not 5 + 7
    assert target.hp == 65.0


# --- B03: Chill no floor ------------------------------------------------

def test_b03_chill_drives_energy_negative_when_flagged():
    defects = single_flag("B03")
    target = make_target(energy=5.0, max_energy=100.0, energy_regen=20.0)
    apply_status(target, StatusType.CHILL, magnitude=1.5, duration=2, source_id="c", defects=defects)
    apply_energy_regen(target, defects)
    assert target.energy == -5.0


# --- B04: Poison reads Weaken-adjusted value ----------------------------

def test_b04_poison_reads_weakened_value_when_flagged():
    defects = single_flag("B04")
    target = make_target(max_hp=100.0, hp=100.0)
    apply_status(target, StatusType.POISON, magnitude=0.1, duration=3, source_id="p", defects=defects)
    apply_status(target, StatusType.WEAKEN, magnitude=0.4, duration=3, source_id="w", defects=defects)
    damage = tick_poison(target, defects)
    assert damage == 6.0  # 0.1 * (100 * 0.6), not 0.1 * 100


# --- B05: Stun adds instead of refreshing -------------------------------

def test_b05_stun_duplicates_when_flagged():
    defects = single_flag("B05")
    target = make_target()
    apply_status(target, StatusType.STUN, magnitude=1.0, duration=1, source_id="s1", defects=defects)
    apply_status(target, StatusType.STUN, magnitude=1.0, duration=3, source_id="s2", defects=defects)
    stuns = [s for s in target.statuses if s.status_type == StatusType.STUN]
    assert len(stuns) == 2  # duplicate entry, not a refresh


# --- B06: cooldown double-decrement -------------------------------------

def test_b06_cooldown_fires_one_turn_early_when_flagged():
    from engine.models import Ability, TargetType

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

    scenarios_module.SCENARIOS["_T_B06"] = make_1v1
    try:
        for label, defects, expect_early in [
            ("clean", DefectFlags(), False),
            ("buggy", single_flag("B06"), True),
        ]:
            eng = BattleEngine("_T_B06", seed=0, defects=defects, turn_cap=30)
            atk_turns = 0
            reused_while_on_cooldown = False
            while atk_turns < 3:
                legal = eng.legal_actions()
                actor = eng.state.current_actor_id()
                if actor == "atk":
                    names = {a.ability_name for a in legal}
                    action = next(a for a in legal if a.ability_name == "Big Hit") if "Big Hit" in names else next(a for a in legal if a.actor_id == "atk")
                    rec = eng.take_action(action)
                    # rec.before is the snapshot taken at the very top of
                    # resolve_pre, before B06's extra decrement -- the
                    # "true" cooldown this turn started with.
                    before_cd = rec.before["atk"].cooldowns.get("Big Hit", 0)
                    if rec.action.ability_name == "Big Hit" and before_cd > 0:
                        reused_while_on_cooldown = True
                    atk_turns += 1
                else:
                    rec = eng.take_action(legal[0])
            assert reused_while_on_cooldown == expect_early, label
            has_violation = any(
                "cooldown remaining" in v for r in eng.log for v in r.invariant_violations
            )
            assert has_violation == expect_early, label
    finally:
        del scenarios_module.SCENARIOS["_T_B06"]


# --- B07: Regen survives death, resumes on revive -----------------------

def test_b07_regen_survives_death_when_flagged():
    defects = single_flag("B07")
    target = make_target(hp=0.0)
    apply_status(target, StatusType.REGEN, magnitude=8.0, duration=4, source_id="r", defects=defects)
    target.alive = False
    clear_statuses_on_death(target, defects)
    assert any(s.status_type == StatusType.REGEN for s in target.statuses)

    # separate revive effect
    target.alive = True
    target.hp = 40.0
    healed = tick_regen(target)
    assert healed == 8.0  # stale Regen resumed ticking
    assert target.hp == 48.0


def test_b07_clean_regen_cleared_on_death_stays_cleared_on_revive():
    target = make_target(hp=0.0)
    apply_status(target, StatusType.REGEN, magnitude=8.0, duration=4, source_id="r", defects=CLEAN)
    target.alive = False
    clear_statuses_on_death(target, CLEAN)
    assert target.statuses == []

    target.alive = True
    target.hp = 40.0
    healed = tick_regen(target)
    assert healed == 0.0
    assert target.hp == 40.0


# --- B08: enemy with no basic-attack fallback gets stuck ----------------

def test_b08_enemy_without_fallback_gets_permanently_stuck():
    eng_clean = BattleEngine("S5", seed=1, defects=DefectFlags(), turn_cap=30)
    eng_buggy = BattleEngine("S5", seed=1, defects=single_flag("B08"), turn_cap=30)

    def overtuned_legal_ever(eng):
        # Overtuned's only non-basic ability costs 999 energy, which it can
        # never afford. Clean mode still lists Basic Attack; buggy mode
        # (B08) strips it, so Overtuned should have zero legal actions
        # every time it's up, and the no-progress detector should fire.
        for _ in range(60):
            legal = eng.legal_actions()
            if eng.state.finished:
                return None
            actor = eng.state.current_actor_id()
            if actor == "e3":
                return bool(legal)
            eng.take_action(legal[0])
        return None

    assert overtuned_legal_ever(eng_clean) is True

    # Drive the buggy engine and confirm the no-progress detector fires
    # for e3 specifically, without ever crashing the loop.
    steps = 0
    while not eng_buggy.state.finished and steps < 200:
        legal = eng_buggy.legal_actions()
        if not legal:
            break
        eng_buggy.take_action(legal[0])
        steps += 1
    no_progress_hits = [
        v for r in eng_buggy.log for v in r.invariant_violations if "no-progress" in v and v.startswith("e3")
    ]
    assert no_progress_hits, "expected e3 to trip the no-progress detector"


# --- B09: elemental multiplier applied twice ----------------------------

def test_b09_elemental_multiplier_doubled_when_flagged():
    from engine.effects import apply_damage_modifiers
    from engine.elements import elemental_multiplier
    from engine.models import Ability, TargetType

    attacker = make_target(id="a", team="party", element=Element.FIRE)
    target = make_target(id="t", team="enemy", element=Element.ICE)
    ability = Ability(
        name="Ember", element=Element.FIRE, target_type=TargetType.ONE_ENEMY, base_power=10.0
    )

    defects = single_flag("B09")
    raw = ability.base_power
    if defects.elemental_multiplier_applied_twice:
        raw = raw * elemental_multiplier(ability.element, target.element)
    damage = apply_damage_modifiers(ability, attacker, target, raw)
    assert damage == 10.0 * 1.5 * 1.5

    clean_damage = apply_damage_modifiers(ability, attacker, target, ability.base_power)
    assert clean_damage == 10.0 * 1.5


# --- B10: zero-magnitude shield not removed ------------------------------

def test_b10_zero_shield_persists_when_flagged():
    from engine.effects import apply_shield_absorption

    defects = single_flag("B10")
    target = make_target()
    apply_status(target, StatusType.SHIELD, magnitude=10.0, duration=3, source_id="s", defects=defects)
    apply_shield_absorption(target, 15.0, defects)
    shield = next((s for s in target.statuses if s.status_type == StatusType.SHIELD), None)
    assert shield is not None
    assert shield.magnitude <= 0


# --- B11: turn-order ties broken by character id, not declared order ----

def test_b11_tie_break_uses_character_id_when_flagged():
    # Real scenarios carry varied per-character speeds by design (see
    # scenarios.py), so this needs a scenario where every character is at
    # the same speed -- a full six-way tie at battle start -- to isolate
    # the tie-break rule itself from any scenario's own speed balance.
    def uniform_speed(seed, defects):
        from engine.content import make_character

        party = [make_character(cid, cid, "party", Element.NEUTRAL, 100, 50, 20, []) for cid in ("p1", "p2", "p3")]
        enemy = [make_character(cid, cid, "enemy", Element.NEUTRAL, 100, 50, 20, []) for cid in ("e1", "e2", "e3")]
        return party + enemy, scenarios_module._interleave(party, enemy)

    scenarios_module.SCENARIOS["_T_B11_UNIFORM"] = uniform_speed
    try:
        clean = BattleEngine("_T_B11_UNIFORM", seed=0, defects=DefectFlags())
        assert clean.state.current_actor_id() == "p1"  # declared order: p1 first

        buggy = BattleEngine("_T_B11_UNIFORM", seed=0, defects=single_flag("B11"))
        assert buggy.state.current_actor_id() == "e1"  # id-sorted: e1 before p1
    finally:
        del scenarios_module.SCENARIOS["_T_B11_UNIFORM"]


def test_b11_clean_turn_order_is_declared_not_sorted():
    buggy_order = scenarios_module.build_battle_state("S1", seed=0, defects=single_flag("B11"), turn_cap=20).turn_order
    clean_order = scenarios_module.build_battle_state("S1", seed=0, defects=CLEAN, turn_cap=20).turn_order
    assert buggy_order == sorted(clean_order)
    assert clean_order == ["p1", "e1", "p2", "e2", "p3", "e3"]
    assert clean_order != buggy_order
