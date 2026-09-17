"""The six hand-built scenarios from docs/scenario-matrix.md.

Each builder returns a fresh set of Character instances for one
(scenario_id, seed) pair. Seeding perturbs starting HP and energy by a
bounded percentage so repeated exploration runs see some variety without
an open-ended world -- see `_perturb`.

All six scenarios are 3v3 so every method's action budget means the same
thing everywhere (see the budget section of the project plan).
"""
from __future__ import annotations

import random
from typing import Callable

from engine.content import (
    AEGIS_WARD,
    CATACLYSM,
    CINDER_BURN,
    CRIPPLING_BLOW,
    EMBER_SLASH,
    FLAME_LASH,
    FROST_BIND,
    GLACIAL_SPIKE,
    MENDING_LIGHT,
    OVERTUNED_BOLT,
    RALLYING_CRY,
    REVIVE,
    SHOCK_COIL,
    TOXIC_STRIKE,
    VOLT_LANCE,
    make_character,
)
from engine.defects import DefectFlags
from engine.models import BASE_ACTION_VALUE, BattleState, Character, Element

PERTURB_HP_PCT = 0.10
PERTURB_ENERGY_PCT = 0.15


def _perturb(rng: random.Random, characters: list[Character]) -> None:
    for c in characters:
        hp_mult = 1.0 + rng.uniform(-PERTURB_HP_PCT, PERTURB_HP_PCT)
        energy_mult = 1.0 + rng.uniform(-PERTURB_ENERGY_PCT, PERTURB_ENERGY_PCT)
        c.max_hp = round(c.max_hp * hp_mult, 1)
        c.hp = c.max_hp
        c.max_energy = round(c.max_energy * energy_mult, 1)
        c.energy = min(c.energy, c.max_energy)


def _interleave(party: list[Character], enemy: list[Character]) -> list[str]:
    order: list[str] = []
    for p, e in zip(party, enemy):
        order.append(p.id)
        order.append(e.id)
    return order


def build_s1_bulwark(seed: int, defects: DefectFlags) -> tuple[list[Character], list[str]]:
    """Shield x DoT: a shielder keeps the party alive against two
    Fire-heavy Burn appliers."""
    party = [
        make_character("p1", "Warden", "party", Element.NEUTRAL, 110, 100, 22, [AEGIS_WARD], speed=80),
        make_character("p2", "Blade", "party", Element.FIRE, 90, 90, 20, [EMBER_SLASH, CINDER_BURN], speed=125),
        make_character("p3", "Bolt", "party", Element.LIGHTNING, 85, 95, 20, [VOLT_LANCE, SHOCK_COIL], speed=160),
    ]
    enemy = [
        make_character("e1", "Cinderfang", "enemy", Element.FIRE, 95, 90, 20, [FLAME_LASH, CINDER_BURN], speed=100),
        make_character("e2", "Ashclaw", "enemy", Element.FIRE, 90, 85, 20, [FLAME_LASH], speed=100),
        make_character("e3", "Grunt", "enemy", Element.NEUTRAL, 100, 60, 15, [], speed=80),
    ]
    rng = random.Random(seed)
    _perturb(rng, party + enemy)
    return party + enemy, _interleave(party, enemy)


def build_s2_pyroclasm(seed: int, defects: DefectFlags) -> tuple[list[Character], list[str]]:
    """Multi-source stacking: two independent Fire attackers apply Burn
    to the same high-HP enemy from different sources."""
    party = [
        make_character("p1", "Pyra", "party", Element.FIRE, 90, 90, 20, [CINDER_BURN, EMBER_SLASH], speed=125),
        make_character("p2", "Ignis", "party", Element.FIRE, 88, 90, 20, [FLAME_LASH, EMBER_SLASH], speed=125),
        make_character("p3", "Warden", "party", Element.NEUTRAL, 105, 100, 20, [AEGIS_WARD], speed=80),
    ]
    enemy = [
        make_character("e1", "Colossus", "enemy", Element.NEUTRAL, 160, 70, 15, [CATACLYSM], speed=80),
        make_character("e2", "Skirmisher1", "enemy", Element.LIGHTNING, 80, 85, 20, [VOLT_LANCE], speed=160),
        make_character("e3", "Skirmisher2", "enemy", Element.ICE, 80, 85, 20, [GLACIAL_SPIKE], speed=100),
    ]
    rng = random.Random(seed)
    _perturb(rng, party + enemy)
    return party + enemy, _interleave(party, enemy)


def build_s3_deep_freeze(seed: int, defects: DefectFlags) -> tuple[list[Character], list[str]]:
    """Energy denial: high-energy-cost casters against Ice control
    enemies applying Chill."""
    party = [
        make_character("p1", "Caster1", "party", Element.LIGHTNING, 85, 90, 18, [CATACLYSM, VOLT_LANCE], speed=100),
        make_character("p2", "Caster2", "party", Element.FIRE, 85, 95, 18, [CINDER_BURN, EMBER_SLASH], speed=100),
        make_character("p3", "Support", "party", Element.NEUTRAL, 100, 100, 20, [AEGIS_WARD], speed=80),
    ]
    enemy = [
        make_character("e1", "Frostweaver1", "enemy", Element.ICE, 85, 90, 18, [FROST_BIND, GLACIAL_SPIKE], speed=100),
        make_character("e2", "Frostweaver2", "enemy", Element.ICE, 85, 90, 18, [FROST_BIND, GLACIAL_SPIKE], speed=100),
        make_character("e3", "IceboundGrunt", "enemy", Element.NEUTRAL, 110, 60, 15, [], speed=80),
    ]
    rng = random.Random(seed)
    _perturb(rng, party + enemy)
    return party + enemy, _interleave(party, enemy)


def build_s4_attrition(seed: int, defects: DefectFlags) -> tuple[list[Character], list[str]]:
    """%-damage x modifiers: a near-mirror match, Poison and Weaken on
    both sides."""
    party = [
        make_character("p1", "Venomblade", "party", Element.NEUTRAL, 95, 90, 18, [TOXIC_STRIKE, EMBER_SLASH], speed=100),
        make_character("p2", "Breaker", "party", Element.LIGHTNING, 90, 85, 18, [CRIPPLING_BLOW, VOLT_LANCE], speed=125),
        make_character("p3", "Medic", "party", Element.NEUTRAL, 95, 90, 18, [MENDING_LIGHT], speed=100),
    ]
    enemy = [
        make_character("e1", "Venomclaw", "enemy", Element.NEUTRAL, 95, 90, 18, [TOXIC_STRIKE, GLACIAL_SPIKE], speed=100),
        make_character("e2", "Crusher", "enemy", Element.LIGHTNING, 90, 85, 18, [CRIPPLING_BLOW, CATACLYSM], speed=125),
        make_character("e3", "Healer", "enemy", Element.NEUTRAL, 95, 90, 18, [MENDING_LIGHT], speed=100),
    ]
    rng = random.Random(seed)
    _perturb(rng, party + enemy)
    return party + enemy, _interleave(party, enemy)


def build_s5_lockdown(seed: int, defects: DefectFlags) -> tuple[list[Character], list[str]]:
    """Turn-skip x cooldown: frequent low-cooldown party actors against
    Lightning stun-lock enemies with tight cooldowns. Also the B08
    showcase: `Overtuned` has one wildly unaffordable ability and,
    when defects.enemy_no_basic_fallback is set, no Basic Attack to
    fall back on -- it should get permanently stuck."""
    party = [
        make_character("p1", "Quickblade", "party", Element.FIRE, 90, 90, 22, [EMBER_SLASH, VOLT_LANCE], speed=160),
        make_character("p2", "Duelist", "party", Element.ICE, 88, 90, 22, [GLACIAL_SPIKE], speed=125),
        make_character("p3", "Support", "party", Element.NEUTRAL, 100, 100, 20, [AEGIS_WARD], speed=80),
    ]
    enemy = [
        make_character("e1", "Shockmaw", "enemy", Element.LIGHTNING, 90, 90, 20, [SHOCK_COIL, VOLT_LANCE], speed=125),
        make_character("e2", "Stormcaller", "enemy", Element.LIGHTNING, 85, 90, 20, [SHOCK_COIL], speed=125),
        make_character(
            "e3",
            "Overtuned",
            "enemy",
            Element.LIGHTNING,
            80,
            60,
            15,
            [OVERTUNED_BOLT],
            strip_basic_attack=defects.enemy_no_basic_fallback,
            speed=80,
        ),
    ]
    rng = random.Random(seed)
    _perturb(rng, party + enemy)
    return party + enemy, _interleave(party, enemy)


def build_s6_last_stand(seed: int, defects: DefectFlags) -> tuple[list[Character], list[str]]:
    """Death x persistent healing: a Regen/revive-heavy party against a
    single hard-hitting burst enemy."""
    party = [
        make_character("p1", "Cleric", "party", Element.NEUTRAL, 95, 110, 22, [MENDING_LIGHT, REVIVE], speed=80),
        make_character("p2", "Fighter1", "party", Element.FIRE, 75, 85, 18, [EMBER_SLASH], speed=125),
        make_character("p3", "Fighter2", "party", Element.LIGHTNING, 80, 85, 18, [VOLT_LANCE], speed=125),
    ]
    enemy = [
        make_character("e1", "Behemoth", "enemy", Element.LIGHTNING, 130, 90, 18, [CATACLYSM], speed=80),
        make_character("e2", "Adds1", "enemy", Element.NEUTRAL, 55, 60, 15, [], speed=100),
        make_character("e3", "Adds2", "enemy", Element.NEUTRAL, 55, 60, 15, [], speed=100),
    ]
    rng = random.Random(seed)
    _perturb(rng, party + enemy)
    return party + enemy, _interleave(party, enemy)


ScenarioBuilder = Callable[[int, DefectFlags], tuple[list[Character], list[str]]]

SCENARIOS: dict[str, ScenarioBuilder] = {
    "S1": build_s1_bulwark,
    "S2": build_s2_pyroclasm,
    "S3": build_s3_deep_freeze,
    "S4": build_s4_attrition,
    "S5": build_s5_lockdown,
    "S6": build_s6_last_stand,
}


def build_battle_state(
    scenario_id: str, seed: int, defects: DefectFlags, turn_cap: int = 30
) -> BattleState:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario_id {scenario_id!r}, expected one of {list(SCENARIOS)}")
    characters, turn_order = SCENARIOS[scenario_id](seed, defects)

    # Defect B11 (tie_break_by_character_id): ties for the next turn are
    # meant to be broken by this declared order (see BattleState.
    # current_actor_id) -- the buggy build sorts ids instead of using the
    # scenario's own interleave. `current_actor_id` itself stays pure and
    # defect-free; every seeded defect that touches the schedule is
    # decided here, once, at construction.
    if defects.tie_break_by_character_id:
        turn_order = sorted(turn_order)

    # Every character's first turn is one personal cycle away; the clock
    # starts at the fastest character's cycle length, so battle time 0 is
    # "nobody has acted yet" rather than "everybody already has".
    for c in characters:
        c.action_value = BASE_ACTION_VALUE / c.speed
    opening_av = min(c.action_value for c in characters)

    return BattleState(
        scenario_id=scenario_id,
        seed=seed,
        turn_order=turn_order,
        characters={c.id: c for c in characters},
        turn_cap=turn_cap,
        elapsed_av=opening_av,
        opening_av=opening_av,
    )
