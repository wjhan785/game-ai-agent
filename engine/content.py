"""Ability and character data. Abilities are shared constants;
characters are built fresh by `make_character`."""

from engine.models import (
    Ability,
    Character,
    Element,
    StatusApplication,
    StatusType,
    TargetType,
)


def basic_attack(element: Element = Element.NEUTRAL, power: float = 8.0) -> Ability:
    """The guaranteed 0-cost, 0-cooldown fallback. Defect B08 is about an
    enemy that is missing exactly this."""
    return Ability(
        name="Basic Attack",
        element=element,
        energy_cost=0,
        cooldown=0,
        target_type=TargetType.ONE_ENEMY,
        base_power=power,
        is_basic_attack=True,
    )


# --- Fire / sustained damage & Burn -----------------------------------

EMBER_SLASH = Ability(
    name="Ember Slash",
    element=Element.FIRE,
    energy_cost=20,
    cooldown=0,
    target_type=TargetType.ONE_ENEMY,
    base_power=18.0,
)

CINDER_BURN = Ability(
    name="Cinder Burn",
    element=Element.FIRE,
    energy_cost=30,
    cooldown=2,
    target_type=TargetType.ONE_ENEMY,
    base_power=10.0,
    applies=StatusApplication(status_type=StatusType.BURN, magnitude=6.0, duration=3),
)

FLAME_LASH = Ability(
    name="Flame Lash",
    element=Element.FIRE,
    energy_cost=25,
    cooldown=1,
    target_type=TargetType.ONE_ENEMY,
    base_power=8.0,
    applies=StatusApplication(status_type=StatusType.BURN, magnitude=5.0, duration=3),
)

# --- Ice / control -------------------------------------------------------

GLACIAL_SPIKE = Ability(
    name="Glacial Spike",
    element=Element.ICE,
    energy_cost=20,
    cooldown=0,
    target_type=TargetType.ONE_ENEMY,
    base_power=16.0,
)

FROST_BIND = Ability(
    name="Frost Bind",
    element=Element.ICE,
    energy_cost=30,
    cooldown=3,
    target_type=TargetType.ONE_ENEMY,
    base_power=4.0,
    applies=StatusApplication(status_type=StatusType.CHILL, magnitude=1.2, duration=2),
)

# --- Lightning / stun-lock ------------------------------------------------

SHOCK_COIL = Ability(
    name="Shock Coil",
    element=Element.LIGHTNING,
    energy_cost=35,
    cooldown=2,
    target_type=TargetType.ONE_ENEMY,
    base_power=6.0,
    applies=StatusApplication(status_type=StatusType.STUN, magnitude=1.0, duration=1),
)

VOLT_LANCE = Ability(
    name="Volt Lance",
    element=Element.LIGHTNING,
    energy_cost=15,
    cooldown=0,
    target_type=TargetType.ONE_ENEMY,
    base_power=22.0,
)

# --- Support / defensive --------------------------------------------------

AEGIS_WARD = Ability(
    name="Aegis Ward",
    element=Element.NEUTRAL,
    energy_cost=25,
    cooldown=3,
    target_type=TargetType.ONE_ALLY,
    base_power=0.0,
    applies=StatusApplication(status_type=StatusType.SHIELD, magnitude=30.0, duration=3),
)

MENDING_LIGHT = Ability(
    name="Mending Light",
    element=Element.NEUTRAL,
    energy_cost=20,
    cooldown=2,
    target_type=TargetType.ONE_ALLY,
    base_power=0.0,
    applies=StatusApplication(status_type=StatusType.REGEN, magnitude=8.0, duration=4),
)

RALLYING_CRY = Ability(
    name="Rallying Cry",
    element=Element.NEUTRAL,
    energy_cost=15,
    cooldown=2,
    target_type=TargetType.ALL_ENEMIES,
    base_power=0.0,
    applies=StatusApplication(status_type=StatusType.WEAKEN, magnitude=0.3, duration=2),
)

REVIVE = Ability(
    name="Revive",
    element=Element.NEUTRAL,
    energy_cost=50,
    cooldown=6,
    target_type=TargetType.ONE_ALLY,
    base_power=0.0,
    is_revive=True,
    revive_hp_fraction=0.5,
)

# --- Poison / Weaken bruisers ----------------------------------------------

TOXIC_STRIKE = Ability(
    name="Toxic Strike",
    element=Element.NEUTRAL,
    energy_cost=25,
    cooldown=2,
    target_type=TargetType.ONE_ENEMY,
    base_power=6.0,
    applies=StatusApplication(status_type=StatusType.POISON, magnitude=0.08, duration=3),
)

CRIPPLING_BLOW = Ability(
    name="Crippling Blow",
    element=Element.NEUTRAL,
    energy_cost=20,
    cooldown=2,
    target_type=TargetType.ONE_ENEMY,
    base_power=10.0,
    applies=StatusApplication(status_type=StatusType.WEAKEN, magnitude=0.35, duration=3),
)

# --- Burst -----------------------------------------------------------------

CATACLYSM = Ability(
    name="Cataclysm",
    element=Element.LIGHTNING,
    energy_cost=40,
    cooldown=3,
    target_type=TargetType.ONE_ENEMY,
    base_power=32.0,
)

# --- An unaffordable ability: without Basic Attack (B08) its owner is stuck.
OVERTUNED_BOLT = Ability(
    name="Overtuned Bolt",
    element=Element.LIGHTNING,
    energy_cost=999,
    cooldown=1,
    target_type=TargetType.ONE_ENEMY,
    base_power=50.0,
)


def make_character(
    char_id: str,
    name: str,
    team: str,
    element: Element,
    max_hp: float,
    max_energy: float,
    energy_regen: float,
    abilities: list[Ability],
    starting_energy: float | None = None,
    strip_basic_attack: bool = False,
    speed: float = 100.0,
) -> Character:
    """One fresh Character. `strip_basic_attack` is B08's content side;
    `speed` must come from the palette in engine.models."""
    kit = list(abilities)
    if not strip_basic_attack:
        kit = [basic_attack(element=element)] + kit
    return Character(
        id=char_id,
        name=name,
        team=team,  # type: ignore[arg-type]
        element=element,
        max_hp=max_hp,
        hp=max_hp,
        max_energy=max_energy,
        energy=starting_energy if starting_energy is not None else max_energy,
        energy_regen=energy_regen,
        abilities=kit,
        speed=speed,
    )
