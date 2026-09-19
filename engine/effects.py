"""Status application, stacking rules and tick logic. Most seeded defects
live here, each behind one DefectFlags field."""

from engine.defects import DefectFlags
from engine.elements import elemental_multiplier
from engine.models import (
    Ability,
    Character,
    StatusEffect,
    StatusType,
)


def has_status(character: Character, status_type: StatusType) -> bool:
    return any(s.status_type == status_type for s in character.statuses)


def get_status(character: Character, status_type: StatusType) -> StatusEffect | None:
    """For non-stacking statuses (at most one entry)."""
    return next((s for s in character.statuses if s.status_type == status_type), None)


def all_of(character: Character, status_type: StatusType) -> list[StatusEffect]:
    return [s for s in character.statuses if s.status_type == status_type]


def apply_status(
    character: Character,
    status_type: StatusType,
    magnitude: float,
    duration: int,
    source_id: str,
    defects: DefectFlags,
) -> None:
    """Burn adds a new instance; every other status refreshes in place.

    B05: re-stunning appends a second Stun entry instead of refreshing.
    """
    new_entry = StatusEffect(
        status_type=status_type,
        magnitude=magnitude,
        turns_remaining=duration,
        source_id=source_id,
    )

    if status_type == StatusType.BURN:
        character.statuses.append(new_entry)
        return

    existing = get_status(character, status_type)
    if existing is None:
        character.statuses.append(new_entry)
        return

    if status_type == StatusType.STUN and defects.stun_adds_instead_of_refresh:
        character.statuses.append(new_entry)
        return

    existing.magnitude = magnitude
    existing.turns_remaining = duration


def tick_poison(character: Character, defects: DefectFlags) -> float:
    """Damage as a percentage of max HP. Returns damage applied.

    B04: with Weaken on the target, the percentage uses the weakened value.
    """
    poison = get_status(character, StatusType.POISON)
    if poison is None or character.hp <= 0:
        return 0.0

    base = character.max_hp
    if defects.poison_reads_weakened_value:
        weaken = get_status(character, StatusType.WEAKEN)
        if weaken is not None:
            base = character.max_hp * (1.0 - weaken.magnitude)

    damage = poison.magnitude * base
    return apply_dot_damage_with_shield_guard(character, damage, defects)


def tick_burn(character: Character, defects: DefectFlags) -> float:
    """Sum of all Burn instances.

    B02: two or more instances multiply instead of add.
    """
    burns = all_of(character, StatusType.BURN)
    if not burns or character.hp <= 0:
        return 0.0

    if defects.burn_stacks_multiply and len(burns) >= 2:
        total = 1.0
        for b in burns:
            total *= b.magnitude
    else:
        total = sum(b.magnitude for b in burns)

    return apply_dot_damage_with_shield_guard(character, total, defects)


def tick_regen(character: Character) -> float:
    """Flat heal. The caller must skip dead characters (see B07)."""
    regen = get_status(character, StatusType.REGEN)
    if regen is None:
        return 0.0
    healed = min(regen.magnitude, character.max_hp - character.hp)
    character.hp = min(character.max_hp, character.hp + regen.magnitude)
    return healed


def clear_statuses_on_death(character: Character, defects: DefectFlags) -> None:
    """Death clears every status.

    B07: Regen is kept, so it resumes after a revive.
    """
    if defects.regen_survives_death:
        character.statuses = [
            s for s in character.statuses if s.status_type == StatusType.REGEN
        ]
    else:
        character.statuses = []


def total_chill_pct(character: Character) -> float:
    chill = get_status(character, StatusType.CHILL)
    return chill.magnitude if chill is not None else 0.0


def apply_energy_regen(character: Character, defects: DefectFlags) -> None:
    """Start-of-turn energy regen, reduced by Chill and clamped to [0, max].

    B03: no clamps, so a Chill above 100% drives energy negative.
    """
    chill_pct = total_chill_pct(character)
    if defects.chill_no_floor:
        delta = character.energy_regen * (1.0 - chill_pct)
        character.energy = character.energy + delta
        character.energy = min(character.max_energy, character.energy)
    else:
        chill_pct = min(1.0, chill_pct)
        delta = character.energy_regen * (1.0 - chill_pct)
        character.energy = max(
            0.0, min(character.max_energy, character.energy + delta)
        )


def decrement_cooldowns(character: Character) -> None:
    for name in list(character.cooldowns):
        character.cooldowns[name] = max(0, character.cooldowns[name] - 1)
        if character.cooldowns[name] == 0:
            del character.cooldowns[name]


def decrement_status_durations(character: Character, defects: DefectFlags) -> None:
    """End of turn: age every status by one turn and drop expired ones."""
    survivors: list[StatusEffect] = []
    for s in character.statuses:
        s.turns_remaining -= 1
        if s.turns_remaining > 0:
            survivors.append(s)
    character.statuses = survivors


def apply_shield_absorption(
    target: Character, incoming_damage: float, defects: DefectFlags
) -> float:
    """Shield absorbs direct hits only; a drained Shield is removed.

    B10: the drained Shield stays at magnitude 0.
    """
    shield = get_status(target, StatusType.SHIELD)
    if shield is None or shield.magnitude <= 0:
        return incoming_damage

    absorbed = min(shield.magnitude, incoming_damage)
    shield.magnitude -= absorbed
    remaining_damage = incoming_damage - absorbed

    if shield.magnitude <= 0 and not defects.zero_shield_not_removed:
        target.statuses = [s for s in target.statuses if s is not shield]

    return remaining_damage


def apply_dot_damage_with_shield_guard(
    target: Character, raw_dot_damage: float, defects: DefectFlags
) -> float:
    """DoT damage goes straight to HP, bypassing Shield.

    B01: DoT damage goes through the Shield instead.
    """
    if defects.shield_absorbs_dot:
        raw_dot_damage = apply_shield_absorption(target, raw_dot_damage, defects)
    target.hp = max(0.0, target.hp - raw_dot_damage)
    return raw_dot_damage


def apply_damage_modifiers(
    ability: Ability, attacker: Character, target: Character, raw_damage: float
) -> float:
    """Elemental multiplier, then the attacker's Weaken. The only place the
    multiplier is applied (B09 adds a second one in resolution.py)."""
    damage = raw_damage * elemental_multiplier(ability.element, target.element)

    weaken = get_status(attacker, StatusType.WEAKEN)
    if weaken is not None:
        damage = damage * (1.0 - weaken.magnitude)

    return max(0.0, damage)
