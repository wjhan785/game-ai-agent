"""Status application, stacking rules, and tick logic.

This module -- together with resolution.py's fixed pipeline order -- is
where most of the ten seeded defects live. Every buggy branch is guarded by
one engine.defects.DefectFlags field; grep the flag name to find it.
"""
from __future__ import annotations

from engine.defects import DefectFlags
from engine.elements import elemental_multiplier
from engine.models import (
    Ability,
    Character,
    NON_STACKING_STATUSES,
    StatusEffect,
    StatusType,
)


def has_status(character: Character, status_type: StatusType) -> bool:
    return any(s.status_type == status_type for s in character.statuses)


def get_status(character: Character, status_type: StatusType) -> StatusEffect | None:
    """For non-stacking statuses only -- there is at most one entry."""
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
    """Attach a status effect to `character`, respecting each type's
    stacking rule.

    Clean semantics: Burn always appends a new independent instance (see
    tick_burn for how those combine). Every other status is non-stacking --
    a fresh application refreshes the existing instance's magnitude and
    duration in place rather than creating a second entry.

    Defect B05 (stun_adds_instead_of_refresh): when the target is already
    stunned, append a second Stun entry with the new duration instead of
    refreshing -- this is exactly the "duplicate entry for a non-stacking
    effect" shape the invariant checker looks for.
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
        # Append a second Stun entry rather than refreshing the existing
        # one in place -- this is the literal "duplicate entry for a
        # non-stacking status" shape invariants._check_duplicate_non_
        # stacking looks for, not just a bigger number in one entry.
        character.statuses.append(new_entry)
        return

    # Clean refresh-in-place for every non-stacking status, Stun included.
    existing.magnitude = magnitude
    existing.turns_remaining = duration


def tick_poison(character: Character, defects: DefectFlags) -> float:
    """Percentage-of-max-HP damage. Returns damage applied (0 if none).

    Defect B04 (poison_reads_weakened_value): if the target also carries
    Weaken, compute the percentage against the Weaken-adjusted value
    instead of the target's own raw max_hp.
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
    """Flat per-source damage, summed across all active Burn instances.

    Defect B02 (burn_stacks_multiply): with two or more simultaneously
    active Burn instances from different sources, combine their magnitudes
    multiplicatively instead of additively.
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
    """Flat per-turn healing. Caller is responsible for not invoking this
    on a dead character in clean mode -- see resolution.py and defect B07,
    which is precisely about that responsibility being skipped."""
    regen = get_status(character, StatusType.REGEN)
    if regen is None:
        return 0.0
    healed = min(regen.magnitude, character.max_hp - character.hp)
    character.hp = min(character.max_hp, character.hp + regen.magnitude)
    return healed


def clear_statuses_on_death(character: Character, defects: DefectFlags) -> None:
    """Clean: every status is cleared the moment a character dies, so a
    later revive starts from a blank slate.

    Defect B07 (regen_survives_death): Regen specifically is left in place,
    so if the character is revived by a separate effect, the stale Regen
    resumes ticking with its leftover duration.
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
    """Energy regen for the start of `character`'s own turn, reduced by
    Chill.

    Clean: the chill percentage is capped at 100% and the resulting energy
    is clamped to [0, max_energy] -- Chill can slow regen to a halt but
    never drive energy negative.

    Defect B03 (chill_no_floor): neither clamp is applied. A single strong
    Chill application (magnitude > 1.0, a legitimate content choice) then
    subtracts more than the character gained this tick, and with no floor
    the character's energy goes negative.
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
    """End-of-turn bookkeeping: age every status by one turn, dropping any
    whose duration has expired.

    Shield decays on whichever comes first: absorption draining its pool
    (handled at the moment of absorption, see apply_shield_absorption /
    defect B10) or this turn-count clock reaching zero (handled here,
    unconditionally -- B10 only concerns the absorption pathway).
    """
    survivors: list[StatusEffect] = []
    for s in character.statuses:
        s.turns_remaining -= 1
        if s.turns_remaining > 0:
            survivors.append(s)
    character.statuses = survivors


def apply_shield_absorption(
    target: Character, incoming_damage: float, defects: DefectFlags
) -> float:
    """Reduce `incoming_damage` by the target's Shield, if any. Only called
    for DIRECT ability hits -- DoT ticks must never call this (see
    apply_dot_damage_with_shield_guard below and defect B01).

    Clean: a Shield whose pool is drained to zero is removed immediately.
    Defect B10 (zero_shield_not_removed): the drained Shield is left in the
    status list with magnitude 0, so has_status(SHIELD) still returns True.
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
    """Wraps a DoT tick's damage application. Clean mode applies the damage
    directly, bypassing Shield entirely -- Shield only ever intercepts
    direct ability hits.

    Defect B01 (shield_absorbs_dot): routes the DoT damage through the same
    shield-absorption path a direct hit would use.
    """
    if defects.shield_absorbs_dot:
        raw_dot_damage = apply_shield_absorption(target, raw_dot_damage, defects)
    target.hp = max(0.0, target.hp - raw_dot_damage)
    return raw_dot_damage


def apply_damage_modifiers(
    ability: Ability, attacker: Character, target: Character, raw_damage: float
) -> float:
    """The single general damage-modifier pass: elemental advantage, then
    the attacker's own Weaken (which reduces outgoing damage).

    This function is the ONLY place elemental advantage is applied in
    clean mode. Defect B09 lives in resolution.py, not here: the buggy
    ability-resolution path pre-multiplies by elemental advantage before
    ever calling this function, so the multiplier lands twice. Keeping the
    multiplication itself in exactly one function, and letting the defect
    be "an extra call to elemental_multiplier before this pass" rather than
    a branch inside it, mirrors how this bug actually happens in real
    code -- a second code path re-deriving the same number.
    """
    damage = raw_damage * elemental_multiplier(ability.element, target.element)

    weaken = get_status(attacker, StatusType.WEAKEN)
    if weaken is not None:
        damage = damage * (1.0 - weaken.magnitude)

    return max(0.0, damage)
