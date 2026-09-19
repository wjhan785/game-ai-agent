"""The turn pipeline. This order is the spec; most defects break it.

Per turn-slot:
  1. Dead check      dead characters are skipped.
  2. DoT/HoT tick    Poison, then Burn, then Regen.
  3. Death check     a tick can kill.
  4. Energy regen    reduced by Chill, clamped.
  5. Stun check      a stunned turn skips to step 7.
  6. Action          legal-move check against cooldowns as they stood
                     before this turn's decrement.
  7. End of turn     decrement cooldowns and status durations, then
                     advance the scheduler.
Invariants are checked afterwards by engine.py.

Split into resolve_pre / resolve_action / resolve_post so the engine can
ask for a decision between steps 5 and 6.
"""

from typing import Optional

from pydantic import BaseModel, Field

from engine.defects import DefectFlags
from engine.effects import (
    apply_damage_modifiers,
    apply_energy_regen,
    apply_shield_absorption,
    apply_status,
    clear_statuses_on_death,
    decrement_cooldowns,
    decrement_status_durations,
    get_status,
    tick_burn,
    tick_poison,
    tick_regen,
)
from engine.elements import elemental_multiplier
from engine.models import (
    BASE_ACTION_VALUE,
    CYCLE_AV,
    Action,
    BattleState,
    Character,
    StatusEffect,
    StatusType,
    TargetType,
)


class CharacterSnapshot(BaseModel):
    hp: float
    energy: float
    alive: bool
    statuses: list[StatusEffect]
    cooldowns: dict[str, int]
    action_value: float = 0.0


def snapshot(character: Character) -> CharacterSnapshot:
    return CharacterSnapshot(
        hp=character.hp,
        energy=character.energy,
        alive=character.alive,
        statuses=[s.model_copy() for s in character.statuses],
        cooldowns=dict(character.cooldowns),
        action_value=character.action_value,
    )


class AbilityEffectRecord(BaseModel):
    ability_name: str
    target_ids: list[str]
    raw_damage: float = 0.0
    per_target_damage: dict[str, float] = Field(default_factory=dict)
    per_target_shield_absorbed: dict[str, float] = Field(default_factory=dict)
    applied_status: Optional[str] = None
    revived: list[str] = Field(default_factory=list)


class TurnRecord(BaseModel):
    step_count: int
    round_number: int
    actor_id: str
    elapsed_av: float  # scheduler clock at the start of this turn-slot
    skipped_reason: Optional[str] = None  # "dead" | "stunned" | "no_legal_actions"
    action: Optional[Action] = None
    dot_hot_damage: dict[str, dict[str, float]] = Field(default_factory=dict)
    energy_regen_delta: dict[str, float] = Field(default_factory=dict)
    ability_effect: Optional[AbilityEffectRecord] = None
    deaths: list[str] = Field(default_factory=list)
    before: dict[str, CharacterSnapshot] = Field(default_factory=dict)
    after: dict[str, CharacterSnapshot] = Field(default_factory=dict)
    invariant_violations: list[str] = Field(default_factory=list)

    def needs_action(self) -> bool:
        return self.skipped_reason is None


def _snapshot_all(state: BattleState) -> dict[str, CharacterSnapshot]:
    return {cid: snapshot(c) for cid, c in state.characters.items()}


def resolve_pre(state: BattleState, defects: DefectFlags) -> TurnRecord:
    """Steps 1-5. `record.needs_action()` says whether a decision is needed."""
    actor_id = state.current_actor_id()
    actor = state.characters[actor_id]
    record = TurnRecord(
        step_count=state.step_count,
        round_number=state.round_number,
        actor_id=actor_id,
        elapsed_av=state.elapsed_av,
        before=_snapshot_all(state),
    )

    if not actor.alive:
        record.skipped_reason = "dead"
        return record

    apply_pre_check_cooldown_decrement(state, defects)

    # Step 2: DoT/HoT, fixed order Poison -> Burn -> Regen.
    poison_dmg = tick_poison(actor, defects)
    burn_dmg = tick_burn(actor, defects)
    if poison_dmg or burn_dmg:
        record.dot_hot_damage[actor_id] = {"poison": poison_dmg, "burn": burn_dmg}

    # Step 3: death check after DoT.
    if actor.hp <= 0 and actor.alive:
        actor.alive = False
        clear_statuses_on_death(actor, defects)
        record.deaths.append(actor_id)

    # Regen after the death check; the dead are never healed.
    if actor.alive:
        regen_healed = tick_regen(actor)
        if regen_healed:
            record.dot_hot_damage.setdefault(actor_id, {})["regen"] = regen_healed

    if not actor.alive:
        record.skipped_reason = "dead"
        return record

    # Step 4: energy regen, Chill-reduced.
    energy_before = actor.energy
    apply_energy_regen(actor, defects)
    record.energy_regen_delta[actor_id] = actor.energy - energy_before

    # Step 5: stun check.
    stun = get_status(actor, StatusType.STUN)
    if stun is not None and stun.turns_remaining > 0:
        record.skipped_reason = "stunned"
        return record

    return record


def _resolve_ability_effect(
    state: BattleState, action: Action, defects: DefectFlags
) -> AbilityEffectRecord:
    actor = state.characters[action.actor_id]
    ability = actor.ability_by_name(action.ability_name)
    assert ability is not None

    actor.energy -= ability.energy_cost
    if ability.cooldown > 0:
        actor.cooldowns[ability.name] = ability.cooldown

    if ability.target_type == TargetType.SELF:
        target_ids = [action.actor_id]
    elif ability.target_type == TargetType.ALL_ENEMIES:
        opposing = "enemy" if actor.team == "party" else "party"
        target_ids = [c.id for c in state.alive_on_team(opposing)]
    else:
        assert action.target_id is not None
        target_ids = [action.target_id]

    record = AbilityEffectRecord(ability_name=ability.name, target_ids=target_ids)

    for target_id in target_ids:
        target = state.characters[target_id]

        if ability.is_revive:
            target.alive = True
            target.hp = target.max_hp * ability.revive_hp_fraction
            record.revived.append(target_id)
            continue

        if ability.base_power > 0:
            raw = ability.base_power
            if defects.elemental_multiplier_applied_twice:
                # B09: multiplier here too, then again in apply_damage_modifiers.
                raw = raw * elemental_multiplier(ability.element, target.element)
            damage = apply_damage_modifiers(ability, actor, target, raw)
            pre_shield = damage
            damage = apply_shield_absorption(target, damage, defects)
            target.hp = max(0.0, target.hp - damage)
            record.per_target_damage[target_id] = damage
            # Measured across the absorption call, not against base_power.
            absorbed = pre_shield - damage
            if absorbed > 0:
                record.per_target_shield_absorbed[target_id] = absorbed

            if target.hp <= 0 and target.alive:
                target.alive = False
                clear_statuses_on_death(target, defects)

        if ability.applies is not None and target.alive:
            # Don't re-attach a status to a target this hit just killed.
            apply_status(
                target,
                ability.applies.status_type,
                ability.applies.magnitude,
                ability.applies.duration,
                source_id=action.actor_id,
                defects=defects,
            )
            record.applied_status = ability.applies.status_type.value

    return record


def resolve_action(
    state: BattleState, action: Action, defects: DefectFlags, record: TurnRecord
) -> None:
    """Step 6. Caller has already validated `action` with rules.check_legal."""
    record.action = action
    record.ability_effect = _resolve_ability_effect(state, action, defects)
    for target_id in record.ability_effect.target_ids + record.ability_effect.revived:
        if not state.characters[target_id].alive and target_id not in record.deaths:
            record.deaths.append(target_id)


def resolve_post(state: BattleState, defects: DefectFlags, record: TurnRecord) -> None:
    """Step 7: end-of-turn bookkeeping, then advance the scheduler."""
    actor = state.characters[record.actor_id]
    if actor.alive:
        decrement_cooldowns(actor)
    if actor.alive:
        decrement_status_durations(actor, defects)

    # Reschedule the actor one personal cycle later, even if it was skipped.
    actor.action_value = state.elapsed_av + BASE_ACTION_VALUE / actor.speed

    record.after = _snapshot_all(state)

    state.elapsed_av = min(c.action_value for c in state.characters.values())
    state.round_number = int((state.elapsed_av - state.opening_av) / CYCLE_AV)
    state.step_count += 1

    check_battle_end(state)


def apply_pre_check_cooldown_decrement(state: BattleState, defects: DefectFlags) -> None:
    """B06: an extra cooldown decrement before the legal-move check, on top
    of resolve_post's own, so cooldowns drain twice per turn."""
    if defects.cooldown_decrement_before_check:
        actor = state.characters[state.current_actor_id()]
        if actor.alive:
            decrement_cooldowns(actor)


def check_battle_end(state: BattleState) -> None:
    party_alive = bool(state.alive_on_team("party"))
    enemy_alive = bool(state.alive_on_team("enemy"))
    if not party_alive:
        state.finished = True
        state.outcome = "enemy_win"
    elif not enemy_alive:
        state.finished = True
        state.outcome = "party_win"
    elif state.round_number >= state.turn_cap:
        state.finished = True
        state.outcome = "turn_cap"
