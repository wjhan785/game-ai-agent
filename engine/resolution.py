"""The fixed turn-resolution pipeline.

This is the spec. Most of the ten seeded defects are violations of this
exact ordering, so the ordering is written down once, here, and every
other module defers to it.

Per character turn-slot, in order:

  1. Dead check       -- a dead character's turn-slot is skipped entirely.
  2. DoT/HoT tick      -- Poison, then Burn, then Regen. Fixed order.
  3. Death check       -- a DoT tick can kill; re-check before energy regen.
  4. Energy regen      -- reduced by Chill, clamped (defect B03 removes
                           the clamp).
  5. Stun check        -- if stunned, this turn-slot is consumed and
                           resolution jumps straight to end-of-turn
                           bookkeeping; no action is possible.
  6. Legal-move check + action -- evaluated against cooldowns as they
                           stood since the end of the actor's last turn
                           (i.e. BEFORE this turn's cooldown decrement,
                           which happens in step 8). Defect B06 moves the
                           decrement to before this check instead.
  8. End-of-turn        -- decrement cooldowns (unless B06 already did it
                           in step 6's place), decrement status durations,
                           drop expired effects.
  9. Invariant check    -- performed one layer up, by engine.py, against
                           the TurnRecord this module returns.

The pipeline is split into three functions -- `resolve_pre`, `resolve_
action`, `resolve_post` -- rather than one, because whether an action is
even needed (steps 1-5) can only be known after running the dead/stun
checks, and the caller (engine.BattleEngine) needs to ask an agent for a
decision in between. See engine.py's `advance_to_decision` for how the
three are stitched into one turn.
"""
from __future__ import annotations

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
    has_status,
    tick_burn,
    tick_poison,
    tick_regen,
)
from engine.elements import elemental_multiplier
from engine.models import Action, BattleState, Character, StatusEffect, StatusType, TargetType


class CharacterSnapshot(BaseModel):
    hp: float
    energy: float
    alive: bool
    statuses: list[StatusEffect]
    cooldowns: dict[str, int]


def snapshot(character: Character) -> CharacterSnapshot:
    return CharacterSnapshot(
        hp=character.hp,
        energy=character.energy,
        alive=character.alive,
        statuses=[s.model_copy() for s in character.statuses],
        cooldowns=dict(character.cooldowns),
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
    """Steps 1-5. Returns a TurnRecord; `needs_action()` tells the caller
    whether to collect a decision before calling resolve_action."""
    actor_id = state.current_actor_id()
    actor = state.characters[actor_id]
    record = TurnRecord(
        step_count=state.step_count,
        round_number=state.round_number,
        actor_id=actor_id,
        before=_snapshot_all(state),
    )

    if not actor.alive:
        record.skipped_reason = "dead"
        return record

    # Defect B06 only: an EXTRA cooldown decrement at the top of the turn,
    # ahead of the legal-move check that resolve_action's caller runs
    # between resolve_pre and resolve_action -- on top of, not instead of,
    # the one resolve_post always applies at the end. See resolve_post's
    # docstring for why a double decrement is the realistic shape of this
    # bug.
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

    # Regen ticks after the death check -- defect B07 is precisely about
    # this ordering not being enough, because Regen may have been left on
    # a character who died in an EARLIER turn and was since revived.
    if actor.alive:
        regen_healed = tick_regen(actor)
        if regen_healed:
            record.dot_hot_damage.setdefault(actor_id, {})["regen"] = regen_healed
    elif defects.regen_survives_death and has_status(actor, StatusType.REGEN):
        # Defect: a dead character's stale Regen would still be present
        # (clear_statuses_on_death spared it), but a dead character cannot
        # be healed -- the bug surfaces on revive, not here. No-op by
        # design; this branch exists only to document why we do nothing.
        pass

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
    # Cooldown is set on use unconditionally; the decrement clock (steps
    # further along, per-turn) is what defect B06 mis-orders, not this.
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
                # Defect B09: the ability's own calculation ALSO folds in
                # elemental advantage before the general modifier pass
                # (apply_damage_modifiers) applies it again below.
                raw = raw * elemental_multiplier(ability.element, target.element)
            damage = apply_damage_modifiers(ability, actor, target, raw)
            damage = apply_shield_absorption(target, damage, defects)
            shield_after = get_status(target, StatusType.SHIELD)
            absorbed = ability.base_power - damage if shield_after is not None else 0.0
            target.hp = max(0.0, target.hp - damage)
            record.per_target_damage[target_id] = damage
            if damage < ability.base_power:
                record.per_target_shield_absorbed[target_id] = ability.base_power - damage

            if target.hp <= 0 and target.alive:
                target.alive = False
                clear_statuses_on_death(target, defects)

        if ability.applies is not None and target.alive:
            # Guard against re-attaching a status to a target this same
            # ability just killed (e.g. Cinder Burn: damage + Burn in one
            # cast) -- a dead target has already had its statuses cleared
            # by clear_statuses_on_death above and must stay clear.
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
    """Step 8 (end-of-turn bookkeeping) plus advancing the turn pointer.

    Cooldowns are always decremented here, once per own-turn, after the
    legal-move check for this turn's action has already run. Defect B06
    (cooldown_decrement_before_check) does NOT move this decrement away --
    it adds a SECOND one earlier in the pipeline (see
    apply_pre_check_cooldown_decrement, called from resolve_pre, before
    the legal-move check). That is what actually happens in real code: a
    decrement call gets added at the top of the turn during a refactor and
    the original one at the bottom is never removed. The result is a
    cooldown that drains twice as fast as intended, so a 2-turn cooldown
    ability becomes usable after just 1 of the actor's own turns.
    """
    actor = state.characters[record.actor_id]
    if actor.alive:
        decrement_cooldowns(actor)
    if actor.alive:
        decrement_status_durations(actor, defects)

    record.after = _snapshot_all(state)

    # Advance to the next turn-slot.
    state.turn_pointer += 1
    if state.turn_pointer >= len(state.turn_order):
        state.turn_pointer = 0
        state.round_number += 1
    state.step_count += 1

    check_battle_end(state)


def apply_pre_check_cooldown_decrement(state: BattleState, defects: DefectFlags) -> None:
    """Called from resolve_pre, ONLY when defect B06 is enabled, to apply
    an EXTRA cooldown decrement ahead of the legal-move check for this
    same turn -- resolve_post's own decrement still runs too."""
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
