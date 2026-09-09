"""The legal-move checker: energy cost, cooldown, and valid targeting,
enforced before anything is applied.

This is deliberately the single choke point for "can this action happen at
all." resolution.py calls `check_legal` before applying an action, and
`list_legal_actions` (used by both the agent tool surface and the
baselines) is built by filtering every (ability, target) combination
through it. Nothing here decides whose turn it is or whether the actor is
stunned -- that is resolution.py's job, one layer up.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.effects import get_status
from engine.models import Action, Ability, BattleState, StatusType, TargetType


@dataclass(frozen=True)
class LegalCheckResult:
    legal: bool
    reason: str = ""


def _valid_targets_for(state: BattleState, actor_id: str, ability: Ability) -> list[str]:
    actor = state.characters[actor_id]
    if ability.target_type == TargetType.SELF:
        return [actor_id] if actor.alive else []
    if ability.target_type == TargetType.ONE_ENEMY:
        opposing = "enemy" if actor.team == "party" else "party"
        return [c.id for c in state.alive_on_team(opposing)]
    if ability.target_type == TargetType.ONE_ALLY:
        if ability.is_revive:
            return [
                c.id
                for c in state.characters.values()
                if c.team == actor.team and not c.alive
            ]
        return [c.id for c in state.alive_on_team(actor.team)]
    if ability.target_type == TargetType.ALL_ENEMIES:
        opposing = "enemy" if actor.team == "party" else "party"
        return [c.id for c in state.alive_on_team(opposing)]
    return []


def check_legal(state: BattleState, action: Action) -> LegalCheckResult:
    if action.actor_id not in state.characters:
        return LegalCheckResult(False, "unknown actor")
    actor = state.characters[action.actor_id]
    if not actor.alive:
        return LegalCheckResult(False, "actor is dead")
    if action.actor_id != state.current_actor_id():
        return LegalCheckResult(False, "not this actor's turn")

    ability = actor.ability_by_name(action.ability_name)
    if ability is None:
        return LegalCheckResult(False, "unknown ability")

    if actor.energy < ability.energy_cost:
        return LegalCheckResult(False, "insufficient energy")

    if actor.cooldowns.get(ability.name, 0) > 0:
        return LegalCheckResult(False, "ability on cooldown")

    valid_targets = _valid_targets_for(state, action.actor_id, ability)
    if ability.target_type in (TargetType.SELF, TargetType.ALL_ENEMIES):
        # target_id is informational only for these types; no valid targets
        # means the ability cannot be used (e.g. all enemies already dead).
        if not valid_targets:
            return LegalCheckResult(False, "no valid targets")
        return LegalCheckResult(True)

    if action.target_id is None or action.target_id not in valid_targets:
        return LegalCheckResult(False, "invalid target")

    return LegalCheckResult(True)


def list_legal_actions(state: BattleState) -> list[Action]:
    """Every legal (ability, target) pair for whoever's turn it is."""
    actor_id = state.current_actor_id()
    actor = state.characters[actor_id]
    if not actor.alive:
        return []

    actions: list[Action] = []
    for ability in actor.abilities:
        targets = _valid_targets_for(state, actor_id, ability)
        if ability.target_type in (TargetType.SELF, TargetType.ALL_ENEMIES):
            candidate = Action(actor_id=actor_id, ability_name=ability.name, target_id=None)
            if check_legal(state, candidate).legal:
                actions.append(candidate)
        else:
            for target_id in targets:
                candidate = Action(
                    actor_id=actor_id, ability_name=ability.name, target_id=target_id
                )
                if check_legal(state, candidate).legal:
                    actions.append(candidate)
    return actions
