"""Tool surface for the inner (per-turn) loop, driving one BattleEngine.

Small and flat by design -- third-party OpenAI-compatible tool calling
degrades on deeply nested schemas. Four tools cover Week 2's bare
single-turn agent: `get_state`, `list_legal_actions`, `take_action`,
`flag_anomaly`. `ledger_read`/`ledger_write` and `reset_episode` join in
Week 3 once agent/ledger.py and the outer loop exist.

`take_action` returns a precomputed DIFF, not a new state dump -- the
engine does the arithmetic, the model does the judgment. Each per-target
line pairs the ability's own DECLARED numbers (base_power, element) with
the OBSERVED damage the engine actually applied, so the model can catch a
mismatch between what an ability's tooltip promises and what it actually
did by comparing two numbers it's handed, not by re-deriving the engine's
own math.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from agent.llm import tool_schema
from engine.elements import elemental_multiplier
from engine.engine import BattleEngine, IllegalActionError
from engine.models import Action
from engine.resolution import TurnRecord

# --- Tool argument schemas -------------------------------------------------


class GetStateArgs(BaseModel):
    pass


class ListLegalActionsArgs(BaseModel):
    pass


class TakeActionArgs(BaseModel):
    actor_id: str = Field(description="Who you believe is acting this turn -- validated against whose turn it actually is.")
    ability_name: str
    target_id: Optional[str] = Field(
        default=None, description="Required for single-target abilities; omit for self/all-enemies abilities."
    )
    reasoning: str = Field(description="One sentence: why this action, right now.")


class FlagAnomalyArgs(BaseModel):
    description: str = Field(description="What looks wrong, in plain language.")
    severity: Literal["low", "medium", "high"]
    evidence: str = Field(description="The specific observation that triggered this flag -- a number, an entity, a comparison.")


TOOL_MODELS: dict[str, type[BaseModel]] = {
    "get_state": GetStateArgs,
    "list_legal_actions": ListLegalActionsArgs,
    "take_action": TakeActionArgs,
    "flag_anomaly": FlagAnomalyArgs,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_state": (
        "Get a compact summary of the current battle: whose turn it is, "
        "and each character's HP, energy, statuses, and cooldowns."
    ),
    "list_legal_actions": (
        "List every legal (ability, target) pair for whoever's turn it is "
        "now, with each option's declared cost, cooldown, element, power, "
        "and applied status."
    ),
    "take_action": (
        "Submit the action for the current actor's turn. Returns a diff: "
        "HP/status changes, and the declared-vs-observed damage for each "
        "target."
    ),
    "flag_anomaly": (
        "Report something that looks wrong -- a number that doesn't match "
        "its ability's declared tooltip, a status that behaved "
        "unexpectedly, anything inconsistent. The current action trace is "
        "attached automatically as reproduction evidence."
    ),
}


def build_tool_schemas(names: Optional[list[str]] = None) -> list[dict]:
    names = names or list(TOOL_MODELS)
    return [tool_schema(name, TOOL_DESCRIPTIONS[name], TOOL_MODELS[name]) for name in names]


# --- Dispatcher --------------------------------------------------------


class ToolDispatcher:
    """Wraps one BattleEngine, exposing the inner-loop tool surface and a
    generic `dispatch(name, args_dict) -> dict` entry point that consumes
    an agent.llm.CallResult's parsed args directly. Also collects
    `flag_anomaly` calls in memory (`self.flags`) -- Week 3's ledger
    persists these across episodes; this is the Week-2 minimum."""

    def __init__(self, engine: BattleEngine):
        self.engine = engine
        self.flags: list[dict] = []

    def dispatch(self, name: str, args: dict) -> dict:
        handler = {
            "get_state": self.get_state,
            "list_legal_actions": self.list_legal_actions,
            "take_action": self.take_action,
            "flag_anomaly": self.flag_anomaly,
        }.get(name)
        if handler is None:
            raise ValueError(f"unknown tool '{name}'")
        return handler(**args)

    def get_state(self) -> dict:
        state = self.engine.state
        characters = []
        for c in state.characters.values():
            if not c.alive:
                characters.append({"id": c.id, "name": c.name, "team": c.team, "alive": False})
                continue
            characters.append(
                {
                    "id": c.id,
                    "name": c.name,
                    "team": c.team,
                    "element": c.element.value,
                    "alive": True,
                    "hp": c.hp,
                    "max_hp": c.max_hp,
                    "energy": c.energy,
                    "max_energy": c.max_energy,
                    "statuses": [
                        {
                            "type": s.status_type.value,
                            "magnitude": s.magnitude,
                            "turns_remaining": s.turns_remaining,
                        }
                        for s in c.statuses
                    ],
                    "cooldowns": {k: v for k, v in c.cooldowns.items() if v > 0},
                }
            )
        return {
            "round_number": state.round_number,
            "step_count": state.step_count,
            "current_actor_id": state.current_actor_id() if not state.finished else None,
            "finished": state.finished,
            "outcome": state.outcome,
            "characters": characters,
        }

    def list_legal_actions(self) -> dict:
        if self.engine.state.finished:
            return {"actor_id": None, "options": []}
        actor_id = self.engine.state.current_actor_id()
        options = []
        for action in self.engine.legal_actions():
            actor = self.engine.state.characters[action.actor_id]
            ability = actor.ability_by_name(action.ability_name)
            assert ability is not None
            entry: dict = {
                "ability_name": action.ability_name,
                "target_id": action.target_id,
                "energy_cost": ability.energy_cost,
                "cooldown": ability.cooldown,
                "element": ability.element.value,
                "base_power": ability.base_power,
                "target_type": ability.target_type.value,
            }
            if ability.applies is not None:
                entry["applies"] = {
                    "status_type": ability.applies.status_type.value,
                    "magnitude": ability.applies.magnitude,
                    "duration": ability.applies.duration,
                }
            if action.target_id is not None:
                target = self.engine.state.characters.get(action.target_id)
                if target is not None:
                    entry["target_name"] = target.name
                    entry["target_element"] = target.element.value
                    if ability.base_power > 0:
                        entry["declared_expected_damage"] = round(
                            ability.base_power * elemental_multiplier(ability.element, target.element), 2
                        )
            options.append(entry)
        return {"actor_id": actor_id, "options": options}

    def take_action(
        self, actor_id: str, ability_name: str, reasoning: str, target_id: Optional[str] = None
    ) -> dict:
        if self.engine.state.finished:
            return {"error": "battle already finished"}
        current = self.engine.state.current_actor_id()
        if actor_id != current:
            return {
                "error": (
                    f"it is {current}'s turn, not {actor_id}'s -- call "
                    "get_state() or list_legal_actions() and retry"
                )
            }
        action = Action(actor_id=actor_id, ability_name=ability_name, target_id=target_id)
        try:
            record = self.engine.take_action(action)
        except IllegalActionError as exc:
            return {"error": f"illegal action: {exc.reason}"}
        return self._summarize(record)

    def _summarize(self, record: TurnRecord) -> dict:
        actor = self.engine.state.characters.get(record.actor_id)
        diff: dict = {"actor_id": record.actor_id, "round_number": record.round_number}

        if record.dot_hot_damage:
            diff["dot_hot_damage"] = record.dot_hot_damage
        if record.energy_regen_delta:
            diff["energy_regen_delta"] = record.energy_regen_delta

        if record.ability_effect is not None:
            effect = record.ability_effect
            ability = actor.ability_by_name(effect.ability_name) if actor else None
            per_target = []
            for target_id in effect.target_ids:
                before = record.before.get(target_id)
                after = record.after.get(target_id)
                entry: dict = {"target_id": target_id}
                if before is not None and after is not None:
                    entry["hp_before"] = before.hp
                    entry["hp_after"] = after.hp
                    entry["hp_delta"] = round(after.hp - before.hp, 2)
                if target_id in effect.per_target_damage:
                    entry["observed_damage"] = effect.per_target_damage[target_id]
                    target = self.engine.state.characters.get(target_id)
                    if ability is not None and target is not None:
                        entry["declared_expected_damage"] = round(
                            ability.base_power * elemental_multiplier(ability.element, target.element), 2
                        )
                if target_id in effect.per_target_shield_absorbed:
                    entry["shield_absorbed"] = effect.per_target_shield_absorbed[target_id]
                per_target.append(entry)
            diff["ability"] = effect.ability_name
            diff["per_target"] = per_target
            if effect.applied_status:
                diff["applied_status"] = effect.applied_status
            if effect.revived:
                diff["revived"] = effect.revived

        if record.deaths:
            diff["deaths"] = record.deaths
        if record.invariant_violations:
            diff["invariant_violations"] = record.invariant_violations
        return diff

    def flag_anomaly(self, description: str, severity: str, evidence: str) -> dict:
        flag = {
            "description": description,
            "severity": severity,
            "evidence": evidence,
            "step_count": self.engine.state.step_count,
            "round_number": self.engine.state.round_number,
            "trace": [r.model_dump(mode="json") for r in self.engine.log],
        }
        self.flags.append(flag)
        return {"recorded": True, "flag_index": len(self.flags) - 1}
