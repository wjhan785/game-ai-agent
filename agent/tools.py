"""Tool surface for both loops.

Small and flat by design -- third-party OpenAI-compatible tool calling
degrades on deeply nested schemas.

  inner (per turn):   take_action, flag_anomaly, ledger_write
  outer (per episode): ledger_read, ledger_write, reset_episode
  final review:        ledger_read, ledger_write, end_session

`get_state` and `list_legal_actions` exist as dispatcher methods but are
not offered to the model: their output is deterministic and needed every
decision, so agent/inner_loop.py puts it in the prompt directly rather
than spending an LLM round trip to fetch it.

`take_action` returns a precomputed DIFF, not a new state dump -- the
engine does the arithmetic, the model does the judgment. Each per-target
line pairs the ability's own DECLARED numbers (base_power, element) with
the OBSERVED damage the engine actually applied, so the model can catch a
mismatch between what an ability's tooltip promises and what it actually
did by comparing two numbers it's handed, not by re-deriving the engine's
own math.

`flag_anomaly` auto-attaches reproduction: the scenario, seed, round cap
and every action taken so far, written next to the ledger, so any flag can
be replayed to the exact state it was raised in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from agent.ledger import HYPOTHESIS_STATUSES, Ledger
from agent.llm import tool_schema
from engine import rules
from engine.elements import elemental_multiplier
from engine.engine import BattleEngine, IllegalActionError
from engine.models import Action
from engine.resolution import TurnRecord

HypothesisStatus = Literal["open", "testing", "confirmed", "refuted", "inconclusive"]

# --- Tool argument schemas -------------------------------------------------


class GetStateArgs(BaseModel):
    pass


class ListLegalActionsArgs(BaseModel):
    pass


class TakeActionArgs(BaseModel):
    # Field order matters: models tend to fill a schema's properties in
    # order, so the audit is written before the action is chosen.
    audit: str = Field(
        description=(
            "Before choosing: one short line per check of each turn-slot marked new and of this turn's "
            "start-of-turn ticks against the rules and declared data, with the arithmetic, marked ok or MISMATCH."
        )
    )
    anomalies: list[str] = Field(
        default_factory=list,
        description="Every MISMATCH from the audit, one per entry: the character, the rule, expected (with arithmetic) vs observed.",
    )
    actor_id: str = Field(description="Who you believe is acting this turn -- validated against whose turn it actually is.")
    ability_name: str
    target_id: Optional[str] = Field(
        default=None, description="Required for single-target abilities; omit for self/all-enemies abilities."
    )
    reasoning: str = Field(description="One sentence: why this action, right now.")


class FlagAnomalyArgs(BaseModel):
    description: str = Field(description="What looks wrong, in plain language, naming the character and the rule.")
    severity: Literal["low", "medium", "high"]
    evidence: str = Field(description="The numbers: expected value (with the arithmetic) vs observed value, and where it was observed.")


class LedgerReadArgs(BaseModel):
    kind: Literal["hypotheses", "flags", "coverage", "episodes"]


class LedgerWriteArgs(BaseModel):
    status: HypothesisStatus
    hypothesis_id: Optional[int] = Field(default=None, description="Existing hypothesis to update; omit to create a new one.")
    text: Optional[str] = Field(default=None, description="The hypothesis itself, when creating one: one interaction, one suspected wrong outcome.")
    note: str = Field(default="", description="Evidence or reasoning behind this status.")


class ResetEpisodeArgs(BaseModel):
    scenario_id: str
    seed: int = Field(description="Any integer 0-1000000; varies starting HP/energy by up to 10-15%.")
    goal: str = Field(description="What the tactical agent should set up, turn by turn: which characters do what, in what order, and what to watch.")
    focus_actor_ids: list[str] = Field(description="Characters whose every turn the tactical agent decides.")
    hypothesis_ids: list[int] = Field(description="Hypotheses this episode tests.")


class EndSessionArgs(BaseModel):
    summary: str = Field(description="Two or three sentences: what the campaign found and what remains untested.")


TOOL_MODELS: dict[str, type[BaseModel]] = {
    "get_state": GetStateArgs,
    "list_legal_actions": ListLegalActionsArgs,
    "take_action": TakeActionArgs,
    "flag_anomaly": FlagAnomalyArgs,
    "ledger_read": LedgerReadArgs,
    "ledger_write": LedgerWriteArgs,
    "reset_episode": ResetEpisodeArgs,
    "end_session": EndSessionArgs,
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
        "Audit the new turn-slots, then submit the action for the current "
        "actor's turn. Every entry in anomalies is filed as a flag. Returns "
        "a diff: HP/status changes, and the declared-vs-observed damage for "
        "each target."
    ),
    "flag_anomaly": (
        "Report an observation that contradicts the rules or an ability's "
        "declared data. The current action trace is attached automatically "
        "as reproduction."
    ),
    "ledger_read": "Read the exploration ledger: all hypotheses, all agent flags, all coverage seen, or all episodes run.",
    "ledger_write": (
        "Create a hypothesis (omit hypothesis_id, give text) or update an "
        "existing one's status (give hypothesis_id), with a note of the "
        "evidence."
    ),
    "reset_episode": "Start the next episode: scenario, seed, goal for the tactical agent, focus characters, hypotheses under test.",
    "end_session": "Finish the campaign after the final review.",
}

INNER_TOOL_NAMES = ["take_action", "flag_anomaly", "ledger_write"]
PLANNER_TOOL_NAMES = ["ledger_read", "ledger_write", "reset_episode"]
REVIEW_TOOL_NAMES = ["ledger_read", "ledger_write", "end_session"]


def build_tool_schemas(names: Optional[list[str]] = None) -> list[dict]:
    names = names or list(TOOL_MODELS)
    return [tool_schema(name, TOOL_DESCRIPTIONS[name], TOOL_MODELS[name]) for name in names]


# --- Ledger tools (shared by both loops) -------------------------------------


def ledger_read(ledger: Ledger, kind: str) -> dict:
    if kind == "hypotheses":
        return {"hypotheses": ledger.hypotheses()}
    if kind == "flags":
        return {
            "agent_flags": [
                {k: f[k] for k in ("id", "episode", "turn", "severity", "description", "evidence")}
                for f in ledger.flags(source="agent")
            ]
        }
    if kind == "coverage":
        return {"coverage": [{"combo": c["combo_sig"], "first_seen_episode": c["first_seen_ep"]} for c in ledger.coverage()]}
    if kind == "episodes":
        return {
            "episodes": [
                {
                    "id": e["id"],
                    "scenario": e["scenario"],
                    "seed": e["seed"],
                    "goal": e["plan"].get("goal"),
                    "outcome": e["outcome"],
                    "actions_used": e["actions_used"],
                    "agent_flags": len(e["summary"].get("agent_flags", [])),
                }
                for e in ledger.episodes()
            ]
        }
    return {"error": f"unknown kind {kind!r}"}


def ledger_write(
    ledger: Ledger,
    episode_id: Optional[int],
    *,
    status: str,
    hypothesis_id: Optional[int] = None,
    text: Optional[str] = None,
    note: str = "",
) -> dict:
    if status not in HYPOTHESIS_STATUSES:
        return {"error": f"status must be one of {list(HYPOTHESIS_STATUSES)}"}
    if hypothesis_id is None:
        if not text or not text.strip():
            return {"error": "creating a hypothesis needs text"}
        new_id = ledger.add_hypothesis(text.strip(), created_ep=episode_id, status=status, notes=note)
        return {"created": new_id}
    if not ledger.update_hypothesis(hypothesis_id, status, note, episode_id):
        return {"error": f"no hypothesis with id {hypothesis_id}"}
    return {"updated": hypothesis_id, "status": status}


# --- Dispatcher --------------------------------------------------------


class ToolDispatcher:
    """Wraps one BattleEngine, exposing the inner-loop tool surface and a
    generic `dispatch(name, args_dict) -> dict` entry point that consumes
    an agent.llm.CallResult's parsed args directly.

    With a ledger attached, `flag_anomaly` persists the flag (plus a
    reproduction file under `flags_dir`) and `ledger_write` is live;
    without one, flags are kept in memory only (`self.flags`)."""

    def __init__(
        self,
        engine: BattleEngine,
        *,
        ledger: Optional[Ledger] = None,
        episode_id: Optional[int] = None,
        flags_dir: Optional[str | Path] = None,
    ):
        self.engine = engine
        self.ledger = ledger
        self.episode_id = episode_id
        self.flags_dir = Path(flags_dir) if flags_dir is not None else None
        self.flags: list[dict] = []

    def dispatch(self, name: str, args: dict) -> dict:
        handler = {
            "get_state": self.get_state,
            "list_legal_actions": self.list_legal_actions,
            "take_action": self.take_action,
            "flag_anomaly": self.flag_anomaly,
            "ledger_write": self.ledger_write,
            "ledger_read": self.ledger_read,
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
                    "speed": c.speed,
                    "action_value": c.action_value,
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
            "forecast": [{"id": cid, "action_value": av} for cid, av in state.turn_forecast(6)],
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
        self,
        actor_id: str,
        ability_name: str,
        reasoning: str,
        target_id: Optional[str] = None,
        audit: str = "",
        anomalies: Optional[list[str]] = None,
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
        check = rules.check_legal(self.engine.state, action)
        if not check.legal:
            return {"error": f"illegal action: {check.reason}"}
        # Filed only once the action is known to be legal, so a retried
        # take_action can't file the same audit findings twice.
        filed = [
            self.flag_anomaly(description=a.strip(), severity="medium", evidence=audit)
            for a in (anomalies or [])
            if a and a.strip()
        ]
        try:
            record = self.engine.take_action(action)
        except IllegalActionError as exc:
            return {"error": f"illegal action: {exc.reason}"}
        diff = self._summarize(record)
        if filed:
            diff["flags_filed"] = len(filed)
        return diff

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
        state = self.engine.state
        flag = {
            "description": description,
            "severity": severity,
            "evidence": evidence,
            "step_count": state.step_count,
            "round_number": state.round_number,
            "trace": [r.model_dump(mode="json") for r in self.engine.log],
        }
        self.flags.append(flag)
        result: dict = {"recorded": True, "flag_index": len(self.flags) - 1}

        if self.ledger is not None and self.episode_id is not None:
            flag_id = self.ledger.add_flag(
                episode=self.episode_id,
                turn=state.step_count,
                source="agent",
                description=description,
                severity=severity,
                evidence=evidence,
            )
            if self.flags_dir is not None:
                trace_path = self.flags_dir / f"flag_{flag_id:04d}.json"
                self._write_repro(trace_path, flag_id, description)
                self.ledger.set_flag_trace(flag_id, str(trace_path))
            flag["ledger_id"] = flag_id
            result["flag_id"] = flag_id
        return result

    def _write_repro(self, path: Path, flag_id: int, description: str) -> None:
        state = self.engine.state
        repro = {
            "flag_id": flag_id,
            "episode": self.episode_id,
            "description": description,
            "scenario_id": state.scenario_id,
            "seed": state.seed,
            "turn_cap": self.engine.turn_cap,
            "step_count": state.step_count,
            "actions": [r.action.model_dump(mode="json") for r in self.engine.log if r.action is not None],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(repro, indent=2), encoding="utf-8")

    def ledger_write(
        self,
        status: str,
        hypothesis_id: Optional[int] = None,
        text: Optional[str] = None,
        note: str = "",
    ) -> dict:
        if self.ledger is None:
            return {"error": "no ledger attached"}
        return ledger_write(self.ledger, self.episode_id, status=status, hypothesis_id=hypothesis_id, text=text, note=note)

    def ledger_read(self, kind: str) -> dict:
        if self.ledger is None:
            return {"error": "no ledger attached"}
        return ledger_read(self.ledger, kind)
