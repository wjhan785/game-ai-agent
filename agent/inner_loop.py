"""The inner (per-turn) tactical loop: decide one character's turn toward
the current episode goal, and flag anything that contradicts the rules.

Request layout is ordered for DeepSeek's automatic prefix caching:
  1. SYSTEM_PROMPT        -- identical on every inner call in every episode
  2. episode brief        -- identical on every call within one episode
                             (roster with declared ability data, goal,
                             hypotheses under test, focus characters)
  3. turn context         -- volatile: a bounded window of recent turn-slots,
                             this turn so far, current state, legal actions
The window is bounded so the volatile tail never grows with episode
length, and nothing in 1-2 varies call to call (no timestamps, no
unordered iteration).

`get_state`/`list_legal_actions` are not offered as tools: their output
goes straight into the turn context. A turn may call `flag_anomaly` and
`ledger_write` before it must call `take_action` to submit the decision.
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from agent.llm import Cassette, SpendTracker, UsageStats, add_usage, call_with_tools
from agent.prompts import (
    AUDIT_CHECKLIST,
    OBSERVATION_FORMAT,
    RULES_SPEC,
    compact_legal,
    compact_state,
    dumps,
    format_roster,
    pending_turn,
    recent_events,
)
from agent.tools import TOOL_MODELS, ToolDispatcher, build_tool_schemas
from engine.engine import BattleEngine

SYSTEM_PROMPT = f"""You are an automated QA tester for a turn-based tactics combat system. You control every character on both teams, one decision at a time. Winning does not matter. Your job is to steer the battle into situations that exercise the rules below -- especially where statuses, elements, resources, death and turn order interact -- and to report observed behavior that contradicts the rules or an ability's declared data.

{RULES_SPEC}

{OBSERVATION_FORMAT}

{AUDIT_CHECKLIST}

Each decision
- You are given the episode brief, the recent turn-slots (those marked "new" happened since your last decision), the current turn so far (the acting character's start-of-turn ticks), the current state, and the acting character's legal abilities with their valid targets.
- Verify before you act. Setting up an interaction proves nothing until its numbers are checked: in take_action's audit, check every new turn-slot and this turn's ticks against the checklist, one short line each with the arithmetic, marked ok or MISMATCH. Put every MISMATCH in anomalies -- each is filed as a flag. Legitimate reductions (Weaken, Shield, elemental disadvantage, the HP floor at 0, the max-HP cap on healing) are not mismatches. Do not repeat an issue listed under flags raised this episode.
- Then choose: ability_name and target_id from the legal list; omit target_id for abilities listed with "-". Call take_action exactly once.
- For a finding that needs more room than one line, call flag_anomaly (with severity and evidence) before take_action instead. You may call ledger_write to record a hypothesis (status "open") for something suspicious you cannot confirm yet.
- Some turns may be played by a scripted policy without you; they still appear among the new turn-slots and still need auditing."""

MAX_TOOL_CALLS_PER_DECISION = 4
DEFAULT_WINDOW = 6


def episode_brief(
    engine: BattleEngine,
    *,
    goal: str = "",
    hypotheses: Optional[list[dict]] = None,
    focus_actor_ids: Optional[list[str]] = None,
    confirmed: Optional[list[dict]] = None,
) -> str:
    state = engine.state
    lines = [
        "EPISODE BRIEF",
        f"Scenario {state.scenario_id}, seed {state.seed}, round cap {state.turn_cap}.",
        "Roster (declared data):",
        format_roster(state),
        f"Goal: {goal.strip() if goal and goal.strip() else 'No planner goal this episode: explore interactions freely.'}",
    ]
    if hypotheses:
        lines.append("Hypotheses under test:")
        lines.extend(f"- #{h['id']}: {h['text']}" for h in hypotheses)
    if focus_actor_ids:
        lines.append(f"Focus characters (you decide every one of their turns): {', '.join(focus_actor_ids)}")
    if confirmed:
        lines.append("Already confirmed in earlier episodes -- do not re-report these; spend your attention on anything else:")
        lines.extend(f"- #{h['id']}: {h['text']}" for h in confirmed)
    return "\n".join(lines)


def turn_context(dispatcher: ToolDispatcher, window: int = DEFAULT_WINDOW, audit_from: int = 0) -> str:
    engine = dispatcher.engine
    state = engine.state
    actor_id = state.current_actor_id()
    flagged = [f["description"] for f in dispatcher.flags]
    parts = [
        "RECENT TURN-SLOTS (oldest first):",
        "\n".join(dumps(ev) for ev in recent_events(engine.log, state, window, audit_from)) or "(none yet)",
        f"CURRENT TURN: {actor_id} ({state.characters[actor_id].name})",
        dumps(pending_turn(engine.pending_record(), state)),
        "STATE:",
        dumps(compact_state(state)),
        "LEGAL (ability -> targets):",
        dumps(compact_legal(dispatcher.list_legal_actions()["options"])),
        "FLAGS RAISED THIS EPISODE: " + (dumps(flagged) if flagged else "none"),
    ]
    return "\n".join(parts)


def build_messages(
    dispatcher: ToolDispatcher, brief: str, window: int = DEFAULT_WINDOW, audit_from: int = 0
) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{brief}\n\n---\n\n{turn_context(dispatcher, window, audit_from)}"},
    ]


class DecisionOutcome(BaseModel):
    actor_id: str
    llm_calls: int
    flags_raised: int
    fell_back_to_first_legal: bool
    take_action_result: Optional[dict] = None
    reasoning: Optional[str] = None
    audit: Optional[str] = None
    flags: list[dict] = Field(default_factory=list)
    hypotheses_written: list[dict] = Field(default_factory=list)
    failed_calls: int = 0
    retries: int = 0
    usage: UsageStats = Field(default_factory=UsageStats)


def decide_and_act(
    dispatcher: ToolDispatcher,
    brief: str,
    *,
    window: int = DEFAULT_WINDOW,
    audit_from: int = 0,
    mode: str = "live",
    cassette: Optional[Cassette] = None,
    spend: Optional[SpendTracker] = None,
    allow_peak: bool = False,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_DECISION,
) -> DecisionOutcome:
    actor_id = dispatcher.engine.state.current_actor_id()
    tool_names = ["take_action", "flag_anomaly"] + (["ledger_write"] if dispatcher.ledger is not None else [])
    tools = build_tool_schemas(tool_names)
    tool_models = {name: TOOL_MODELS[name] for name in tool_names}
    messages = build_messages(dispatcher, brief, window, audit_from)

    llm_calls = failed_calls = retries = 0
    usage = UsageStats()
    flags: list[dict] = []
    hypotheses_written: list[dict] = []
    take_action_result: Optional[dict] = None
    reasoning: Optional[str] = None
    audit: Optional[str] = None

    for _ in range(max_tool_calls):
        result = call_with_tools(
            role="inner",
            messages=messages,
            tools=tools,
            tool_models=tool_models,
            mode=mode,
            cassette=cassette,
            spend=spend,
            allow_peak=allow_peak,
        )
        llm_calls += 1
        retries += result.retries_used
        usage = add_usage(usage, result.usage)
        if not result.ok or result.tool_name is None or result.args is None:
            failed_calls += 1
            break

        dispatch_result = dispatcher.dispatch(result.tool_name, result.args)

        if result.tool_name == "flag_anomaly":
            flags.append({k: result.args[k] for k in ("description", "severity", "evidence")})
        if result.tool_name == "ledger_write" and "error" not in dispatch_result:
            hypotheses_written.append({**result.args, **dispatch_result})
        if result.tool_name == "take_action" and "error" not in dispatch_result:
            take_action_result = dispatch_result
            reasoning = result.args.get("reasoning")
            audit = result.args.get("audit")
            flags.extend(
                {"description": a.strip(), "severity": "medium", "evidence": audit}
                for a in result.args.get("anomalies") or []
                if a and a.strip()
            )
            break

        # Continue the conversation: echo the assistant's tool call, then
        # its result, and ask again -- this is how one turn can flag an
        # anomaly (or retry an illegal action) and still submit a decision.
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": result.tool_call_id,
                        "type": "function",
                        "function": {"name": result.tool_name, "arguments": result.raw_arguments},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": json.dumps(dispatch_result, separators=(",", ":")),
            }
        )

    fell_back = False
    if take_action_result is None:
        # Exhausted attempts without a successful take_action -- fall back
        # to the first legal option so the episode keeps moving. This is
        # a reported metric (fell_back_to_first_legal), not a crash.
        fell_back = True
        reasoning = "fallback: exhausted tool-call attempts this decision"
        options = dispatcher.list_legal_actions()["options"]
        first = options[0]
        take_action_result = dispatcher.take_action(
            actor_id=actor_id,
            ability_name=first["ability_name"],
            target_id=first["target_id"],
            reasoning=reasoning,
        )

    return DecisionOutcome(
        actor_id=actor_id,
        llm_calls=llm_calls,
        flags_raised=len(flags),
        fell_back_to_first_legal=fell_back,
        take_action_result=take_action_result,
        reasoning=reasoning,
        audit=audit,
        flags=flags,
        hypotheses_written=hypotheses_written,
        failed_calls=failed_calls,
        retries=retries,
        usage=usage,
    )
