"""Bare single-turn inner loop: no cross-turn memory yet -- each decision is answered from a
fresh, minimal context: a fixed system prompt (the stable prefix,
byte-identical every call so DeepSeek's automatic prefix caching can
actually fire) plus this turn's state and legal actions (volatile,
appended last).

Only `take_action` and `flag_anomaly` are exposed as callable tools here.
`get_state` and `list_legal_actions` are fetched directly through the
same ToolDispatcher (a Python call, not an LLM round trip) and embedded
in the prompt -- their result is deterministic and the harness already
knows the model needs it every decision, so spending an LLM call just to
ask for it would be pure waste. A turn may call `flag_anomaly` zero or
more times before it must call `take_action` to actually submit the
decision and let the engine advance.
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel

from agent.llm import Cassette, SpendTracker, call_with_tools
from agent.tools import TOOL_MODELS, ToolDispatcher, build_tool_schemas

SYSTEM_PROMPT = """You are a QA agent controlling every character (both party and enemy) in a turn-based tactics battle, one decision at a time. Your goal is not to win the battle -- it is to explore combat interactions thoroughly and flag anything that looks inconsistent with an ability's declared numbers.

Each turn you are shown the current battle state and every legal (ability, target) option for whoever's turn it is, each with its declared energy cost, cooldown, element, and power. You must call `take_action` exactly once to submit the decision for the CURRENT actor's turn -- ability_name and target_id (when applicable) must be one of the listed legal options.

Before or instead of your first take_action call, you may call `flag_anomaly` if something you observe contradicts an ability's declared numbers or an earlier observation in this conversation -- a damage number that doesn't match power times the elemental multiplier, a status that failed to apply/refresh/expire as expected, a resource reading outside its bounds. Be specific: name the character, the number, and what you expected instead. You must still call take_action afterward to submit this turn's decision."""

DECISION_TOOL_NAMES = ["take_action", "flag_anomaly"]
MAX_TOOL_CALLS_PER_DECISION = 3


class DecisionOutcome(BaseModel):
    actor_id: str
    llm_calls: int
    flags_raised: int
    fell_back_to_first_legal: bool
    take_action_result: Optional[dict] = None
    reasoning: Optional[str] = None


def _format_context(dispatcher: ToolDispatcher) -> str:
    state = dispatcher.get_state()
    legal = dispatcher.list_legal_actions()
    return json.dumps({"state": state, "legal_actions": legal["options"]}, separators=(",", ":"))


def decide_and_act(
    dispatcher: ToolDispatcher,
    *,
    mode: str = "live",
    cassette: Optional[Cassette] = None,
    spend: Optional[SpendTracker] = None,
    allow_peak: bool = False,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_DECISION,
) -> DecisionOutcome:
    actor_id = dispatcher.engine.state.current_actor_id()
    tools = build_tool_schemas(DECISION_TOOL_NAMES)
    tool_models = {name: TOOL_MODELS[name] for name in DECISION_TOOL_NAMES}

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"It is {actor_id}'s turn.\n\n{_format_context(dispatcher)}"},
    ]

    llm_calls = 0
    flags_raised = 0
    take_action_result: Optional[dict] = None
    reasoning: Optional[str] = None

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
        if not result.ok or result.tool_name is None or result.args is None:
            break

        dispatch_result = dispatcher.dispatch(result.tool_name, result.args)

        if result.tool_name == "flag_anomaly":
            flags_raised += 1
        if result.tool_name == "take_action" and "error" not in dispatch_result:
            take_action_result = dispatch_result
            reasoning = result.args.get("reasoning")
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
        flags_raised=flags_raised,
        fell_back_to_first_legal=fell_back,
        take_action_result=take_action_result,
        reasoning=reasoning,
    )
