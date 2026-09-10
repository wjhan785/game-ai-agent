"""Tests for agent/inner_loop.py's decide_and_act, entirely offline via
cassette "replay" mode -- no network, no spend. Cassette entries are
built from the exact request bodies decide_and_act will construct (same
SYSTEM_PROMPT, same _format_context, same agent.llm._request_body), so
these tests exercise the real request/response wiring, not a mock of it.
"""
from __future__ import annotations

import json

from engine.defects import DefectFlags
from engine.engine import BattleEngine
from agent.inner_loop import DECISION_TOOL_NAMES, SYSTEM_PROMPT, _format_context, decide_and_act
from agent.llm import ROLE_CONFIGS, Cassette, _request_body
from agent.tools import ToolDispatcher, build_tool_schemas


def _fake_response(tool_name: str, args: dict, call_id: str = "call_1"):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 50},
    }


def _setup(scenario_id="S1", seed=1, turn_cap=20):
    engine = BattleEngine(scenario_id, seed, DefectFlags(), turn_cap)
    dispatcher = ToolDispatcher(engine)
    actor_id = engine.state.current_actor_id()
    tools = build_tool_schemas(DECISION_TOOL_NAMES)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"It is {actor_id}'s turn.\n\n{_format_context(dispatcher)}"},
    ]
    return dispatcher, actor_id, tools, messages


def test_decide_and_act_take_action_on_first_call(tmp_path):
    dispatcher, actor_id, tools, messages = _setup()
    options = dispatcher.list_legal_actions()["options"]
    chosen = options[0]

    request = _request_body(ROLE_CONFIGS["inner"], messages, tools, temperature=0.2)
    cassette = Cassette(dir_path=tmp_path)
    cassette.save(
        request,
        _fake_response(
            "take_action",
            {"actor_id": actor_id, "ability_name": chosen["ability_name"], "target_id": chosen["target_id"], "reasoning": "test"},
        ),
    )

    outcome = decide_and_act(dispatcher, mode="replay", cassette=cassette)
    assert outcome.llm_calls == 1
    assert outcome.flags_raised == 0
    assert outcome.fell_back_to_first_legal is False
    assert outcome.take_action_result is not None
    assert "error" not in outcome.take_action_result
    assert len(dispatcher.engine.log) == 1


def test_decide_and_act_falls_back_when_cassette_is_empty(tmp_path):
    dispatcher, actor_id, tools, messages = _setup()
    cassette = Cassette(dir_path=tmp_path)  # no entries saved

    outcome = decide_and_act(dispatcher, mode="replay", cassette=cassette)
    assert outcome.llm_calls == 1
    assert outcome.fell_back_to_first_legal is True
    assert outcome.take_action_result is not None
    assert "error" not in outcome.take_action_result
    assert len(dispatcher.engine.log) == 1  # the fallback action still advanced the engine


def test_decide_and_act_flag_anomaly_then_take_action(tmp_path):
    dispatcher, actor_id, tools, messages = _setup()
    options = dispatcher.list_legal_actions()["options"]
    chosen = options[0]
    cassette = Cassette(dir_path=tmp_path)

    # Call 1: flag_anomaly.
    request_1 = _request_body(ROLE_CONFIGS["inner"], messages, tools, temperature=0.2)
    flag_args = {"description": "damage looked off", "severity": "low", "evidence": "observed 10 vs declared 8"}
    cassette.save(request_1, _fake_response("flag_anomaly", flag_args, call_id="call_1"))

    # Call 2: same messages plus the assistant's flag_anomaly call and its
    # tool result -- exactly what decide_and_act appends before retrying.
    flag_dispatch_result = {"recorded": True, "flag_index": 0}
    messages_2 = messages + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "flag_anomaly", "arguments": json.dumps(flag_args)}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(flag_dispatch_result, separators=(",", ":"))},
    ]
    request_2 = _request_body(ROLE_CONFIGS["inner"], messages_2, tools, temperature=0.2)
    take_args = {"actor_id": actor_id, "ability_name": chosen["ability_name"], "target_id": chosen["target_id"], "reasoning": "test"}
    cassette.save(request_2, _fake_response("take_action", take_args, call_id="call_2"))

    outcome = decide_and_act(dispatcher, mode="replay", cassette=cassette)
    assert outcome.llm_calls == 2
    assert outcome.flags_raised == 1
    assert outcome.fell_back_to_first_legal is False
    assert "error" not in outcome.take_action_result
    assert len(dispatcher.flags) == 1
    assert len(dispatcher.engine.log) == 1
