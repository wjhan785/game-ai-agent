"""agent/inner_loop.py offline, via cassettes keyed on the real requests."""

import json

from engine.engine import BattleEngine
from engine.models import Action
from agent.inner_loop import SYSTEM_PROMPT, build_messages, decide_and_act, episode_brief, turn_context
from agent.ledger import Ledger
from agent.llm import ROLE_CONFIGS, Cassette, _request_body
from agent.tools import ToolDispatcher, build_tool_schemas
from tests.conftest import advance_to_actor


def _fake_response(tool_name: str, args: dict, call_id: str = "call_1"):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "prompt_cache_hit_tokens": 32, "prompt_cache_miss_tokens": 18},
    }


def _setup(ledger=None, episode_id=None, tmp_path=None):
    engine = BattleEngine("S1", seed=1, turn_cap=20)
    dispatcher = ToolDispatcher(
        engine, ledger=ledger, episode_id=episode_id,
        flags_dir=(tmp_path / "flags") if tmp_path is not None else None,
    )
    brief = episode_brief(engine, goal="test goal", focus_actor_ids=["p1"])
    names = ["take_action", "flag_anomaly"] + (["ledger_write"] if ledger is not None else [])
    return dispatcher, brief, build_tool_schemas(names)


def _take(dispatcher, anomalies=None):
    actor = dispatcher.engine.state.current_actor_id()
    opt = dispatcher.list_legal_actions()["options"][0]
    return {
        "audit": "no new turn-slots yet: ok",
        "anomalies": anomalies or [],
        "actor_id": actor,
        "ability_name": opt["ability_name"],
        "target_id": opt["target_id"],
        "reasoning": "test",
    }


def _follow_up(messages, name, args, call_id, dispatch_result):
    return messages + [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(dispatch_result, separators=(",", ":"))},
    ]


def test_take_action_on_first_call(tmp_path):
    dispatcher, brief, tools = _setup()
    messages = build_messages(dispatcher, brief)
    cassette = Cassette(dir_path=tmp_path)
    cassette.save(_request_body(ROLE_CONFIGS["inner"], messages, tools, 0.2), _fake_response("take_action", _take(dispatcher)))

    outcome = decide_and_act(dispatcher, brief, mode="replay", cassette=cassette)
    assert outcome.llm_calls == 1
    assert outcome.fell_back_to_first_legal is False
    assert outcome.reasoning == "test"
    assert outcome.audit == "no new turn-slots yet: ok"
    assert outcome.usage.cache_hit_tokens == 32
    assert len(dispatcher.engine.log) == 1


def test_audit_anomalies_are_filed_as_flags(tmp_path):
    dispatcher, brief, tools = _setup()
    messages = build_messages(dispatcher, brief)
    cassette = Cassette(dir_path=tmp_path)
    finding = "e1 took 49.5 from Volt Lance; tooltip 22 x 1.5 = 33"
    cassette.save(_request_body(ROLE_CONFIGS["inner"], messages, tools, 0.2),
                  _fake_response("take_action", _take(dispatcher, anomalies=[finding, "  "])))

    outcome = decide_and_act(dispatcher, brief, mode="replay", cassette=cassette)
    assert [f["description"] for f in dispatcher.flags] == [finding]
    assert outcome.flags == [{"description": finding, "severity": "medium", "evidence": "no new turn-slots yet: ok"}]
    assert outcome.flags_raised == 1


def test_audit_anomalies_not_refiled_when_the_action_is_retried(tmp_path):
    dispatcher, brief, tools = _setup()
    messages = build_messages(dispatcher, brief)
    cassette = Cassette(dir_path=tmp_path)
    bad = {**_take(dispatcher, anomalies=["x"]), "ability_name": "Not An Ability"}
    cassette.save(_request_body(ROLE_CONFIGS["inner"], messages, tools, 0.2), _fake_response("take_action", bad))
    second = _follow_up(messages, "take_action", bad, "call_1", {"error": "illegal action: unknown ability"})
    cassette.save(_request_body(ROLE_CONFIGS["inner"], second, tools, 0.2),
                  _fake_response("take_action", _take(dispatcher, anomalies=["x"]), "call_2"))

    outcome = decide_and_act(dispatcher, brief, mode="replay", cassette=cassette)
    assert outcome.llm_calls == 2
    assert len(dispatcher.flags) == 1


def test_falls_back_when_cassette_is_empty(tmp_path):
    dispatcher, brief, _ = _setup()
    outcome = decide_and_act(dispatcher, brief, mode="replay", cassette=Cassette(dir_path=tmp_path))
    assert outcome.fell_back_to_first_legal is True
    assert outcome.failed_calls == 1
    assert len(dispatcher.engine.log) == 1  # the fallback still advanced the engine


def test_flag_anomaly_then_take_action(tmp_path):
    dispatcher, brief, tools = _setup()
    messages = build_messages(dispatcher, brief)
    cassette = Cassette(dir_path=tmp_path)
    flag = {"description": "damage looked off", "severity": "low", "evidence": "observed 10 vs 8"}
    cassette.save(_request_body(ROLE_CONFIGS["inner"], messages, tools, 0.2), _fake_response("flag_anomaly", flag))
    second = _follow_up(messages, "flag_anomaly", flag, "call_1", {"recorded": True, "flag_index": 0})
    cassette.save(_request_body(ROLE_CONFIGS["inner"], second, tools, 0.2), _fake_response("take_action", _take(dispatcher), "call_2"))

    outcome = decide_and_act(dispatcher, brief, mode="replay", cassette=cassette)
    assert outcome.llm_calls == 2
    assert outcome.flags == [flag]
    assert outcome.fell_back_to_first_legal is False
    assert len(dispatcher.flags) == 1


def test_ledger_write_offered_and_persisted_with_a_ledger(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    ep = ledger.start_episode("S1", 1, {}, "p")
    dispatcher, brief, tools = _setup(ledger=ledger, episode_id=ep, tmp_path=tmp_path)
    assert [t["function"]["name"] for t in tools] == ["take_action", "flag_anomaly", "ledger_write"]
    messages = build_messages(dispatcher, brief)
    cassette = Cassette(dir_path=tmp_path / "cassettes")
    write = {"status": "open", "hypothesis_id": None, "text": "burn may double under shield", "note": ""}
    cassette.save(_request_body(ROLE_CONFIGS["inner"], messages, tools, 0.2), _fake_response("ledger_write", write))
    second = _follow_up(messages, "ledger_write", write, "call_1", {"created": 1})
    cassette.save(_request_body(ROLE_CONFIGS["inner"], second, tools, 0.2), _fake_response("take_action", _take(dispatcher), "call_2"))

    outcome = decide_and_act(dispatcher, brief, mode="replay", cassette=cassette)
    assert outcome.hypotheses_written[0]["created"] == 1
    assert ledger.hypotheses()[0]["text"] == "burn may double under shield"
    assert ledger.hypotheses()[0]["created_ep"] == ep


def test_prefix_is_stable_across_decisions_in_an_episode():
    dispatcher, brief, _ = _setup()
    first = build_messages(dispatcher, brief)
    dispatcher.take_action(**_take(dispatcher))
    second = build_messages(dispatcher, brief)
    assert first[0]["content"] == second[0]["content"] == SYSTEM_PROMPT
    assert first[1]["content"].startswith(brief)
    assert second[1]["content"].startswith(brief)
    assert first[1]["content"] != second[1]["content"]  # the volatile tail moved


def test_turn_context_shows_start_of_turn_ticks():
    # S1 seed 0: e1 Cinderfang burns p1 Warden; on p1's next turn the
    # pending record carries the Burn tick against the statuses held.
    engine = BattleEngine("S1", seed=0)
    advance_to_actor(engine, "p1")
    engine.take_action(Action(actor_id="p1", ability_name="Basic Attack", target_id="e1"))
    advance_to_actor(engine, "e1")
    engine.take_action(Action(actor_id="e1", ability_name="Cinder Burn", target_id="p1"))
    advance_to_actor(engine, "p1")
    dispatcher = ToolDispatcher(engine)
    ctx = turn_context(dispatcher)
    current = ctx.split("CURRENT TURN: p1 (Warden)\n", 1)[1].split("\n", 1)[0]
    pending = json.loads(current)
    assert pending["ticks"]["burn"] == 6
    assert any(s.startswith("burn:6/") for s in pending["statuses_before_ticks"])
