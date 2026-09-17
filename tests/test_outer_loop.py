"""agent/outer_loop.py, offline. Planning phases replay hand-built cassette
entries keyed on the real request construction; sessions run against an
empty cassette, where every model call falls back -- which exercises the
whole session flow (planning -> episode -> ledger -> logs -> review) for
free and deterministically."""
from __future__ import annotations

import json

import pytest

from agent.ledger import Ledger
from agent.llm import ROLE_CONFIGS, Cassette, SpendTracker, _request_body
from agent.outer_loop import (
    SCENARIO_NOTES,
    SessionConfig,
    build_scenario_catalog,
    planner_system_prompt,
    planning_messages,
    planning_phase,
    run_session,
)
from agent.tools import PLANNER_TOOL_NAMES, build_tool_schemas
from engine.defects import single_flag


def _response(tool_name: str, args: dict, call_id: str):
    return {
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}
        ]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 64, "prompt_cache_miss_tokens": 36},
    }


def _follow(messages, name, args, call_id, result):
    return messages + [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, separators=(",", ":"))},
    ]


PLAN = {"scenario_id": "S2", "seed": 42, "goal": "p1 and p2 both burn e1 in round 0; watch e1's tick",
        "focus_actor_ids": ["p1", "p2"], "hypothesis_ids": []}


def _setup(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    system = planner_system_prompt(build_scenario_catalog(None, ["S1", "S2"]))
    tools = build_tool_schemas(PLANNER_TOOL_NAMES)
    return ledger, system, tools, Cassette(dir_path=tmp_path / "cassettes")


def test_catalog_lists_every_allowed_scenario_with_its_design_focus():
    catalog = build_scenario_catalog(None, list(SCENARIO_NOTES))
    for sid, (name, focus) in SCENARIO_NOTES.items():
        assert f"{sid} {name} -- design focus: {focus}" in catalog
    assert "Overtuned Bolt" in catalog


def test_catalog_reflects_the_build_under_test():
    clean = build_scenario_catalog(None, ["S5"])
    stripped = build_scenario_catalog(single_flag("B08"), ["S5"])
    overtuned = lambda cat: cat.split("e3 Overtuned", 1)[1]
    assert "Basic Attack" in overtuned(clean)
    assert "Basic Attack" not in overtuned(stripped)


def test_plan_on_first_call(tmp_path):
    ledger, system, tools, cassette = _setup(tmp_path)
    messages = planning_messages(ledger, system)
    cassette.save(_request_body(ROLE_CONFIGS["outer"], messages, tools, 0.2), _response("reset_episode", PLAN, "c1"))

    result = planning_phase(ledger, system, allowed_scenarios=["S1", "S2"], mode="replay", cassette=cassette)
    assert result.plan == PLAN
    assert result.fallback is False
    assert result.llm_calls == 1
    assert result.usage.cache_hit_tokens == 64


def test_create_hypothesis_then_plan_against_it(tmp_path):
    ledger, system, tools, cassette = _setup(tmp_path)
    messages = planning_messages(ledger, system)
    write = {"status": "open", "hypothesis_id": None, "text": "two Burns on one target may not sum", "note": ""}
    cassette.save(_request_body(ROLE_CONFIGS["outer"], messages, tools, 0.2), _response("ledger_write", write, "c1"))
    plan = {**PLAN, "hypothesis_ids": [1]}
    second = _follow(messages, "ledger_write", write, "c1", {"created": 1, "calls_left": 9})
    cassette.save(_request_body(ROLE_CONFIGS["outer"], second, tools, 0.2), _response("reset_episode", plan, "c2"))

    result = planning_phase(ledger, system, allowed_scenarios=["S1", "S2"], mode="replay", cassette=cassette)
    assert result.plan["hypothesis_ids"] == [1]
    assert [c["tool"] for c in result.tool_calls] == ["ledger_write", "reset_episode"]
    assert ledger.hypotheses()[0]["text"] == "two Burns on one target may not sum"


def test_invalid_plan_is_rejected_then_falls_back(tmp_path):
    ledger, system, tools, cassette = _setup(tmp_path)
    messages = planning_messages(ledger, system)
    bad = {**PLAN, "scenario_id": "S9"}
    cassette.save(_request_body(ROLE_CONFIGS["outer"], messages, tools, 0.2), _response("reset_episode", bad, "c1"))

    result = planning_phase(ledger, system, allowed_scenarios=["S1", "S2"], mode="replay", cassette=cassette)
    assert result.fallback is True
    assert result.plan["scenario_id"] == "S1"  # least explored, first in order
    assert result.plan["goal"] == ""


def test_last_call_offers_only_the_terminal_tool(tmp_path):
    ledger, system, tools, cassette = _setup(tmp_path)
    messages = planning_messages(ledger, system, max_calls=2)
    read = {"kind": "hypotheses"}
    cassette.save(_request_body(ROLE_CONFIGS["outer"], messages, tools, 0.2), _response("ledger_read", read, "c1"))
    second = _follow(messages, "ledger_read", read, "c1", {"hypotheses": [], "calls_left": 1})
    only_reset = build_tool_schemas(["reset_episode"])
    cassette.save(_request_body(ROLE_CONFIGS["outer"], second, only_reset, 0.2), _response("reset_episode", PLAN, "c2"))

    result = planning_phase(ledger, system, allowed_scenarios=["S1", "S2"], max_calls=2, mode="replay", cassette=cassette)
    assert result.plan == PLAN and result.fallback is False


def test_reachable_status_pairs_follow_declared_targeting():
    from agent.outer_loop import scenario_reachability

    reach = scenario_reachability(None, list(SCENARIO_NOTES))
    assert ("burn", "shield") in reach["S1"]  # enemy Burn onto a party member the Warden shields
    assert ("burn", "shield") not in reach["S2"]  # S2's Burn only ever lands on enemies
    assert all(("poison", "shield") not in pairs for pairs in reach.values())
    assert all(("burn", "regen") not in pairs for pairs in reach.values())
    assert ("poison", "weaken") in reach["S4"]


def test_digest_coverage_split_into_reachable_and_unreachable(tmp_path):
    from agent.outer_loop import annotate_coverage, scenario_reachability

    digest = annotate_coverage(Ledger(tmp_path / "l.db").digest(), scenario_reachability(None, ["S1", "S4"]))
    cov = digest["coverage"]
    assert "status_pairs_never_seen_together" not in cov
    assert cov["status_pairs_never_seen_but_reachable"]["burn+shield"] == ["S1"]
    assert "poison+shield" in cov["status_pairs_unreachable_in_these_scenarios"]


@pytest.mark.parametrize("field,value", [("seed", -1), ("focus_actor_ids", ["p9"]), ("hypothesis_ids", [7]), ("goal", "  ")])
def test_plan_validation(tmp_path, field, value):
    from agent.outer_loop import _validate_plan

    ledger = Ledger(tmp_path / "ledger.db")
    assert _validate_plan({**PLAN, field: value}, ledger, ["S2"]) is not None
    assert _validate_plan(PLAN, ledger, ["S2"]) is None


def test_offline_session_end_to_end(tmp_path):
    summary = run_session(
        SessionConfig(run_name="t", episodes=2, turn_cap=3, allowed_scenarios=["S1", "S2"]),
        runs_root=tmp_path, mode="replay", cassette=Cassette(dir_path=tmp_path / "empty"), progress=lambda _: None,
    )
    run_dir = tmp_path / "t"
    assert summary["episodes_completed"] == 2
    assert summary["planner_fallbacks"] == 2
    assert summary["aborted_reason"] is None
    assert (run_dir / "session.json").exists()
    assert (run_dir / "episodes" / "ep_0001.jsonl").exists() and (run_dir / "episodes" / "ep_0002.jsonl").exists()

    ledger = Ledger(run_dir / "ledger.db")
    episodes = ledger.episodes()
    assert [e["scenario"] for e in episodes] == ["S1", "S2"]  # fallback spreads across least-explored
    assert episodes[0]["summary"]["planner_fallback"] is True
    assert episodes[0]["actions_used"] == summary["actions"] - episodes[1]["actions_used"]


def test_session_refuses_to_reuse_a_run_directory(tmp_path):
    config = SessionConfig(run_name="t", episodes=1, turn_cap=2, allowed_scenarios=["S1"], final_review=False)
    kwargs = dict(runs_root=tmp_path, mode="replay", cassette=Cassette(dir_path=tmp_path / "empty"), progress=lambda _: None)
    run_session(config, **kwargs)
    with pytest.raises(FileExistsError):
        run_session(config, **kwargs)


def test_session_aborts_cleanly_on_spend_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    spend = SpendTracker(path=tmp_path / "spend.json", max_session_spend=0.0)
    summary = run_session(
        SessionConfig(run_name="t", episodes=3, allowed_scenarios=["S1"]),
        runs_root=tmp_path, mode="live", spend=spend, allow_peak=True, progress=lambda _: None,
    )
    assert summary["aborted_reason"] and "cap" in summary["aborted_reason"]
    assert summary["episodes_completed"] == 0
    assert json.loads((tmp_path / "t" / "session.json").read_text())["aborted_reason"] == summary["aborted_reason"]
    assert spend.total_usd == 0.0
