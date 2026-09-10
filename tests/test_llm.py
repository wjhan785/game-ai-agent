"""Tests for agent/llm.py that don't touch the network: cost accounting,
the peak-hour gate, the spend guard, cassette round-trip, tool-schema
generation from a Pydantic model, and the structured-output validate
path (exercised directly against a hand-built cassette, in "replay"
mode -- see tests/test_llm_smoke.py for the one test that makes a real
API call).
"""
from __future__ import annotations

import datetime
import json

from pydantic import BaseModel

from agent.llm import (
    ROLE_CONFIGS,
    Cassette,
    PeakHourBlocked,
    SpendTracker,
    UsageStats,
    _request_body,
    call_with_tools,
    check_not_peak,
    compute_cost,
    is_peak_hour_utc,
    tool_schema,
)


def test_compute_cost_matches_published_rate_card():
    # 2000 cached prefix + 500 volatile miss + 80 output, per the project
    # plan's own per-decision example -- exact value:
    # 2000/1e6*0.007 + 500/1e6*0.22 + 80/1e6*0.66 = 0.0001768.
    cost = compute_cost(cache_hit_tokens=2000, cache_miss_tokens=500, completion_tokens=80)
    assert abs(cost - 0.0001768) < 1e-9


def test_compute_cost_cache_miss_is_much_more_expensive_than_cache_hit():
    hit_cost = compute_cost(cache_hit_tokens=1_000_000, cache_miss_tokens=0, completion_tokens=0)
    miss_cost = compute_cost(cache_hit_tokens=0, cache_miss_tokens=1_000_000, completion_tokens=0)
    assert miss_cost / hit_cost > 30  # ~31x per the project plan


def test_compute_cost_zero_usage_is_zero():
    assert compute_cost(0, 0, 0) == 0.0


def test_is_peak_hour_utc_weekday_inside_window():
    # Tuesday 2026-09-08, 02:00 UTC -- inside the 01:00-04:00 window.
    dt = datetime.datetime(2026, 9, 8, 2, 0, tzinfo=datetime.timezone.utc)
    assert is_peak_hour_utc(dt) is True


def test_is_peak_hour_utc_weekday_outside_window():
    dt = datetime.datetime(2026, 9, 8, 12, 0, tzinfo=datetime.timezone.utc)
    assert is_peak_hour_utc(dt) is False


def test_is_peak_hour_utc_weekend_always_off_peak():
    # Saturday 2026-09-12, 02:00 UTC -- would be peak on a weekday.
    dt = datetime.datetime(2026, 9, 12, 2, 0, tzinfo=datetime.timezone.utc)
    assert is_peak_hour_utc(dt) is False


def test_check_not_peak_blocks_without_allow_peak():
    dt = datetime.datetime(2026, 9, 8, 7, 0, tzinfo=datetime.timezone.utc)
    try:
        check_not_peak(allow_peak=False, now=dt)
        assert False, "expected PeakHourBlocked"
    except PeakHourBlocked:
        pass
    check_not_peak(allow_peak=True, now=dt)  # must not raise


def test_spend_tracker_persists_and_enforces_cap(tmp_path):
    path = tmp_path / "spend.json"
    tracker = SpendTracker(path=path, max_spend=0.01)
    tracker.record(0.004)
    tracker.check_budget()  # under cap, must not raise

    reloaded = SpendTracker(path=path, max_spend=0.01)
    assert abs(reloaded.total_usd - 0.004) < 1e-9

    reloaded.record(0.007)
    try:
        reloaded.check_budget()
        assert False, "expected BudgetExceededError"
    except Exception as exc:
        assert "cap" in str(exc)


def test_cassette_round_trip(tmp_path):
    cassette = Cassette(dir_path=tmp_path)
    request = {"model": "deepseek-flash", "messages": [{"role": "user", "content": "hi"}]}
    assert cassette.load(request) is None

    response = {"choices": [{"message": {"tool_calls": []}}], "usage": {"prompt_tokens": 5}}
    cassette.save(request, response)
    loaded = cassette.load(request)
    assert loaded["response"] == response

    # Never touches headers/Authorization -- the request body alone is
    # what's hashed and stored, so an API key can never land in a cassette.
    on_disk = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "authorization" not in json.dumps(on_disk).lower()
    assert "api_key" not in json.dumps(on_disk).lower()


def test_tool_schema_from_pydantic_model():
    class Ping(BaseModel):
        message: str

    schema = tool_schema("ping", "Say hi.", Ping)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "ping"
    assert "message" in schema["function"]["parameters"]["properties"]


class _Report(BaseModel):
    status: str
    note: str


def _fake_cassette_response(tool_name: str, args: dict, *, prompt_tokens=10, completion_tokens=5, cache_hit=3):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_cache_hit_tokens": cache_hit,
            "prompt_cache_miss_tokens": prompt_tokens - cache_hit,
        },
    }


def test_call_with_tools_replay_mode_success(tmp_path):
    cassette = Cassette(dir_path=tmp_path)
    messages = [{"role": "user", "content": "call report"}]
    tools = [tool_schema("report", "Report status.", _Report)]
    # Built via the real _request_body, not hand-copied, so this test
    # can't silently drift from what call_with_tools actually hashes.
    request = _request_body(ROLE_CONFIGS["inner"], messages, tools, temperature=0.2)
    cassette.save(request, _fake_cassette_response("report", {"status": "ok", "note": "hi"}))

    result = call_with_tools(
        role="inner",
        messages=messages,
        tools=tools,
        tool_models={"report": _Report},
        mode="replay",
        cassette=cassette,
    )
    assert result.ok is True
    assert result.tool_name == "report"
    assert result.args == {"status": "ok", "note": "hi"}
    assert result.usage.cache_hit_tokens == 3


def test_call_with_tools_replay_mode_missing_cassette_entry_is_reported_not_raised(tmp_path):
    cassette = Cassette(dir_path=tmp_path)
    messages = [{"role": "user", "content": "no matching cassette entry"}]
    tools = [tool_schema("report", "Report status.", _Report)]

    result = call_with_tools(
        role="inner",
        messages=messages,
        tools=tools,
        tool_models={"report": _Report},
        mode="replay",
        cassette=cassette,
    )
    assert result.ok is False
    assert result.malformed is True
    assert "cassette" in result.failure_reason
