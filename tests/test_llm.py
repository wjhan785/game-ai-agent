"""agent/llm.py offline: costs, peak gate, spend caps, cassettes, schemas
and the validate/retry path."""

import datetime
import json

from pydantic import BaseModel

from agent.llm import (
    PRICE_PER_MTOK,
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
    # 2000 cached + 500 miss + 80 output, priced from the rate card.
    price = PRICE_PER_MTOK["off_peak"]
    expected = 2000 / 1e6 * price["cache_hit_in"] + 500 / 1e6 * price["cache_miss_in"] + 80 / 1e6 * price["out"]
    cost = compute_cost(cache_hit_tokens=2000, cache_miss_tokens=500, completion_tokens=80)
    assert abs(cost - expected) < 1e-12


def test_compute_cost_cache_miss_is_much_more_expensive_than_cache_hit():
    hit_cost = compute_cost(cache_hit_tokens=1_000_000, cache_miss_tokens=0, completion_tokens=0)
    miss_cost = compute_cost(cache_hit_tokens=0, cache_miss_tokens=1_000_000, completion_tokens=0)
    assert miss_cost / hit_cost > 30


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


def test_replay_walks_the_recorded_repair_path(tmp_path):
    # First response is invalid, the repair succeeds; replay must follow both.
    cassette = Cassette(dir_path=tmp_path)
    messages = [{"role": "user", "content": "call report"}]
    tools = [tool_schema("report", "Report status.", _Report)]

    first = _fake_cassette_response("report", {"status": "ok"})
    cassette.save(_request_body(ROLE_CONFIGS["inner"], list(messages), tools, 0.2), first)

    bad_message = first["choices"][0]["message"]
    repaired_messages = list(messages) + [
        {k: v for k, v in bad_message.items() if v is not None},
        {"role": "tool", "tool_call_id": "call_1", "content": "PLACEHOLDER"},
    ]
    # The repair message embeds the exact validation error text, so derive
    # it the same way call_with_tools does.
    from agent.llm import _parse_tool_call

    failure = _parse_tool_call(bad_message, {"report": _Report}, UsageStats(), 0).failure_reason
    repaired_messages[-1]["content"] = f"Invalid call: {failure}. Please retry with corrected arguments."
    cassette.save(
        _request_body(ROLE_CONFIGS["inner"], repaired_messages, tools, 0.2),
        _fake_cassette_response("report", {"status": "ok", "note": "fixed"}),
    )

    result = call_with_tools(
        role="inner", messages=messages, tools=tools, tool_models={"report": _Report},
        mode="replay", cassette=cassette,
    )
    assert result.ok is True
    assert result.retries_used == 1
    assert result.args == {"status": "ok", "note": "fixed"}
    assert result.usage.prompt_tokens == 20  # both recorded calls counted


def test_spend_tracker_session_cap_ignores_earlier_spend(tmp_path):
    path = tmp_path / "spend.json"
    SpendTracker(path=path).record(5.0)  # earlier phases
    tracker = SpendTracker(path=path, max_session_spend=0.75)
    tracker.check_budget()  # $5 already spent, but this run has spent $0
    tracker.record(0.80)
    try:
        tracker.check_budget()
        assert False, "expected BudgetExceededError"
    except Exception as exc:
        assert "this run" in str(exc)


def test_spend_tracker_hard_ceiling_always_enforced(tmp_path):
    path = tmp_path / "spend.json"
    SpendTracker(path=path).record(15.0)
    try:
        SpendTracker(path=path).check_budget()
        assert False, "expected BudgetExceededError"
    except Exception as exc:
        assert "ceiling" in str(exc)


def test_spend_tracker_refuses_unreadable_file(tmp_path):
    path = tmp_path / "spend.json"
    path.write_text("{not json", encoding="utf-8")
    try:
        SpendTracker(path=path)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "unreadable" in str(exc)


def test_extract_usage_reads_nested_cached_tokens():
    from agent.llm import _DictUsage, _extract_usage

    usage = _extract_usage(_DictUsage({"prompt_tokens": 100, "completion_tokens": 5,
                                       "prompt_tokens_details": {"cached_tokens": 64}}))
    assert usage.cache_hit_tokens == 64
    assert usage.cache_miss_tokens == 36
