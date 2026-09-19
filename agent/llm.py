"""DeepSeek via the OpenAI SDK: forced tool calls with validate/retry,
cache-aware cost accounting, spend caps, an off-peak gate, and cassette
record/replay.

DeepSeek caches shared prefixes (hits are far cheaper), so callers put
stable content first and volatile content last.
"""

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# --- Config: model per role ----------------------------------------------


class RoleConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"


ROLE_CONFIGS: dict[str, RoleConfig] = {
    "inner": RoleConfig(),
    "outer": RoleConfig(),
}


def get_client(role_config: RoleConfig) -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set."
        )
    base_url = os.environ.get("DEEPSEEK_BASE_URL", role_config.base_url)
    return OpenAI(api_key=api_key, base_url=base_url)


# --- Pricing, off-peak gate ------------------------------------------------

# USD per million tokens. Peak prices apply only with --allow-peak.
PRICE_PER_MTOK = {
    "off_peak": {"cache_hit_in": 0.003, "cache_miss_in": 0.15, "out": 0.6},
}

# Peak hours, UTC weekdays (09:00-12:00 and 14:00-18:00 SGT).
PEAK_WINDOWS_UTC = [(1, 4), (6, 10)]


def is_peak_hour_utc(now: Optional[datetime.datetime] = None) -> bool:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6 -- weekends are off-peak
        return False
    hour = now.hour
    return any(start <= hour < end for start, end in PEAK_WINDOWS_UTC)


class PeakHourBlocked(Exception):
    pass


def check_not_peak(allow_peak: bool, now: Optional[datetime.datetime] = None) -> None:
    if not allow_peak and is_peak_hour_utc(now):
        raise PeakHourBlocked(
            "refusing to call the API during a DeepSeek peak-pricing window "
            "(01:00-04:00 or 06:00-10:00 UTC, weekdays) -- pass allow_peak=True "
            "/ --allow-peak if this is deliberate."
        )


def compute_cost(
    cache_hit_tokens: int, cache_miss_tokens: int, completion_tokens: int
) -> float:
    price = PRICE_PER_MTOK["off_peak"]
    return (
        (cache_hit_tokens / 1_000_000) * price["cache_hit_in"]
        + (cache_miss_tokens / 1_000_000) * price["cache_miss_in"]
        + (completion_tokens / 1_000_000) * price["out"]
    )


# --- Spend guard ------------------------------------------------------------


class BudgetExceededError(Exception):
    pass


HARD_CEILING_USD = 15.0


class SpendTracker:
    """Cumulative spend, persisted to results/spend.json. Caps: `hard_ceiling`
    (project total), `max_spend` (all-time), `max_session_spend` (this
    tracker only; what --max-spend sets)."""

    def __init__(
        self,
        path: str | Path = "results/spend.json",
        max_spend: Optional[float] = None,
        max_session_spend: Optional[float] = None,
        hard_ceiling: float = HARD_CEILING_USD,
    ):
        self.path = Path(path)
        self.max_spend = max_spend
        self.max_session_spend = max_session_spend
        self.hard_ceiling = hard_ceiling
        self.total_usd = self._load()
        self.session_start_usd = self.total_usd

    @property
    def session_usd(self) -> float:
        return self.total_usd - self.session_start_usd

    def _load(self) -> float:
        if not self.path.exists():
            return 0.0
        try:
            return float(json.loads(self.path.read_text(encoding="utf-8")).get("total_usd", 0.0))
        except (json.JSONDecodeError, ValueError) as exc:
            # Silently restarting from $0 would disarm every cap below.
            raise RuntimeError(f"{self.path} is unreadable ({exc}); fix or remove it deliberately") from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"total_usd": round(self.total_usd, 6)}, indent=2), encoding="utf-8")

    def check_budget(self) -> None:
        if self.total_usd >= self.hard_ceiling:
            raise BudgetExceededError(
                f"total spend ${self.total_usd:.4f} has reached the ${self.hard_ceiling:.2f} project ceiling"
            )
        if self.max_spend is not None and self.total_usd >= self.max_spend:
            raise BudgetExceededError(
                f"spend ${self.total_usd:.4f} has reached the ${self.max_spend:.4f} cap"
            )
        if self.max_session_spend is not None and self.session_usd >= self.max_session_spend:
            raise BudgetExceededError(
                f"this run's spend ${self.session_usd:.4f} has reached its ${self.max_session_spend:.4f} cap"
            )

    def record(self, cost_usd: float) -> None:
        self.total_usd += cost_usd
        self._save()


# --- Cassette record/replay -------------------------------------------------


class Cassette:
    """Request/response recordings keyed by a hash of the request body.
    Stores bodies only, never headers, so the API key can't end up here."""

    def __init__(self, dir_path: str | Path = "fixtures"):
        self.dir_path = Path(dir_path)

    def _key(self, request: dict) -> str:
        blob = json.dumps(request, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:24]

    def _path(self, request: dict) -> Path:
        return self.dir_path / f"{self._key(request)}.json"

    def load(self, request: dict) -> Optional[dict]:
        path = self._path(request)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def save(self, request: dict, response: dict) -> None:
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self._path(request).write_text(
            json.dumps({"request": request, "response": response}, indent=2), encoding="utf-8"
        )


# --- Structured tool-call output, with validate/retry -----------------------


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cost_usd: float = 0.0


def add_usage(a: UsageStats, b: UsageStats) -> UsageStats:
    return UsageStats(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        cache_hit_tokens=a.cache_hit_tokens + b.cache_hit_tokens,
        cache_miss_tokens=a.cache_miss_tokens + b.cache_miss_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
    )


class CallResult(BaseModel):
    ok: bool
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    args: Optional[dict] = None  # validated args, as a plain dict (model_dump)
    raw_arguments: Optional[str] = None
    retries_used: int = 0
    malformed: bool = False
    failure_reason: Optional[str] = None
    usage: UsageStats = UsageStats()

    def parsed_as(self, model_cls: Type[T]) -> T:
        if not self.ok or self.args is None:
            raise ValueError(f"call did not succeed: {self.failure_reason}")
        return model_cls.model_validate(self.args)


def _request_body(
    role_config: RoleConfig,
    messages: list[dict],
    tools: list[dict],
    temperature: float,
) -> dict:
    return {
        "model": role_config.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "temperature": temperature,
        # Thinking mode rejects tool_choice="required", so turn it off.
        "thinking": {"type": "disabled"},
    }


def _extract_usage(usage_obj: Any) -> UsageStats:
    if usage_obj is None:
        return UsageStats()
    prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    cache_hit = getattr(details, "cached_tokens", None) if details is not None else None
    if cache_hit is None:
        cache_hit = getattr(usage_obj, "prompt_cache_hit_tokens", 0) or 0
    cache_miss = getattr(usage_obj, "prompt_cache_miss_tokens", None)
    if cache_miss is None:
        cache_miss = max(0, prompt_tokens - cache_hit)
    cost = compute_cost(cache_hit, cache_miss, completion_tokens)
    return UsageStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_hit_tokens=cache_hit,
        cache_miss_tokens=cache_miss,
        cost_usd=cost,
    )


def call_with_tools(
    role: str,
    messages: list[dict],
    tools: list[dict],
    tool_models: dict[str, Type[BaseModel]],
    *,
    mode: str = "live",  # "live" | "record" | "replay"
    cassette: Optional[Cassette] = None,
    spend: Optional[SpendTracker] = None,
    max_retries: int = 2,
    temperature: float = 0.2,
    allow_peak: bool = False,
) -> CallResult:
    """One forced tool call, with up to `max_retries` repair attempts.
    Retries append the bad call and the error to `messages` in place.
    On failure returns ok=False, malformed=True; the caller picks a fallback."""
    role_config = ROLE_CONFIGS[role]
    if mode == "replay" and cassette is None:
        raise ValueError("mode='replay' requires a cassette")
    if mode != "replay":
        check_not_peak(allow_peak)

    client = get_client(role_config) if mode != "replay" else None
    retries_used = 0
    total_usage = UsageStats()

    # Same loop in every mode, so replay follows the recorded retries.
    while True:
        request = _request_body(role_config, messages, tools, temperature)
        if mode == "replay":
            cached = cassette.load(request)
            if cached is None:
                return CallResult(
                    ok=False,
                    malformed=True,
                    failure_reason="no cassette entry for this request",
                    retries_used=retries_used,
                    usage=total_usage,
                )
            response_dict = cached["response"]
        else:
            if spend is not None:
                spend.check_budget()
            response = client.chat.completions.create(
                model=role_config.model,
                messages=messages,
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=False,
                temperature=temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
            response_dict = response.model_dump(mode="json")
            if mode == "record" and cassette is not None:
                cassette.save(request, response_dict)

        usage = _extract_usage(_DictUsage(response_dict.get("usage")))
        total_usage = add_usage(total_usage, usage)
        if spend is not None and mode != "replay":
            spend.record(usage.cost_usd)

        message = response_dict["choices"][0]["message"]
        result = _parse_tool_call(message, tool_models, total_usage, retries_used)
        if result.ok or retries_used >= max_retries:
            return result

        # Repair turn: echo the assistant's (malformed) call, then a tool
        # message carrying the validation error, and try again.
        messages.append(_strip_none(message))
        tool_calls = message.get("tool_calls") or []
        tool_call_id = tool_calls[0].get("id") if tool_calls else None
        if tool_call_id is not None:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Invalid call: {result.failure_reason}. Please retry with corrected arguments.",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"Invalid response: {result.failure_reason}. You must call exactly one of the provided tools.",
                }
            )
        retries_used += 1


def _strip_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


class _DictUsage:
    """Adapts a plain dict (from a cassette) to the attribute access
    `_extract_usage` expects from an SDK usage object."""

    def __init__(self, d: Optional[dict]):
        self._d = d or {}

    def __getattr__(self, name: str) -> Any:
        value = self._d.get(name)
        if isinstance(value, dict):
            return _DictUsage(value)
        return value


def _parse_tool_call(
    message: dict,
    tool_models: dict[str, Type[BaseModel]],
    usage: UsageStats,
    retries_used: int,
) -> CallResult:
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return CallResult(
            ok=False,
            malformed=True,
            failure_reason="model did not call a tool",
            retries_used=retries_used,
            usage=usage,
        )
    call = tool_calls[0]
    call_id = call.get("id")
    name = call["function"]["name"]
    raw_args = call["function"]["arguments"]

    model_cls = tool_models.get(name)
    if model_cls is None:
        return CallResult(
            ok=False,
            tool_name=name,
            tool_call_id=call_id,
            raw_arguments=raw_args,
            malformed=True,
            failure_reason=f"unknown tool '{name}', expected one of {sorted(tool_models)}",
            retries_used=retries_used,
            usage=usage,
        )
    try:
        parsed = model_cls.model_validate_json(raw_args)
    except ValidationError as exc:
        return CallResult(
            ok=False,
            tool_name=name,
            tool_call_id=call_id,
            raw_arguments=raw_args,
            malformed=True,
            failure_reason=str(exc),
            retries_used=retries_used,
            usage=usage,
        )
    return CallResult(
        ok=True,
        tool_name=name,
        tool_call_id=call_id,
        args=parsed.model_dump(mode="json"),
        raw_arguments=raw_args,
        retries_used=retries_used,
        usage=usage,
    )


def tool_schema(name: str, description: str, model_cls: Type[BaseModel]) -> dict:
    """Builds an OpenAI-format tool schema from a Pydantic model's own
    JSON schema, so a tool's argument shape has exactly one source of
    truth (the model), matching how DefectFlags -> ANSWER_KEY.md works."""
    schema = model_cls.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


# --- Smoke test --------------------------------------------------------


class SmokeReport(BaseModel):
    status: str
    note: str


def run_smoke_test(allow_peak: bool = False) -> CallResult:
    """One live tool call: checks the key, model, parsing and cost accounting."""
    messages = [
        {
            "role": "system",
            "content": "You are a connectivity smoke test. Call the `report` tool exactly once.",
        },
        {
            "role": "user",
            "content": "Call `report` with status='ok' and a short note confirming you received this message.",
        },
    ]
    tools = [tool_schema("report", "Report smoke-test status.", SmokeReport)]
    return call_with_tools(
        role="inner",
        messages=messages,
        tools=tools,
        tool_models={"report": SmokeReport},
        mode="live",
        allow_peak=allow_peak,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="agent.llm CLI")
    parser.add_argument("--smoke", action="store_true", help="run the connectivity smoke test")
    parser.add_argument("--allow-peak", action="store_true", help="allow calling during a peak-pricing window")
    args = parser.parse_args()

    if args.smoke:
        result = run_smoke_test(allow_peak=args.allow_peak)
        print(f"ok: {result.ok}")
        print(f"tool_name: {result.tool_name}")
        print(f"args: {result.args}")
        print(f"malformed: {result.malformed}  failure_reason: {result.failure_reason}")
        print(
            f"usage: prompt={result.usage.prompt_tokens} "
            f"(cache_hit={result.usage.cache_hit_tokens}, cache_miss={result.usage.cache_miss_tokens}) "
            f"completion={result.usage.completion_tokens}"
        )
        print(f"cost_usd: {result.usage.cost_usd:.6f}")
        if not result.ok:
            raise SystemExit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
