"""The planner (outer loop) and the campaign driver.

Before each episode the planner reads the ledger digest, updates
hypotheses, and calls reset_episode with a scenario, seed, goal and focus
characters. A final review closes open hypotheses. If planning fails, the
least-explored scenario is used with no goal (reported, not fatal).
"""

import itertools
import json
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from agent.inner_loop import DEFAULT_WINDOW
from agent.ledger import HYPOTHESIS_STATUSES, Ledger, invariant_kind
from agent.llm import (
    BudgetExceededError,
    Cassette,
    PeakHourBlocked,
    SpendTracker,
    UsageStats,
    add_usage,
    call_with_tools,
)
from agent.prompts import RULES_SPEC, dumps, format_roster
from agent.runner import EpisodeConfig, EpisodeResult, run_episode, write_episode
from agent.tools import (
    PLANNER_TOOL_NAMES,
    REVIEW_TOOL_NAMES,
    TOOL_MODELS,
    build_tool_schemas,
    ledger_read,
    ledger_write,
)
from engine.engine import BattleEngine
from engine.models import BattleState, TargetType

# Names and design focus per scenario, from docs/scenario-matrix.md (the
# design rationale committed before any seeded defect existed).
SCENARIO_NOTES: dict[str, tuple[str, str]] = {
    "S1": ("Bulwark", "shield x damage-over-time"),
    "S2": ("Pyroclasm", "multi-source stacking"),
    "S3": ("Deep Freeze", "energy denial"),
    "S4": ("Attrition", "percentage damage x modifiers"),
    "S5": ("Lockdown", "turn-skip x cooldown timing"),
    "S6": ("Last Stand", "death x persistent healing"),
}
CHARACTER_IDS = ("p1", "p2", "p3", "e1", "e2", "e3")
MAX_SEED = 1_000_000

PLANNER_INSTRUCTIONS = """YOUR JOB AT EACH EPISODE BOUNDARY
You lead an automated QA campaign against this combat system. You do not know which defects exist, or whether any do. A tactical agent plays each episode you plan, controlling both teams, and reports anomalies; the invariant checker independently reports impossible states.
1. Review. For each hypothesis the last episode tested, call ledger_write with its new status and the evidence in the note: confirmed = the suspected wrong behavior was observed, with numbers; refuted = the interaction was exercised and behaved per the rules; inconclusive = the episode never set it up. Leave it "testing" only if you will test it again next episode.
2. Plan. Choose 1-3 hypotheses for the next episode -- open ones by id, or new ones created with ledger_write (status "open"). Aim at what is untested: status pairs never seen together, statuses x elements, statuses x death and revive, resources x debuffs, control x cooldowns, and anything suspicious the tactical agent or invariant checker reported that is not yet confirmed. A good hypothesis names one interaction and the specific wrong outcome it might produce.
3. Call reset_episode: the scenario able to exercise them, a seed, a goal the tactical agent can follow turn by turn (which characters use which abilities on whom, in what order, and what to check), the focus characters (the tactical agent decides all of their turns; other turns may be scripted), and the hypothesis ids under test.
Use ledger_read only for detail the digest leaves out."""


def reachable_status_pairs(state: BattleState) -> set[tuple[str, str]]:
    """Status pairs one character could ever hold together, from each
    ability's declared targeting."""
    landable: dict[str, set[str]] = {cid: set() for cid in state.characters}
    for user in state.characters.values():
        for ability in user.abilities:
            if ability.applies is None:
                continue
            status = ability.applies.status_type.value
            for c in state.characters.values():
                if ability.target_type == TargetType.SELF:
                    ok = c.id == user.id
                elif ability.target_type == TargetType.ONE_ALLY:
                    ok = c.team == user.team
                else:
                    ok = c.team != user.team
                if ok:
                    landable[c.id].add(status)
    pairs: set[tuple[str, str]] = set()
    for statuses in landable.values():
        pairs.update(itertools.combinations(sorted(statuses), 2))
    return pairs


def scenario_reachability(defects: Optional[Any], scenario_ids: list[str]) -> dict[str, set[tuple[str, str]]]:
    return {sid: reachable_status_pairs(BattleEngine(sid, 0, defects).state) for sid in scenario_ids}


def build_scenario_catalog(defects: Optional[Any], scenario_ids: list[str]) -> str:
    """Roster and declared ability data for each scenario, as built by the
    engine under test (seed 0; seeds vary HP/energy by up to 10-15%), with
    the status pairs each scenario can put on one character."""
    reachable = scenario_reachability(defects, scenario_ids)
    blocks = []
    for sid in scenario_ids:
        name, focus = SCENARIO_NOTES.get(sid, (sid, ""))
        state = BattleEngine(sid, 0, defects).state
        pairs = ", ".join("+".join(p) for p in sorted(reachable[sid])) or "none"
        blocks.append(
            f"{sid} {name} -- design focus: {focus}\n{format_roster(state)}\n"
            f"Status pairs that can be held by one character here: {pairs}"
        )
    return "\n\n".join(blocks)


def annotate_coverage(digest: dict, reachable: dict[str, set[tuple[str, str]]]) -> dict:
    """Splits the digest's never-seen status pairs into those some allowed
    scenario can produce (and which) and those none can."""
    coverage = digest["coverage"]
    never = coverage.pop("status_pairs_never_seen_together")
    reach_by_pair: dict[str, list[str]] = {}
    for sid, pairs in reachable.items():
        for p in pairs:
            reach_by_pair.setdefault("+".join(p), []).append(sid)
    coverage["status_pairs_never_seen_but_reachable"] = {p: sorted(reach_by_pair[p]) for p in never if p in reach_by_pair}
    coverage["status_pairs_unreachable_in_these_scenarios"] = [p for p in never if p not in reach_by_pair]
    return digest


def planner_system_prompt(catalog: str) -> str:
    return f"{RULES_SPEC}\n\nSCENARIOS (the build under test)\n{catalog}\n\n{PLANNER_INSTRUCTIONS}"


class PlanResult(BaseModel):
    plan: Optional[dict] = None
    end_summary: Optional[str] = None
    fallback: bool = False
    llm_calls: int = 0
    failed_calls: int = 0
    usage: UsageStats = Field(default_factory=UsageStats)
    tool_calls: list[dict] = Field(default_factory=list)


def _validate_plan(args: dict, ledger: Ledger, allowed: list[str]) -> Optional[str]:
    if args["scenario_id"] not in allowed:
        return f"scenario_id must be one of {allowed}"
    if not 0 <= args["seed"] <= MAX_SEED:
        return f"seed must be between 0 and {MAX_SEED}"
    bad = [a for a in args["focus_actor_ids"] if a not in CHARACTER_IDS]
    if bad:
        return f"unknown focus_actor_ids {bad}; characters are {list(CHARACTER_IDS)}"
    missing = [h for h in args["hypothesis_ids"] if not ledger.hypothesis_exists(h)]
    if missing:
        return f"no hypotheses with ids {missing}; create them with ledger_write first"
    if not args["goal"].strip():
        return "goal must not be empty"
    return None


def _fallback_plan(ledger: Ledger, allowed: list[str]) -> dict:
    counts = {sid: 0 for sid in allowed}
    for e in ledger.episodes():
        if e["scenario"] in counts:
            counts[e["scenario"]] += 1
    scenario = min(allowed, key=lambda sid: (counts[sid], allowed.index(sid)))
    return {
        "scenario_id": scenario,
        "seed": len(ledger.episodes()) + 1,
        "goal": "",
        "focus_actor_ids": [],
        "hypothesis_ids": [],
    }


MAX_PLANNER_CALLS = 10


def planning_messages(
    ledger: Ledger,
    system_prompt: str,
    review_only: bool = False,
    reachable: Optional[dict[str, set[tuple[str, str]]]] = None,
    max_calls: int = MAX_PLANNER_CALLS,
) -> list[dict]:
    digest = ledger.digest()
    if reachable is not None:
        digest = annotate_coverage(digest, reachable)
    if review_only:
        ask = "Final review: update the status of every hypothesis still open or testing, with evidence, then call end_session."
        terminal = "end_session"
    else:
        ask = f"Review the last episode (if any), then plan episode {len(ledger.episodes()) + 1}."
        terminal = "reset_episode"
    budget = (
        f"You have {max_calls} tool calls for this. The digest already holds the ledger's hypotheses, "
        f"recent flags and coverage, so read only for what it leaves out. Your last call must be {terminal}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"LEDGER DIGEST\n{dumps(digest)}\n\n{ask} {budget}"},
    ]


def planning_phase(
    ledger: Ledger,
    system_prompt: str,
    *,
    allowed_scenarios: list[str],
    review_only: bool = False,
    reachable: Optional[dict[str, set[tuple[str, str]]]] = None,
    max_calls: int = MAX_PLANNER_CALLS,
    mode: str = "live",
    cassette: Optional[Cassette] = None,
    spend: Optional[SpendTracker] = None,
    allow_peak: bool = False,
) -> PlanResult:
    episodes = ledger.episodes()
    last_ep = episodes[-1]["id"] if episodes else None
    messages = planning_messages(ledger, system_prompt, review_only, reachable, max_calls)
    tool_names = REVIEW_TOOL_NAMES if review_only else PLANNER_TOOL_NAMES
    terminal = "end_session" if review_only else "reset_episode"

    out = PlanResult()
    for call_index in range(max_calls):
        # The last call offers only the terminal tool.
        names = [terminal] if call_index == max_calls - 1 else tool_names
        result = call_with_tools(
            role="outer", messages=messages, tools=build_tool_schemas(names),
            tool_models={n: TOOL_MODELS[n] for n in names},
            mode=mode, cassette=cassette, spend=spend, allow_peak=allow_peak,
        )
        out.llm_calls += 1
        out.usage = add_usage(out.usage, result.usage)
        if not result.ok or result.tool_name is None or result.args is None:
            out.failed_calls += 1
            break
        name, args = result.tool_name, result.args
        out.tool_calls.append({"tool": name, "args": args})

        if name == "ledger_read":
            response = ledger_read(ledger, args["kind"])
        elif name == "ledger_write":
            response = ledger_write(
                ledger, last_ep, status=args["status"], hypothesis_id=args.get("hypothesis_id"),
                text=args.get("text"), note=args.get("note", ""),
            )
        elif name == "reset_episode":
            error = _validate_plan(args, ledger, allowed_scenarios)
            if error is None:
                out.plan = args
                return out
            response = {"error": error}
        elif name == "end_session":
            out.end_summary = args["summary"]
            return out
        else:
            response = {"error": f"unknown tool {name}"}
        response["calls_left"] = max_calls - call_index - 1

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": result.tool_call_id, "type": "function", "function": {"name": name, "arguments": result.raw_arguments}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": result.tool_call_id, "content": json.dumps(response, separators=(",", ":"))})

    out.fallback = True
    if not review_only:
        out.plan = _fallback_plan(ledger, allowed_scenarios)
    return out


class SessionConfig(BaseModel):
    run_name: str
    episodes: int = 25
    turn_cap: int = 20
    gating: bool = True
    allowed_scenarios: list[str] = Field(default_factory=lambda: list(SCENARIO_NOTES))
    max_decisions_per_episode: int = 150
    window: int = DEFAULT_WINDOW
    max_planner_calls: int = MAX_PLANNER_CALLS
    final_review: bool = True
    method: str = "full_agent"
    planner: bool = True  # False = Greedy-LLM: no planner, no ledger readback


def _episode_summary(result: EpisodeResult, ledger: Ledger, episode_id: int, plan: PlanResult) -> dict:
    agent_flags = ledger.flags(source="agent", episode=episode_id)
    invariant = ledger.flags(source="invariant", episode=episode_id)
    return {
        "outcome": result.engine.state.outcome,
        "rounds": result.engine.state.round_number,
        "actions": len(result.decisions),
        "llm_decisions": result.count("llm"),
        "scripted_decisions": result.count("scripted"),
        "fallback_decisions": result.count("fallback"),
        "agent_flags": [
            {"id": f["id"], "step": f["turn"], "severity": f["severity"],
             "description": f["description"], "evidence": (f["evidence"] or "")[:300]}
            for f in agent_flags
        ],
        "invariant_violation_kinds": sorted({invariant_kind(f["description"]) for f in invariant}),
        "hypotheses_opened_by_tactical_agent": [
            h["created"] for d in result.decisions for h in d.get("hypotheses_written", []) if "created" in h
        ],
        "new_coverage": result.new_coverage,
        "planner_fallback": plan.fallback,
        "aborted": result.aborted_reason,
    }


def run_session(
    config: SessionConfig,
    *,
    runs_root: str | Path = "logs/runs",
    defects: Optional[Any] = None,
    mode: str = "live",
    cassette: Optional[Cassette] = None,
    spend: Optional[SpendTracker] = None,
    allow_peak: bool = False,
    progress: Callable[[str], None] = print,
    resume: bool = False,
) -> dict:
    """Runs a campaign. `resume` continues an existing run up to `config.episodes`."""
    run_dir = Path(runs_root) / config.run_name
    ledger_path = run_dir / "ledger.db"
    if ledger_path.exists() and not resume:
        raise FileExistsError(f"{ledger_path} already exists -- pick a new run name rather than mixing runs")
    if resume and not ledger_path.exists():
        raise FileNotFoundError(f"nothing to resume at {ledger_path}")
    session_path = run_dir / "session.json"
    prev = json.loads(session_path.read_text(encoding="utf-8")) if resume and session_path.exists() else {}
    ledger = Ledger(ledger_path)
    system_prompt = planner_system_prompt(build_scenario_catalog(defects, config.allowed_scenarios))
    reachable = scenario_reachability(defects, config.allowed_scenarios)

    # Totals carry over from the previous segment when resuming.
    totals = {k: prev.get(k, 0) for k in ("planner_calls", "inner_calls", "failed_calls", "planner_fallbacks")}
    usage = UsageStats(
        prompt_tokens=prev.get("tokens_in", 0), completion_tokens=prev.get("tokens_out", 0),
        cache_hit_tokens=prev.get("cache_hit_tokens", 0),
        cache_miss_tokens=prev.get("tokens_in", 0) - prev.get("cache_hit_tokens", 0),
        cost_usd=prev.get("cost_usd", 0.0),
    )
    decisions = {m: prev.get("decisions", {}).get(m, 0) for m in ("llm", "scripted", "fallback")}
    aborted: Optional[str] = None
    final_summary: Optional[str] = None
    common = dict(mode=mode, cassette=cassette, spend=spend, allow_peak=allow_peak)

    try:
        for _ in range(config.episodes - len(ledger.episodes())):
            if config.planner:
                plan = planning_phase(
                    ledger, system_prompt, allowed_scenarios=config.allowed_scenarios, reachable=reachable,
                    max_calls=config.max_planner_calls, **common,
                )
            else:
                plan = PlanResult(plan=_fallback_plan(ledger, config.allowed_scenarios))
            totals["planner_calls"] += plan.llm_calls
            totals["failed_calls"] += plan.failed_calls
            totals["planner_fallbacks"] += int(plan.fallback)
            usage = add_usage(usage, plan.usage)
            p = plan.plan
            assert p is not None

            episode_number = len(ledger.episodes()) + 1
            log_path = run_dir / "episodes" / f"ep_{episode_number:04d}.jsonl"
            ep_id = ledger.start_episode(p["scenario_id"], p["seed"], {**p, "planner_tool_calls": plan.tool_calls}, str(log_path))
            for hid in p["hypothesis_ids"]:
                ledger.update_hypothesis(hid, "testing", "", ep_id)

            result = run_episode(
                EpisodeConfig(
                    scenario_id=p["scenario_id"], seed=p["seed"], turn_cap=config.turn_cap, goal=p["goal"],
                    focus_actor_ids=p["focus_actor_ids"], hypothesis_ids=p["hypothesis_ids"],
                    gating=config.gating, max_decisions=config.max_decisions_per_episode, window=config.window,
                    memory=config.planner,
                ),
                defects=defects, ledger=ledger, episode_id=ep_id, run_dir=run_dir, **common,
            )
            write_episode(result, log_path, method=config.method)

            ep_usage = add_usage(plan.usage, result.usage)
            usage = add_usage(usage, result.usage)
            totals["inner_calls"] += result.llm_calls
            totals["failed_calls"] += sum(d.get("failed_calls", 0) for d in result.decisions)
            for m in decisions:
                decisions[m] += result.count(m)
            summary = _episode_summary(result, ledger, ep_id, plan)
            ledger.finish_episode(
                ep_id, outcome=result.engine.state.outcome, actions_used=len(result.decisions),
                llm_calls=plan.llm_calls + result.llm_calls, tokens_in=ep_usage.prompt_tokens,
                tokens_out=ep_usage.completion_tokens, cache_hit_tokens=ep_usage.cache_hit_tokens,
                cost_usd=ep_usage.cost_usd, summary=summary,
            )
            progress(
                f"[ep {ep_id:>3}] {p['scenario_id']} seed {p['seed']:<7} {str(result.engine.state.outcome):<10} "
                f"actions={len(result.decisions):<3} llm={result.count('llm'):<3} "
                f"agent_flags={len(summary['agent_flags'])} invariant_kinds={len(summary['invariant_violation_kinds'])} "
                f"new_cov={len(result.new_coverage)} ${ep_usage.cost_usd:.4f}"
                + (" [planner fallback]" if plan.fallback else "")
            )
            if result.aborted_reason:
                aborted = result.aborted_reason
                break

        if config.final_review and config.planner and aborted is None:
            review = planning_phase(
                ledger, system_prompt, allowed_scenarios=config.allowed_scenarios, reachable=reachable,
                review_only=True, max_calls=config.max_planner_calls, **common,
            )
            totals["planner_calls"] += review.llm_calls
            totals["failed_calls"] += review.failed_calls
            usage = add_usage(usage, review.usage)
            final_summary = review.end_summary
    except (BudgetExceededError, PeakHourBlocked) as exc:
        aborted = str(exc)

    hypotheses = ledger.hypotheses()
    summary = {
        "run_name": config.run_name,
        "method": config.method,
        "config": config.model_dump(),
        "episodes_completed": len(ledger.episodes()),
        "actions": ledger.total_actions(),
        "decisions": decisions,
        **totals,
        "tokens_in": usage.prompt_tokens,
        "tokens_out": usage.completion_tokens,
        "cache_hit_tokens": usage.cache_hit_tokens,
        "cache_hit_rate": round(usage.cache_hit_tokens / usage.prompt_tokens, 4) if usage.prompt_tokens else None,
        "cost_usd": round(usage.cost_usd, 6),
        "agent_flags": len(ledger.flags(source="agent")),
        "invariant_flags": len(ledger.flags(source="invariant")),
        "hypotheses": {s: sum(1 for h in hypotheses if h["status"] == s) for s in HYPOTHESIS_STATUSES},
        "coverage_items": len(ledger.coverage()),
        "visits": ledger.visit_stats(),
        "final_summary": final_summary,
        "aborted_reason": aborted,
    }
    (run_dir / "session.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ledger.close()
    return summary
