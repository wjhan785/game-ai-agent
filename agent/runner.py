"""Episode orchestration for the bare single-turn agent (Week 2): drives
one episode with agent/inner_loop.py's per-turn decisions, then writes it
to episode JSONL for replay/generate.py. Action-budget accounting and
multi-episode sweeps (matched-budget across methods) are eval/harness.py's
job, once the baselines exist -- this module runs exactly one episode.

Deliberately does NOT import engine.defects (see tests/test_no_bug_leakage.py):
`defects` is passed straight through to BattleEngine as an opaque object,
never introspected here. Omit it (or pass None) for clean mode -- that is
BattleEngine's own default; only eval/harness.py (outside agent/) needs to
construct a DefectFlags to grade against.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from agent.inner_loop import decide_and_act
from agent.llm import Cassette, SpendTracker
from agent.tools import ToolDispatcher
from engine.engine import BattleEngine
from engine.episode_log import write_episode_jsonl
from engine.models import BattleState


def run_episode(
    scenario_id: str,
    seed: int,
    defects: Optional[Any] = None,
    turn_cap: int = 20,
    *,
    mode: str = "live",
    cassette: Optional[Cassette] = None,
    spend: Optional[SpendTracker] = None,
    allow_peak: bool = False,
    max_decisions: Optional[int] = None,
) -> tuple[BattleEngine, BattleState, ToolDispatcher, list[dict]]:
    engine = BattleEngine(scenario_id, seed, defects, turn_cap)
    initial_state = engine.state.model_copy(deep=True)
    dispatcher = ToolDispatcher(engine)

    decisions: list[dict] = []
    while not engine.state.finished:
        if max_decisions is not None and len(decisions) >= max_decisions:
            break
        outcome = decide_and_act(
            dispatcher, mode=mode, cassette=cassette, spend=spend, allow_peak=allow_peak
        )
        decisions.append(outcome.model_dump())

    return engine, initial_state, dispatcher, decisions


def reasoning_by_log_index(engine: BattleEngine, decisions: list[dict]) -> dict[int, dict]:
    """Pairs each action-bearing TurnRecord in `engine.log` with the
    `decisions` entry (from run_episode, in the same chronological order)
    that produced it -- decide_and_act is called exactly once per real
    decision point, one-to-one with `record.action is not None` entries,
    so a simple in-order zip is correct without needing to track engine
    log indices while the episode runs."""
    extra: dict[int, dict] = {}
    it = iter(decisions)
    for i, record in enumerate(engine.log):
        if record.action is None:
            continue
        decision = next(it, None)
        if decision is None:
            break
        if decision.get("reasoning"):
            extra[i] = {"agent_reasoning": decision["reasoning"]}
    return extra


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one episode with the bare single-turn inner loop.")
    parser.add_argument("scenario_id")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--turn-cap", type=int, default=20)
    parser.add_argument("--mode", choices=["live", "record", "replay"], default="live")
    parser.add_argument("--max-spend", type=float, default=0.50)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--allow-peak", action="store_true")
    parser.add_argument("--out", default=None, help="path to write episode JSONL to")
    args = parser.parse_args()

    cassette = Cassette() if args.mode in ("record", "replay") else None
    spend = SpendTracker(max_spend=args.max_spend) if args.mode != "replay" else None

    engine, initial_state, dispatcher, decisions = run_episode(
        args.scenario_id,
        args.seed,
        turn_cap=args.turn_cap,
        mode=args.mode,
        cassette=cassette,
        spend=spend,
        allow_peak=args.allow_peak,
        max_decisions=args.max_decisions,
    )

    fallbacks = sum(1 for d in decisions if d["fell_back_to_first_legal"])
    print(
        f"outcome={engine.state.outcome} finished={engine.state.finished} "
        f"steps={engine.state.step_count} decisions={len(decisions)} "
        f"flags={len(dispatcher.flags)} fallbacks={fallbacks}"
    )
    if spend is not None:
        print(f"spend so far (cumulative, results/spend.json): ${spend.total_usd:.4f}")

    out_path = Path(args.out) if args.out else Path(f"logs/{args.scenario_id}_seed{args.seed}.jsonl")
    extra = reasoning_by_log_index(engine, decisions)
    write_episode_jsonl(out_path, engine, initial_state, method="inner_loop_bare", extra_by_index=extra)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
