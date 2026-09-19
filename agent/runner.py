"""Runs one episode with LLM-call gating and ledger bookkeeping.

Gating: the model decides only on new state signatures or focus
characters; other turns take a seeded random legal action. `defects` is
passed to BattleEngine unopened.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from agent.inner_loop import DEFAULT_WINDOW, decide_and_act, episode_brief
from agent.ledger import Ledger, action_signature, invariant_kind, record_coverage_combos, state_signature
from agent.llm import BudgetExceededError, Cassette, PeakHourBlocked, SpendTracker, UsageStats, add_usage
from agent.tools import ToolDispatcher
from engine.engine import BattleEngine
from engine.episode_log import write_episode_jsonl
from engine.models import BattleState
from engine.resolution import TurnRecord


class EpisodeConfig(BaseModel):
    scenario_id: str
    seed: int
    turn_cap: int = 20
    goal: str = ""
    focus_actor_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[int] = Field(default_factory=list)
    gating: bool = True
    max_decisions: int = 150
    window: int = DEFAULT_WINDOW
    memory: bool = True  # False: the brief shows nothing read back from the ledger


class Novelty(Protocol):
    def seen(self, state_sig: str) -> bool: ...
    def record_visit(self, state_sig: str, action_sig: str, episode: int) -> None: ...


class MemoryNovelty:
    """In-process visit set, for runs without a ledger. Persisting it across
    episodes is the caller's choice (reuse the same instance)."""

    def __init__(self) -> None:
        self._states: set[str] = set()

    def seen(self, state_sig: str) -> bool:
        return state_sig in self._states

    def record_visit(self, state_sig: str, action_sig: str, episode: int) -> None:
        self._states.add(state_sig)


@dataclass
class EpisodeResult:
    engine: BattleEngine
    initial_state: BattleState
    dispatcher: ToolDispatcher
    decisions: list[dict]
    usage: UsageStats
    new_coverage: list[str] = field(default_factory=list)
    aborted_reason: Optional[str] = None

    def count(self, mode: str) -> int:
        return sum(1 for d in self.decisions if d["mode"] == mode)

    @property
    def llm_calls(self) -> int:
        return sum(d.get("llm_calls", 0) for d in self.decisions)


class Bookkeeper:
    """Records coverage and invariant flags to the ledger. Shared with baselines/."""

    def __init__(self, engine: BattleEngine, ledger: Optional[Ledger], episode_id: Optional[int]):
        self.engine = engine
        self.ledger = ledger
        self.episode_id = episode_id
        self.processed = 0
        self.action_base = ledger.total_actions() if ledger is not None else 0
        self.invariant_kinds: set[str] = set()
        self.new_coverage: list[str] = []

    def absorb(self, actions_taken: int) -> None:
        records = self.engine.log[self.processed:]
        self.processed = len(self.engine.log)
        if self.ledger is None or self.episode_id is None:
            return
        for rec in records:
            self._record(rec, self.action_base + actions_taken)

    def _record(self, rec: TurnRecord, action_idx: int) -> None:
        element = None
        if rec.action is not None:
            ability = self.engine.state.characters[rec.actor_id].ability_by_name(rec.action.ability_name)
            element = ability.element.value if ability is not None else None
        for combo in sorted(record_coverage_combos(rec, element)):
            if self.ledger.record_coverage(combo, self.episode_id, action_idx):
                self.new_coverage.append(combo)
        for violation in rec.invariant_violations:
            kind = invariant_kind(violation)
            if kind in self.invariant_kinds:
                continue
            self.invariant_kinds.add(kind)
            self.ledger.add_flag(
                episode=self.episode_id, turn=rec.step_count, source="invariant", description=violation
            )


def run_episode(
    config: EpisodeConfig,
    *,
    defects: Optional[Any] = None,
    ledger: Optional[Ledger] = None,
    episode_id: Optional[int] = None,
    run_dir: Optional[str | Path] = None,
    novelty: Optional[Novelty] = None,
    mode: str = "live",
    cassette: Optional[Cassette] = None,
    spend: Optional[SpendTracker] = None,
    allow_peak: bool = False,
) -> EpisodeResult:
    engine = BattleEngine(config.scenario_id, config.seed, defects, config.turn_cap)
    initial_state = engine.state.model_copy(deep=True)
    flags_dir = Path(run_dir) / "flags" if run_dir is not None else None
    dispatcher = ToolDispatcher(engine, ledger=ledger, episode_id=episode_id, flags_dir=flags_dir)

    hypotheses: list[dict] = []
    confirmed: list[dict] = []
    if ledger is not None and config.memory:
        wanted = set(config.hypothesis_ids)
        hypotheses = [h for h in ledger.hypotheses() if h["id"] in wanted]
        confirmed = [h for h in ledger.hypotheses(("confirmed",)) if h["id"] not in wanted]
    brief = episode_brief(
        engine, goal=config.goal, hypotheses=hypotheses, focus_actor_ids=config.focus_actor_ids, confirmed=confirmed
    )

    if novelty is None:
        novelty = ledger if ledger is not None else MemoryNovelty()
    focus = set(config.focus_actor_ids)
    rng = random.Random(f"scripted:{config.scenario_id}:{config.seed}:{episode_id}")
    bookkeeper = Bookkeeper(engine, ledger, episode_id)
    bookkeeper.absorb(0)

    decisions: list[dict] = []
    usage = UsageStats()
    aborted: Optional[str] = None
    # Slots from this step on are new to the model.
    audit_from = 0

    while not engine.state.finished and len(decisions) < config.max_decisions:
        actor_id = engine.state.current_actor_id()
        step = engine.state.step_count
        sig = state_signature(engine.state)
        informative = not config.gating or actor_id in focus or not novelty.seen(sig)
        before = len(engine.log)
        try:
            if informative:
                outcome = decide_and_act(
                    dispatcher, brief, window=config.window, audit_from=audit_from, mode=mode,
                    cassette=cassette, spend=spend, allow_peak=allow_peak,
                )
                usage = add_usage(usage, outcome.usage)
                decision = outcome.model_dump(exclude={"take_action_result", "usage"})
                decision["mode"] = "fallback" if outcome.fell_back_to_first_legal else "llm"
                audit_from = step
            else:
                choice = rng.choice(engine.legal_actions())
                dispatcher.take_action(
                    actor_id=actor_id, ability_name=choice.ability_name,
                    target_id=choice.target_id, reasoning="scripted",
                )
                decision = {"actor_id": actor_id, "mode": "scripted", "reasoning": None, "flags": [], "llm_calls": 0}
        except (BudgetExceededError, PeakHourBlocked) as exc:
            aborted = str(exc)
            break

        if len(engine.log) > before:
            taken = engine.log[before].action
            if taken is not None:
                novelty.record_visit(sig, action_signature(taken.ability_name, taken.target_id), episode_id or 0)
        decisions.append(decision)
        bookkeeper.absorb(len(decisions))

    return EpisodeResult(
        engine=engine,
        initial_state=initial_state,
        dispatcher=dispatcher,
        decisions=decisions,
        usage=usage,
        new_coverage=bookkeeper.new_coverage,
        aborted_reason=aborted,
    )


def decision_annotations(engine: BattleEngine, decisions: list[dict]) -> dict[int, dict]:
    """Match each action in the log to its decision (one per action, in order)."""
    extra: dict[int, dict] = {}
    it = iter(decisions)
    for i, record in enumerate(engine.log):
        if record.action is None:
            continue
        decision = next(it, None)
        if decision is None:
            break
        note: dict = {}
        if decision.get("mode"):
            note["decision_mode"] = decision["mode"]
        if decision.get("reasoning"):
            note["agent_reasoning"] = decision["reasoning"]
        if decision.get("audit"):
            note["agent_audit"] = decision["audit"]
        if decision.get("flags"):
            note["agent_flags"] = decision["flags"]
        if note:
            extra[i] = note
    return extra


def write_episode(result: EpisodeResult, path: str | Path, method: str) -> Path:
    path = Path(path)
    write_episode_jsonl(
        path, result.engine, result.initial_state, method=method,
        extra_by_index=decision_annotations(result.engine, result.decisions),
    )
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one episode with the inner loop (no planner, no ledger).")
    parser.add_argument("scenario_id")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--turn-cap", type=int, default=20)
    parser.add_argument("--goal", default="")
    parser.add_argument("--no-gating", action="store_true")
    parser.add_argument("--mode", choices=["live", "record", "replay"], default="live")
    parser.add_argument("--max-spend", type=float, default=0.50, help="cap on THIS run's spend, USD")
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--allow-peak", action="store_true")
    parser.add_argument("--out", default=None, help="path to write episode JSONL to")
    args = parser.parse_args()

    cassette = Cassette() if args.mode in ("record", "replay") else None
    spend = SpendTracker(max_session_spend=args.max_spend) if args.mode != "replay" else None
    config = EpisodeConfig(
        scenario_id=args.scenario_id, seed=args.seed, turn_cap=args.turn_cap, goal=args.goal,
        gating=not args.no_gating, max_decisions=args.max_decisions,
    )
    result = run_episode(config, mode=args.mode, cassette=cassette, spend=spend, allow_peak=args.allow_peak)

    engine = result.engine
    print(
        f"outcome={engine.state.outcome} finished={engine.state.finished} steps={engine.state.step_count} "
        f"decisions={len(result.decisions)} llm={result.count('llm')} scripted={result.count('scripted')} "
        f"fallbacks={result.count('fallback')} flags={len(result.dispatcher.flags)}"
    )
    if result.aborted_reason:
        print(f"ABORTED: {result.aborted_reason}")
    if spend is not None:
        print(f"this run: ${spend.session_usd:.4f}   cumulative (results/spend.json): ${spend.total_usd:.4f}")

    out_path = Path(args.out) if args.out else Path(f"logs/{args.scenario_id}_seed{args.seed}.jsonl")
    write_episode(result, out_path, method="inner_loop")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
