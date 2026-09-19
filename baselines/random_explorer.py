"""Random baseline: uniform-random legal moves, no LLM. Writes the same
ledger and logs as the agent, so the bug report grades it unchanged."""

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from pydantic import BaseModel, Field

from agent.ledger import Ledger, action_signature, invariant_kind, state_signature
from agent.runner import Bookkeeper
from enemy_ai.env import SCENARIO_SLOTS, random_action
from engine.engine import BattleEngine
from engine.episode_log import write_episode_jsonl

METHOD = "random"


class RandomConfig(BaseModel):
    run_name: str
    actions: int = 800  # total budget across the campaign
    turn_cap: int = 20
    scenarios: list[str] = Field(default_factory=lambda: list(SCENARIO_SLOTS))
    max_decisions_per_episode: int = 150
    seed: int = 0


def run_random_episode(
    engine: BattleEngine, rng: np.random.Generator, max_decisions: int, ledger: Ledger, episode_id: int
) -> tuple[int, list[str]]:
    """Plays one battle at random. Returns (actions taken, new coverage)."""
    bookkeeper = Bookkeeper(engine, ledger, episode_id)
    bookkeeper.absorb(0)
    actions = 0
    while not engine.state.finished and actions < max_decisions:
        legal = engine.legal_actions()
        if not legal:
            break
        sig = state_signature(engine.state)
        action = random_action(engine.state, rng, legal)
        engine.take_action(action)
        ledger.record_visit(sig, action_signature(action.ability_name, action.target_id), episode_id)
        actions += 1
        bookkeeper.absorb(actions)
    return actions, bookkeeper.new_coverage


def run_random_campaign(
    config: RandomConfig,
    *,
    runs_root: str | Path = "logs/runs",
    defects: Optional[Any] = None,
    progress: Callable[[str], None] = print,
) -> dict:
    run_dir = Path(runs_root) / config.run_name
    ledger_path = run_dir / "ledger.db"
    if ledger_path.exists():
        raise FileExistsError(f"{ledger_path} already exists -- pick a new run name rather than mixing runs")
    ledger = Ledger(ledger_path)
    seeds = np.random.default_rng(config.seed)

    while ledger.total_actions() < config.actions:
        number = len(ledger.episodes()) + 1
        scenario_id = config.scenarios[(number - 1) % len(config.scenarios)]
        battle_seed = int(seeds.integers(1_000_000))
        log_path = run_dir / "episodes" / f"ep_{number:04d}.jsonl"
        ep_id = ledger.start_episode(scenario_id, battle_seed, {"method": METHOD}, str(log_path))

        engine = BattleEngine(scenario_id, battle_seed, defects, config.turn_cap)
        initial_state = engine.state.model_copy(deep=True)
        budget = min(config.max_decisions_per_episode, config.actions - ledger.total_actions())
        rng = np.random.default_rng([config.seed, number])
        actions, new_coverage = run_random_episode(engine, rng, budget, ledger, ep_id)
        write_episode_jsonl(log_path, engine, initial_state, method=METHOD)

        kinds = sorted({invariant_kind(f["description"]) for f in ledger.flags(source="invariant", episode=ep_id)})
        summary = {
            "outcome": engine.state.outcome,
            "rounds": engine.state.round_number,
            "actions": actions,
            "invariant_violation_kinds": kinds,
            "new_coverage": new_coverage,
        }
        ledger.finish_episode(
            ep_id, outcome=engine.state.outcome, actions_used=actions, llm_calls=0,
            tokens_in=0, tokens_out=0, cache_hit_tokens=0, cost_usd=0.0, summary=summary,
        )
        progress(
            f"[ep {ep_id:>3}] {scenario_id} seed {battle_seed:<7} {str(engine.state.outcome):<10} "
            f"actions={actions:<3} invariant_kinds={len(kinds)} new_cov={len(new_coverage)}"
        )
        if actions == 0:  # no progress possible; don't loop forever
            break

    summary = {
        "run_name": config.run_name,
        "method": METHOD,
        "config": config.model_dump(),
        "episodes_completed": len(ledger.episodes()),
        "actions": ledger.total_actions(),
        "decisions": {METHOD: ledger.total_actions()},
        "planner_calls": 0,
        "inner_calls": 0,
        "failed_calls": 0,
        "planner_fallbacks": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cache_hit_tokens": 0,
        "cache_hit_rate": None,
        "cost_usd": 0.0,
        "agent_flags": 0,
        "invariant_flags": len(ledger.flags(source="invariant")),
        "coverage_items": len(ledger.coverage()),
        "visits": ledger.visit_stats(),
        "aborted_reason": None,
    }
    (run_dir / "session.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ledger.close()
    return summary
