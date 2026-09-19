"""Each invariant-visible defect trips the checker; clean mode never does."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

from engine.defects import DefectFlags, INVARIANT_VISIBLE, single_flag
from engine.engine import BattleEngine

SCENARIOS_AND_SEEDS = [(sid, seed) for sid in ("S1", "S2", "S3", "S4", "S5", "S6") for seed in range(8)]


def _run_random(scenario_id: str, seed: int, defects: DefectFlags) -> list[str]:
    eng = BattleEngine(scenario_id, seed=seed, defects=defects, turn_cap=30)
    rng = random.Random(seed * 7919 + 3)
    steps = 0
    while not eng.state.finished and steps < 500:
        legal = eng.legal_actions()
        if not legal:
            break
        eng.take_action(rng.choice(legal))
        steps += 1
    return [v for r in eng.log for v in r.invariant_violations]


def test_clean_mode_never_trips_invariants_across_scenarios():
    for sid, seed in SCENARIOS_AND_SEEDS:
        violations = _run_random(sid, seed, DefectFlags())
        assert violations == [], f"{sid}/{seed}: {violations}"


def test_each_invariant_visible_defect_trips_somewhere():
    """Not a per-scenario claim (B08 in particular only fires in S5's
    Overtuned setup) -- just that across the scenario x seed grid, each
    invariant-visible defect trips at least once when enabled alone."""
    for defect_id in sorted(INVARIANT_VISIBLE):
        defects = single_flag(defect_id)
        tripped = False
        for sid, seed in SCENARIOS_AND_SEEDS:
            if _run_random(sid, seed, defects):
                tripped = True
                break
        assert tripped, f"{defect_id} never tripped any invariant across the grid"
