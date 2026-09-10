"""Single-flag ablation: given a recorded trace, which one of the ten
defects (if any) explains it.

A trace was produced under some DefectFlags configuration (usually
exactly one flag set, by construction of the eval harness). Attribution
replays the same action sequence under each candidate single-flag
configuration in turn (see engine.defects.single_flag) and checks which
ones reproduce the trace with zero divergence -- that is, which flags'
semantics are indistinguishable from what actually happened, given this
particular action sequence and this scenario/seed.

This is also the eval harness's automatic false-positive adjudicator: an
agent's `flag_anomaly` call is auto-classified as a true positive when
the trace up to that point diverges from clean and the flagged entity
matches the divergence; everything else is manual residue (see
eval/adjudicate.py, not yet built).
"""
from __future__ import annotations

from pydantic import BaseModel

from engine.defects import ALL_DEFECT_IDS, DefectFlags, single_flag
from engine.resolution import TurnRecord
from oracle.differential import DifferentialResult, DivergencePoint, replay_and_diff


class AttributionResult(BaseModel):
    triggered: bool  # did the trace diverge from clean engine behavior at all
    clean_divergence: DivergencePoint | None
    matching_defects: list[str]  # defect IDs whose single-flag replay reproduces the trace exactly
    per_defect: dict[str, DifferentialResult]


def attribute(
    scenario_id: str,
    seed: int,
    turn_cap: int,
    original_log: list[TurnRecord],
) -> AttributionResult:
    """Replay `original_log` under clean defects (trigger check) and under
    each single defect flag (attribution). `matching_defects` names every
    flag whose replay reproduces the trace exactly -- normally this is a
    single ID, but it can legitimately be empty (trace matches none of the
    ten -- clean trace, or a multi-flag/compound trace this function
    doesn't attempt to decompose) or contain more than one ID when several
    defects are unreachable along this particular action sequence and so
    are indistinguishable from clean here (see the reachability-ceiling
    guard in the project plan)."""
    clean_result = replay_and_diff(scenario_id, seed, turn_cap, original_log, DefectFlags())

    per_defect: dict[str, DifferentialResult] = {}
    matching: list[str] = []
    for defect_id in ALL_DEFECT_IDS:
        result = replay_and_diff(scenario_id, seed, turn_cap, original_log, single_flag(defect_id))
        per_defect[defect_id] = result
        if not result.diverged:
            matching.append(defect_id)

    return AttributionResult(
        triggered=clean_result.diverged,
        clean_divergence=clean_result.divergence,
        matching_defects=matching,
        per_defect=per_defect,
    )
