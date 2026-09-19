"""Single-flag ablation: which one defect (if any) explains a trace made
under a single-defect build. Replays it under each single flag."""

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
    """Replay under clean and under each single flag. `matching_defects` lists
    every flag that reproduces the trace exactly: usually one, possibly none
    or several (defects this trace never reached look identical)."""
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
