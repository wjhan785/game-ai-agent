"""Differential oracle: replay a recorded action sequence under another
build and report the first step where the two runs disagree.

Logs are compared by step index. That is valid up to the first divergence
because each snapshot includes action_value, a superset of what the
scheduler reads: equal snapshots mean the same next actor. This relies on
no defect flag ever changing a character's speed.
"""

from typing import Optional

from pydantic import BaseModel

from engine import rules
from engine.defects import DefectFlags
from engine.engine import BattleEngine, IllegalActionError
from engine.models import Action
from engine.resolution import CharacterSnapshot, TurnRecord

FLOAT_TOL = 1e-6


class DivergencePoint(BaseModel):
    step: int
    field: str  # "actor_id" | "skipped_reason" | "state" | "log_length"
    original: str
    candidate: str
    detail: Optional[str] = None


class ReplayResult(BaseModel):
    log: list[TurnRecord]
    consumed: int
    total_actions: int
    finished: bool
    outcome: Optional[str] = None
    stopped_reason: Optional[str] = None  # None if it ran cleanly to completion or ran out of actions


class DifferentialResult(BaseModel):
    diverged: bool
    divergence: Optional[DivergencePoint] = None
    replay: ReplayResult


def extract_actions(log: list[TurnRecord]) -> list[Action]:
    """The ordered sequence of actions actually submitted in a recorded
    episode -- skip-only turn-slots (`record.action is None`) carry no
    decision and are not part of the replay-driving sequence."""
    return [r.action for r in log if r.action is not None]


def replay(
    scenario_id: str,
    seed: int,
    turn_cap: int,
    defects: DefectFlags,
    actions: list[Action],
) -> ReplayResult:
    """Feed `actions` to a fresh engine under `defects`. Stops quietly on an
    illegal action or when the actions run out."""
    engine = BattleEngine(scenario_id, seed, defects, turn_cap)
    consumed = 0
    stopped_reason: Optional[str] = None

    for action in actions:
        if engine.state.finished:
            stopped_reason = "battle_finished_before_actions_exhausted"
            break
        if not engine.legal_actions():
            stopped_reason = "no_decision_pending"
            break
        check = rules.check_legal(engine.state, action)
        if not check.legal:
            stopped_reason = f"illegal action at consumption {consumed}: {check.reason}"
            break
        try:
            engine.take_action(action)
        except IllegalActionError as exc:
            stopped_reason = f"illegal action at consumption {consumed}: {exc.reason}"
            break
        consumed += 1

    return ReplayResult(
        log=engine.log,
        consumed=consumed,
        total_actions=len(actions),
        finished=engine.state.finished,
        outcome=engine.state.outcome,
        stopped_reason=stopped_reason,
    )


def _diff_snapshot(
    char_id: str, o: CharacterSnapshot, c: CharacterSnapshot
) -> Optional[str]:
    parts: list[str] = []
    if abs(o.hp - c.hp) > FLOAT_TOL:
        parts.append(f"hp {o.hp} != {c.hp}")
    if abs(o.energy - c.energy) > FLOAT_TOL:
        parts.append(f"energy {o.energy} != {c.energy}")
    if o.alive != c.alive:
        parts.append(f"alive {o.alive} != {c.alive}")
    if o.cooldowns != c.cooldowns:
        parts.append(f"cooldowns {o.cooldowns} != {c.cooldowns}")
    if abs(o.action_value - c.action_value) > FLOAT_TOL:
        # Rarely first: a scheduling difference usually shows as an actor_id mismatch.
        parts.append(f"action_value {o.action_value} != {c.action_value}")

    o_statuses = sorted(
        (s.status_type.value, round(s.magnitude, 6), s.turns_remaining, s.source_id)
        for s in o.statuses
    )
    c_statuses = sorted(
        (s.status_type.value, round(s.magnitude, 6), s.turns_remaining, s.source_id)
        for s in c.statuses
    )
    if o_statuses != c_statuses:
        parts.append(f"statuses {o_statuses} != {c_statuses}")

    if not parts:
        return None
    return f"{char_id}: " + "; ".join(parts)


def _diff_records(step: int, o: TurnRecord, c: TurnRecord) -> Optional[DivergencePoint]:
    if o.actor_id != c.actor_id:
        return DivergencePoint(step=step, field="actor_id", original=o.actor_id, candidate=c.actor_id)

    o_reason = o.skipped_reason or ""
    c_reason = c.skipped_reason or ""
    if o_reason != c_reason:
        return DivergencePoint(
            step=step, field="skipped_reason", original=o_reason or "(none)", candidate=c_reason or "(none)"
        )

    detail_parts: list[str] = []
    for char_id in o.after:
        if char_id not in c.after:
            continue
        d = _diff_snapshot(char_id, o.after[char_id], c.after[char_id])
        if d:
            detail_parts.append(d)
    if detail_parts:
        return DivergencePoint(
            step=step,
            field="state",
            original="after",
            candidate="after",
            detail=" | ".join(detail_parts),
        )
    return None


def diverges(original_log: list[TurnRecord], candidate_log: list[TurnRecord]) -> Optional[DivergencePoint]:
    """First disagreement between two logs by position, or None. A length
    difference alone is a "log_length" divergence."""
    n = min(len(original_log), len(candidate_log))
    for i in range(n):
        d = _diff_records(i, original_log[i], candidate_log[i])
        if d is not None:
            return d
    if len(original_log) != len(candidate_log):
        return DivergencePoint(
            step=n,
            field="log_length",
            original=str(len(original_log)),
            candidate=str(len(candidate_log)),
        )
    return None


def replay_and_diff(
    scenario_id: str,
    seed: int,
    turn_cap: int,
    original_log: list[TurnRecord],
    candidate_defects: DefectFlags,
) -> DifferentialResult:
    """Replay the original actions under `candidate_defects` and diff the logs."""
    actions = extract_actions(original_log)
    result = replay(scenario_id, seed, turn_cap, candidate_defects, actions)
    divergence = diverges(original_log, result.log)
    return DifferentialResult(diverged=divergence is not None, divergence=divergence, replay=result)
