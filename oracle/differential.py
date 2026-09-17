"""The differential oracle: replay a recorded action sequence against an
engine running a different DefectFlags configuration, and report the
first point where the trajectories disagree.

Why step index is a safe alignment key: turn order is now speed-driven
(BattleState.current_actor_id picks the character with the lowest
action_value, tie-broken by turn_order), not a fixed cycle -- so the
argument for positional alignment can no longer rest on the schedule
being the same list every time. It rests on this instead: action_value is
part of `CharacterSnapshot`, and `_diff_records` below compares full
`after` snapshots -- so "the two runs agree at every step up to i" is a
claim about a superset of everything the scheduler reads (every
character's hp/energy/alive/statuses/cooldowns/action_value). Identical
snapshots at step i-1 therefore imply an identical `current_actor_id()`
selection at step i, by construction, not by assumption -- and
`_diff_records` checks `actor_id` FIRST, precisely because a scheduling
difference is real evidence on its own, before any state difference
explains it. `BattleEngine._advance_to_decision` still advances
`elapsed_av`/`round_number`/`step_count` for EVERY turn-slot, whether or
not a decision was needed, so that sequence is identical across any two
DefectFlags configurations up to and including the first step where the
runs actually disagree. That is what makes comparing `TurnRecord` logs
positionally (index i in one log vs index i in the other) valid up to
the first divergence -- which is exactly the point this module exists to
find. Nothing here needs to re-derive alignment past that point; once a
divergence is found, replay stops being trustworthy and the caller has
its answer.

One rule this depends on: no defect flag may change a character's speed.
`scenarios.build_battle_state` reads defects only to decide the abilities
on a character's kit (B08) and the tie-break order itself (B11) -- never
speed -- so "same scenario + seed => same schedule" stays true regardless
of which defects are on, and a schedule difference is always attributable
to the run, never to an accident of construction.

`replay()` drives a fresh BattleEngine by feeding it the ORIGINAL run's
recorded actions in order, one per decision point the candidate engine
raises. If the candidate needs a decision the original didn't record (or
vice versa), the fed action will either be for the wrong actor -- caught
by rules.check_legal as "not this actor's turn" -- or the lists will
simply run out; either way `diverges()` still finds the true divergence
at the correct earlier step, because up to that point both logs are
byte-identical (same defects would have produced no divergence at all).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

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
    """Drive a fresh BattleEngine under `defects`, feeding `actions` in
    order to each decision point it raises. Stops early (without raising)
    if a fed action turns out illegal under `defects`' semantics, or once
    `actions` is exhausted."""
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
        # In practice this never fires as the FIRST divergence: any real
        # difference in a character's scheduling already shows up one step
        # earlier as an actor_id mismatch (_diff_records checks that
        # first). It is here so the module docstring's claim -- that the
        # diffed snapshot is a superset of everything the scheduler reads
        # -- is actually true, not just asserted.
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
    """First point of disagreement between two TurnRecord logs, compared
    positionally. Returns None if the shorter log's entries all agree
    with the longer log's corresponding prefix AND the logs are the same
    length; a length mismatch alone is reported as a "log_length"
    divergence at the position where the shorter log ran out."""
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
    """Replay `original_log`'s action sequence under `candidate_defects`
    and diff the resulting trajectory against the original. A divergence
    proves the original trace exercised behavior that `candidate_defects`
    does not reproduce."""
    actions = extract_actions(original_log)
    result = replay(scenario_id, seed, turn_cap, candidate_defects, actions)
    divergence = diverges(original_log, result.log)
    return DifferentialResult(diverged=divergence is not None, divergence=divergence, replay=result)
