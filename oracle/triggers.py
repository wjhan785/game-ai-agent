"""Per-defect trigger detection for traces recorded against a build with
several defects enabled at once.

oracle/attribute.py answers "which single flag reproduces this trace",
which only works when one flag produced it. A campaign runs against the
whole buggy build, so here each defect is tested by leave-one-out: replay
the recorded actions under the same build minus that one flag. Because
every other defect is still present, the two trajectories agree until the
first moment that defect's branch changes something -- so a divergence
means the defect changed the outcome of THIS trace, and its position is
the first time it did.

This is the "trigger" half of the trigger-vs-detection split: identical
procedure for every method, decided by the oracle alone.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from engine import resolution, rules
from engine.defects import ALL_DEFECT_IDS, DefectFlags
from engine.models import Action, BattleState
from engine.resolution import TurnRecord
from engine.scenarios import build_battle_state
from oracle.differential import FLOAT_TOL, DifferentialResult, _diff_snapshot, replay_and_diff

_CHAR_ID = re.compile(r"\b([pe][1-3])\b")


class TriggerResult(BaseModel):
    defect_id: str
    triggered: bool
    step: Optional[int] = None
    field: Optional[str] = None
    detail: Optional[str] = None
    entities: list[str] = []


class TriggerReport(BaseModel):
    reproduces: bool  # sanity: the trace replays exactly under its own build
    self_check: DifferentialResult
    triggers: dict[str, TriggerResult]


def _entities(original_log: list[TurnRecord], result: DifferentialResult) -> list[str]:
    d = result.divergence
    if d is None:
        return []
    if d.field == "state" and d.detail:
        return sorted(set(_CHAR_ID.findall(d.detail)))
    if d.step < len(original_log):
        return [original_log[d.step].actor_id]
    return []


def leave_one_out_triggers(
    scenario_id: str,
    seed: int,
    turn_cap: int,
    original_log: list[TurnRecord],
    build: DefectFlags,
) -> TriggerReport:
    self_check = replay_and_diff(scenario_id, seed, turn_cap, original_log, build)
    triggers: dict[str, TriggerResult] = {}
    for defect_id, field in ALL_DEFECT_IDS.items():
        if not getattr(build, field):
            continue
        candidate = build.model_copy(update={field: False})
        result = replay_and_diff(scenario_id, seed, turn_cap, original_log, candidate)
        d = result.divergence
        triggers[defect_id] = TriggerResult(
            defect_id=defect_id,
            triggered=result.diverged,
            step=d.step if d else None,
            field=d.field if d else None,
            detail=(d.detail or f"{d.original} -> {d.candidate}") if d else None,
            entities=_entities(original_log, result),
        )
    return TriggerReport(reproduces=not self_check.diverged, self_check=self_check, triggers=triggers)


# --- Per-step activity --------------------------------------------------------
#
# Leave-one-out gives the FIRST step a defect bit. Grading a flag raised at
# step s needs more: was the defect active anywhere the agent was looking?
# Every TurnRecord's `before` snapshot is the complete dynamic state at the
# top of its turn-slot -- including each character's action_value, since
# the turn scheduler is itself part of dynamic state now that turn order
# is speed-driven rather than fixed -- so the exact state before any step
# can be rebuilt from the log alone, and that one step re-run under the
# build and under the build minus one defect. A difference means the
# defect acted on that very step, whatever happened before it.
# `_state_before` asserts the restored scheduler picks the actor the log
# actually recorded; on a mismatch (which should not happen off a clean
# reproduction -- see leave_one_out_triggers's own self-check) the step is
# reported in `inconsistent_steps` rather than silently simulating the
# wrong character.
#
# Two defects are decided at battle construction, not inside a step, so
# neither has a one-step counterfactual:
#  - B08 (enemy_no_basic_fallback) is active on exactly the steps where a
#    living, unstunned character had no legal action at all -- impossible
#    in a clean build, where every character keeps its fallback.
#  - B11 (tie_break_by_character_id) is active on exactly the steps where
#    the actual actor differs from the spec-correct selection for the
#    action_values `_state_before` restored -- see its activity rule below.

CONSTRUCTION_TIME_DEFECTS = {"B08", "B11"}


class StepActivity(BaseModel):
    steps: dict[str, list[int]]  # defect id -> steps where it acted
    entities: dict[str, dict[int, list[str]]]  # defect id -> step -> characters affected
    inconsistent_steps: list[int]  # steps the build itself didn't reproduce from the log


class SchedulerReconstructionError(Exception):
    """`_state_before` restored every character's action_value from the
    log, but the scheduler it rebuilt would pick a different actor than
    the log recorded. Under the fixed turn order this couldn't happen --
    `current_actor_id()` was derived FROM `record.actor_id`, not from
    independently restored state, so it never had a chance to disagree.
    With action_value in play the actor is derived, not assumed, so this
    is a real (if narrow) failure mode: raised only if a caller ignores
    stepwise_activity's own inconsistent_steps bookkeeping, which is where
    this should normally be caught and reported instead of raised."""


def _state_before(initial_state: BattleState, record: TurnRecord) -> BattleState:
    state = initial_state.model_copy(deep=True)
    for cid, snap in record.before.items():
        c = state.characters[cid]
        c.hp, c.energy, c.alive = snap.hp, snap.energy, snap.alive
        c.statuses = [s.model_copy() for s in snap.statuses]
        c.cooldowns = dict(snap.cooldowns)
        c.action_value = snap.action_value
    state.step_count = record.step_count
    state.round_number = record.round_number
    state.elapsed_av = record.elapsed_av
    state.finished = False
    state.outcome = None
    # The scheduler is derived from the restored action_values, not from
    # record.actor_id -- the whole point of restoring action_value at all
    # is so a wrong reconstruction is CAUGHT here, rather than silently
    # simulating the wrong character (see SchedulerReconstructionError).
    if state.current_actor_id() != record.actor_id:
        raise SchedulerReconstructionError(
            f"restored state selects {state.current_actor_id()!r} but the log recorded "
            f"{record.actor_id!r} at step {record.step_count}"
        )
    return state


def simulate_step(state_before: BattleState, action: Optional[Action], defects: DefectFlags) -> TurnRecord:
    """One turn-slot, mirroring BattleEngine's own sequencing. When the
    candidate build would need a decision the log never made, or would
    reject the logged action, that is recorded in `skipped_reason` -- it is
    itself an outcome difference."""
    state = state_before.model_copy(deep=True)
    rec = resolution.resolve_pre(state, defects)
    resolution.check_battle_end(state)
    if state.finished and rec.needs_action():
        rec.skipped_reason = "battle_ended"
        return rec
    if rec.needs_action():
        if not rules.list_legal_actions(state):
            rec.skipped_reason = "no_legal_actions"
        elif action is None:
            rec.skipped_reason = "decision_required"
            return rec
        else:
            check = rules.check_legal(state, action)
            if not check.legal:
                rec.skipped_reason = f"illegal: {check.reason}"
                return rec
            resolution.resolve_action(state, action, defects, rec)
    resolution.resolve_post(state, defects, rec)
    return rec


def _step_difference(a: TurnRecord, b: TurnRecord) -> Optional[list[str]]:
    """Characters whose outcome differs between two simulations of the same
    step, or None if the step came out identical."""
    if (a.skipped_reason or "") != (b.skipped_reason or ""):
        return [a.actor_id]
    affected: set[str] = set()
    for cid in set(a.after) | set(b.after):
        if cid not in a.after or cid not in b.after:
            affected.add(cid)
            continue
        d = _diff_snapshot(cid, a.after[cid], b.after[cid])
        if d:
            affected.add(cid)
    for cid in set(a.dot_hot_damage) | set(b.dot_hot_damage):
        ta, tb = a.dot_hot_damage.get(cid, {}), b.dot_hot_damage.get(cid, {})
        if any(abs(ta.get(k, 0.0) - tb.get(k, 0.0)) > FLOAT_TOL for k in set(ta) | set(tb)):
            affected.add(cid)
    return sorted(affected) if affected else None


def stepwise_activity(
    initial_state: BattleState,
    original_log: list[TurnRecord],
    build: DefectFlags,
) -> StepActivity:
    enabled = [(did, f) for did, f in ALL_DEFECT_IDS.items() if getattr(build, f)]
    steps: dict[str, list[int]] = {did: [] for did, _ in enabled}
    entities: dict[str, dict[int, list[str]]] = {did: {} for did, _ in enabled}
    inconsistent: list[int] = []

    # B11's activity check needs "who the CLEAN tie-break would have
    # picked", independent of any one step -- computed once, not per
    # record. turn_order construction depends only on the tie-break flag,
    # so flipping it alone (leave-one-out, same as every other defect
    # here) is enough; the other nine flags never affect it.
    clean_turn_order: Optional[list[str]] = None
    if "B11" in steps:
        clean_build = build.model_copy(update={"tie_break_by_character_id": False})
        clean_turn_order = build_battle_state(
            initial_state.scenario_id, initial_state.seed, clean_build, initial_state.turn_cap
        ).turn_order

    for record in original_log:
        k = record.step_count
        if record.skipped_reason == "no_legal_actions" and "B08" in steps:
            steps["B08"].append(k)
            entities["B08"][k] = [record.actor_id]

        if not record.before:
            continue
        try:
            before = _state_before(initial_state, record)
        except SchedulerReconstructionError:
            inconsistent.append(k)
            continue

        if clean_turn_order is not None:
            clean_state = before.model_copy(deep=True)
            clean_state.turn_order = clean_turn_order
            correct_actor = clean_state.current_actor_id()
            if correct_actor != record.actor_id:
                steps["B11"].append(k)
                entities["B11"][k] = sorted({record.actor_id, correct_actor})

        if record.skipped_reason == "dead":
            continue
        baseline = simulate_step(before, record.action, build)
        if _step_difference(baseline, record) is not None:
            inconsistent.append(k)
            continue
        for did, field in enabled:
            if did in CONSTRUCTION_TIME_DEFECTS:
                continue
            candidate = simulate_step(before, record.action, build.model_copy(update={field: False}))
            affected = _step_difference(baseline, candidate)
            if affected is not None:
                steps[did].append(k)
                entities[did][k] = affected
    return StepActivity(steps=steps, entities=entities, inconsistent_steps=inconsistent)
