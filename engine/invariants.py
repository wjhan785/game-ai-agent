"""Post-action invariant checks.

This module runs after every single action, independently of whether the
agent (or anyone) noticed anything wrong -- that independence is the whole
point: deterministic code catches state corruption; the LLM's job is only
to decide what to try next, never to be the thing eyeballing a raw state
dump for an off-by-one.

Deliberately, nothing in this module imports engine.defects or references
a defect flag. These are checks on STATE VALIDITY, not on which switch is
flipped -- the same independence property that makes the differential
oracle a meaningful ground truth (see oracle/differential.py) applies here.
A violation says "something about this state is wrong"; it says nothing
about which of the ten seeded defects caused it. Six of the ten are
reliably visible here (see engine.defects.INVARIANT_VISIBLE); the other
four are not, by design -- see docs/scenario-matrix.md and RESULTS.md for
why that gap is the interesting part of this project.
"""
from __future__ import annotations

from engine.models import CYCLE_AV, BattleState, NON_STACKING_STATUSES, StatusType
from engine.resolution import TurnRecord

PERIODIC_STATUS_TYPES = frozenset(
    {StatusType.BURN, StatusType.POISON, StatusType.REGEN}
)


def check(state: BattleState, record: TurnRecord) -> list[str]:
    violations: list[str] = []
    violations.extend(_check_resource_bounds(state))
    violations.extend(_check_duplicate_non_stacking(state))
    violations.extend(_check_dead_carrying_periodic_status(state))
    violations.extend(_check_zero_shield_present(state))
    violations.extend(_check_illegal_action_on_cooldown(state, record))
    violations.extend(_check_dead_actor_took_action(record))
    violations.extend(_check_action_values(state))
    return violations


def _check_resource_bounds(state: BattleState) -> list[str]:
    out = []
    for cid, c in state.characters.items():
        if c.hp < 0:
            out.append(f"{cid}: hp went negative ({c.hp})")
        if c.hp > c.max_hp + 1e-6:
            out.append(f"{cid}: hp above max_hp ({c.hp} > {c.max_hp})")
        if c.energy < 0:
            out.append(f"{cid}: energy went negative ({c.energy})")
        if c.energy > c.max_energy + 1e-6:
            out.append(f"{cid}: energy above max_energy ({c.energy} > {c.max_energy})")
    return out


def _check_duplicate_non_stacking(state: BattleState) -> list[str]:
    out = []
    for cid, c in state.characters.items():
        for status_type in NON_STACKING_STATUSES:
            count = sum(1 for s in c.statuses if s.status_type == status_type)
            if count > 1:
                out.append(
                    f"{cid}: {count} simultaneous {status_type.value} entries "
                    f"(non-stacking status should have at most 1)"
                )
    return out


def _check_dead_carrying_periodic_status(state: BattleState) -> list[str]:
    out = []
    for cid, c in state.characters.items():
        if c.alive:
            continue
        for s in c.statuses:
            if s.status_type in PERIODIC_STATUS_TYPES:
                out.append(
                    f"{cid}: dead but still carrying {s.status_type.value} "
                    f"(will resume ticking if revived)"
                )
    return out


def _check_zero_shield_present(state: BattleState) -> list[str]:
    out = []
    for cid, c in state.characters.items():
        for s in c.statuses:
            if s.status_type == StatusType.SHIELD and s.magnitude <= 0:
                out.append(f"{cid}: zero-magnitude shield still present in status list")
    return out


def _check_illegal_action_on_cooldown(state: BattleState, record: TurnRecord) -> list[str]:
    if record.action is None:
        return []
    before = record.before.get(record.actor_id)
    if before is None:
        return []
    cd = before.cooldowns.get(record.action.ability_name, 0)
    if cd > 0:
        return [
            f"{record.actor_id}: used {record.action.ability_name} while it had "
            f"{cd} turn(s) of cooldown remaining"
        ]
    return []


def _check_dead_actor_took_action(record: TurnRecord) -> list[str]:
    if record.action is None:
        return []
    before = record.before.get(record.actor_id)
    if before is not None and not before.alive:
        return [f"{record.actor_id}: took an action while dead"]
    return []


def _check_action_values(state: BattleState) -> list[str]:
    """Internal scheduler consistency only -- never WHICH character was
    selected. Selection conformance to the declared tie-break order is a
    spec question (that is exactly what defect B11 violates, and what the
    differential oracle -- not this checker -- is the ground truth for);
    encoding the correct-selection rule here would make this module a
    second scheduler implementation and catch B11 for the wrong reason,
    the same independence this module's own docstring insists on for
    every other defect.
    """
    out = []
    for cid, c in state.characters.items():
        if c.speed <= 0:
            out.append(f"{cid}: speed is not positive ({c.speed})")
        if c.action_value < 0:
            out.append(f"{cid}: action_value went negative ({c.action_value})")
        if c.action_value < state.elapsed_av - 1e-6:
            out.append(
                f"{cid}: action_value ({c.action_value}) is behind the scheduler "
                f"clock ({state.elapsed_av})"
            )
    expected_round = int((state.elapsed_av - state.opening_av) / CYCLE_AV)
    if state.round_number != expected_round:
        out.append(
            f"round_number {state.round_number} inconsistent with elapsed_av "
            f"{state.elapsed_av} (expected {expected_round})"
        )
    return out
