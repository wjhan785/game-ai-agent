"""Invariant checks run after every action. They test state validity only
and never look at defect flags, so they catch some defects (INVARIANT_VISIBLE)
and, by design, miss the rest."""

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
    """Scheduler consistency only, never WHO was selected: checking that
    would re-implement the scheduler (the oracle's job, not this module's)."""
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
