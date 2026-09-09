"""BattleEngine: the facade the agent, baselines, and oracle all drive.

reset / legal_actions / take_action / state / log -- that's the whole
public surface. Internally it stitches together rules.py's legal-move
checker and resolution.py's three-phase pipeline into a single loop that
auto-resolves any turn-slot that doesn't need a real decision (a dead
character's turn, a stunned character's turn, or -- defect B08 -- an
enemy with zero affordable, off-cooldown abilities and no fallback), so
callers only ever see turns where a choice actually matters.
"""
from __future__ import annotations

from engine import invariants, resolution, rules, scenarios
from engine.defects import DefectFlags
from engine.models import Action, BattleState

NO_PROGRESS_THRESHOLD = 3


class IllegalActionError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class BattleEngine:
    def __init__(
        self,
        scenario_id: str,
        seed: int,
        defects: DefectFlags | None = None,
        turn_cap: int = 30,
    ):
        self.defects = defects if defects is not None else DefectFlags()
        self.turn_cap = turn_cap
        self.log: list[resolution.TurnRecord] = []
        self._pending_record: resolution.TurnRecord | None = None
        self._pending_legal: list[Action] = []
        self._no_progress_counts: dict[str, int] = {}
        self.state: BattleState
        self.reset(scenario_id, seed)

    def reset(self, scenario_id: str, seed: int) -> None:
        self.state = scenarios.build_battle_state(
            scenario_id, seed, self.defects, self.turn_cap
        )
        self.log = []
        self._pending_record = None
        self._pending_legal = []
        self._no_progress_counts = {}
        self._advance_to_decision()

    def legal_actions(self) -> list[Action]:
        if self.state.finished:
            return []
        return list(self._pending_legal)

    def take_action(self, action: Action) -> resolution.TurnRecord:
        if self.state.finished:
            raise RuntimeError("battle already finished")
        if self._pending_record is None:
            raise RuntimeError("no decision pending; call reset() first")

        legal = rules.check_legal(self.state, action)
        if not legal.legal:
            raise IllegalActionError(legal.reason)

        record = self._pending_record
        self._pending_record = None
        resolution.resolve_action(self.state, action, self.defects, record)
        resolution.resolve_post(self.state, self.defects, record)
        self._finalize(record)
        self._advance_to_decision()
        return record

    def _advance_to_decision(self) -> None:
        while not self.state.finished:
            record = resolution.resolve_pre(self.state, self.defects)
            # A DoT tick during resolve_pre can end the battle before any
            # action is even considered (e.g. it kills the last enemy) --
            # check immediately rather than waiting for resolve_post,
            # or we'd wrongly treat this as a pending decision.
            resolution.check_battle_end(self.state)
            if self.state.finished and record.needs_action():
                # The turn-slot's pre-phase ran, but the battle ended
                # before an action could be taken this "turn" -- there is
                # nothing left to decide or finalize downstream, so just
                # record it as a no-op skip for logging purposes.
                record.skipped_reason = "battle_ended"
                self.log.append(record)
                return

            if record.needs_action():
                legal = rules.list_legal_actions(self.state)
                if legal:
                    self._pending_record = record
                    self._pending_legal = legal
                    return
                record.skipped_reason = "no_legal_actions"

            resolution.resolve_post(self.state, self.defects, record)
            self._finalize(record)

    def _finalize(self, record: resolution.TurnRecord) -> None:
        record.invariant_violations = invariants.check(self.state, record)
        self._track_no_progress(record)
        self.log.append(record)

    def _track_no_progress(self, record: resolution.TurnRecord) -> None:
        if record.skipped_reason == "no_legal_actions":
            n = self._no_progress_counts.get(record.actor_id, 0) + 1
            self._no_progress_counts[record.actor_id] = n
            if n >= NO_PROGRESS_THRESHOLD:
                record.invariant_violations.append(
                    f"{record.actor_id}: no legal actions for {n} consecutive "
                    f"turns (no-progress)"
                )
        else:
            self._no_progress_counts[record.actor_id] = 0

    def state_dict(self) -> dict:
        return self.state.model_dump(mode="json")

    def log_dicts(self) -> list[dict]:
        return [r.model_dump(mode="json") for r in self.log]
