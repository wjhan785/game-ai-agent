"""BattleEngine: the facade everything drives. Turn-slots that need no
decision (dead, stunned, no legal action) resolve automatically."""

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

    def pending_record(self) -> resolution.TurnRecord | None:
        """A copy of the pending turn-slot resolved so far, or None."""
        if self._pending_record is None:
            return None
        return self._pending_record.model_copy(deep=True)

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
            # A tick can end the battle before any action.
            resolution.check_battle_end(self.state)
            if self.state.finished and record.needs_action():
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
