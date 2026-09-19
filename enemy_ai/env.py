"""Observation/action encoding and a self-play training env.

Fixed shape (6 characters x 3 abilities) so one policy plays every
scenario. Reward is from the acting team's view; `info["team_flip"]`
marks when the other team acts next.
"""

from typing import Any, Optional

import numpy as np

from engine.engine import BattleEngine
from engine.models import (
    CYCLE_AV,
    Action,
    BattleState,
    Character,
    Element,
    StatusType,
    TargetType,
)
from engine.resolution import TurnRecord
from engine.rules import list_legal_actions

CHARACTER_SLOTS: tuple[str, ...] = ("p1", "p2", "p3", "e1", "e2", "e3")
TARGET_SLOTS: tuple[Optional[str], ...] = CHARACTER_SLOTS + (None,)
SCENARIO_SLOTS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")

MAX_ABILITIES = 3
ACTION_SIZE = MAX_ABILITIES * len(TARGET_SLOTS)

ELEMENTS = tuple(Element)
STATUS_TYPES = tuple(StatusType)
TARGET_TYPES = tuple(TargetType)
FORECAST_N = 4

HP_SCALE = 200.0
ENERGY_SCALE = 120.0
POWER_SCALE = 40.0
MAGNITUDE_SCALE = 30.0
DURATION_SCALE = 5.0
SPEED_SCALE = 200.0
REGEN_SCALE = 30.0
COOLDOWN_SCALE = 5.0

DAMAGE_SCALE = 50.0
KILL_BONUS = 0.5
WIN_BONUS = 3.0


# --- Actions ------------------------------------------------------------


def action_parts(index: int) -> tuple[int, Optional[str]]:
    """Action index -> (ability slot, target id)."""
    if not 0 <= index < ACTION_SIZE:
        raise ValueError(f"action index {index} out of range")
    slot, target = divmod(index, len(TARGET_SLOTS))
    return slot, TARGET_SLOTS[target]


def action_index(ability_slot: int, target_id: Optional[str]) -> int:
    return ability_slot * len(TARGET_SLOTS) + TARGET_SLOTS.index(target_id)


def encode_action(state: BattleState, action: Action) -> int:
    actor = state.characters[action.actor_id]
    names = [a.name for a in actor.abilities]
    return action_index(names.index(action.ability_name), action.target_id)


def decode_action(state: BattleState, index: int) -> Optional[Action]:
    """The Action this index means for whoever is acting, or None if the
    actor has no such ability slot."""
    actor_id = state.current_actor_id()
    actor = state.characters[actor_id]
    slot, target_id = action_parts(index)
    if slot >= len(actor.abilities):
        return None
    return Action(actor_id=actor_id, ability_name=actor.abilities[slot].name, target_id=target_id)


def legal_mask(state: BattleState, legal: Optional[list[Action]] = None) -> np.ndarray:
    """Boolean mask over ACTION_SIZE, True where the action is legal now."""
    if legal is None:
        legal = list_legal_actions(state)
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for action in legal:
        mask[encode_action(state, action)] = True
    return mask


def random_action(state: BattleState, rng: np.random.Generator, legal: Optional[list[Action]] = None) -> Action:
    """A uniform-random legal move for whoever is acting."""
    mask = legal_mask(state, legal)
    action = decode_action(state, int(rng.choice(np.flatnonzero(mask))))
    assert action is not None
    return action


# --- Observation --------------------------------------------------------

CHARACTER_BLOCK = 11 + len(ELEMENTS) + 4 * len(STATUS_TYPES) + 2
ABILITY_BLOCK = 9 + len(ELEMENTS) + len(TARGET_TYPES) + len(STATUS_TYPES) + 3
GLOBAL_BLOCK = len(SCENARIO_SLOTS) + 4 + FORECAST_N * len(CHARACTER_SLOTS)
OBS_SIZE = CHARACTER_BLOCK * len(CHARACTER_SLOTS) + ABILITY_BLOCK * MAX_ABILITIES + GLOBAL_BLOCK


def _one_hot(value: Any, options: tuple) -> list[float]:
    return [1.0 if value == option else 0.0 for option in options]


def _character_block(state: BattleState, char_id: str, current: str) -> list[float]:
    c = state.characters.get(char_id)
    if c is None:
        return [0.0] * CHARACTER_BLOCK
    out = [
        1.0,  # slot filled
        1.0 if c.alive else 0.0,
        c.hp / c.max_hp if c.max_hp else 0.0,
        c.max_hp / HP_SCALE,
        c.energy / c.max_energy if c.max_energy else 0.0,
        c.energy / ENERGY_SCALE,
        c.energy_regen / REGEN_SCALE,
        c.speed / SPEED_SCALE,
        (c.action_value - state.elapsed_av) / CYCLE_AV,
        1.0 if char_id == current else 0.0,
        1.0 if c.team == "party" else 0.0,
    ]
    out += _one_hot(c.element, ELEMENTS)
    for status_type in STATUS_TYPES:
        held = [s for s in c.statuses if s.status_type == status_type]
        out += [
            1.0 if held else 0.0,
            len(held) / 3.0,
            sum(s.magnitude for s in held) / MAGNITUDE_SCALE,
            max((s.turns_remaining for s in held), default=0) / DURATION_SCALE,
        ]
    live_cds = [v for v in c.cooldowns.values() if v > 0]
    out += [len(live_cds) / MAX_ABILITIES, max(live_cds, default=0) / COOLDOWN_SCALE]
    return out


def _ability_block(actor: Optional[Character], slot: int) -> list[float]:
    if actor is None or slot >= len(actor.abilities):
        return [0.0] * ABILITY_BLOCK
    a = actor.abilities[slot]
    cd = actor.cooldowns.get(a.name, 0)
    out = [
        1.0,  # slot filled
        a.energy_cost / POWER_SCALE,
        1.0 if actor.energy >= a.energy_cost else 0.0,
        a.cooldown / COOLDOWN_SCALE,
        cd / COOLDOWN_SCALE,
        1.0 if cd > 0 else 0.0,
        a.base_power / POWER_SCALE,
        1.0 if a.is_basic_attack else 0.0,
        1.0 if a.is_revive else 0.0,
    ]
    out += _one_hot(a.element, ELEMENTS)
    out += _one_hot(a.target_type, TARGET_TYPES)
    applied = a.applies
    out += _one_hot(applied.status_type if applied else None, STATUS_TYPES)
    out += [
        1.0 if applied else 0.0,
        (applied.magnitude / MAGNITUDE_SCALE) if applied else 0.0,
        (applied.duration / DURATION_SCALE) if applied else 0.0,
    ]
    return out


def encode_observation(state: BattleState) -> np.ndarray:
    current = state.current_actor_id()
    actor = state.characters.get(current)
    out: list[float] = []
    for char_id in CHARACTER_SLOTS:
        out += _character_block(state, char_id, current)
    for slot in range(MAX_ABILITIES):
        out += _ability_block(actor, slot)

    out += _one_hot(state.scenario_id, SCENARIO_SLOTS)
    out += [
        state.round_number / max(state.turn_cap, 1),
        state.step_count / 100.0,
        len(state.alive_on_team("party")) / 3.0,
        len(state.alive_on_team("enemy")) / 3.0,
    ]
    forecast = [cid for cid, _ in state.turn_forecast(FORECAST_N)]
    for i in range(FORECAST_N):
        out += _one_hot(forecast[i] if i < len(forecast) else None, CHARACTER_SLOTS)

    obs = np.asarray(out, dtype=np.float32)
    if obs.shape[0] != OBS_SIZE:
        raise AssertionError(f"observation is {obs.shape[0]} wide, expected {OBS_SIZE}")
    return obs


# --- Reward -------------------------------------------------------------


def combat_reward(engine: BattleEngine, records: list[TurnRecord], team: str) -> float:
    """Reward from `team`'s view: hp taken off the other side and restored
    on this one, kills, and the battle's outcome."""
    total = 0.0
    kills = 0
    for record in records:
        for char_id, before in record.before.items():
            after = record.after.get(char_id)
            if after is None:
                continue
            lost = before.hp - after.hp
            own = engine.state.characters[char_id].team == team
            total += (-lost if own else lost) / DAMAGE_SCALE
        for char_id in record.deaths:
            kills += -1 if engine.state.characters[char_id].team == team else 1
    total += KILL_BONUS * kills

    if engine.state.finished:
        winner = {"party_win": "party", "enemy_win": "enemy"}.get(engine.state.outcome or "")
        if winner is not None:
            total += WIN_BONUS if winner == team else -WIN_BONUS
    return total


# --- Env ----------------------------------------------------------------


class CombatEnv:
    """One battle, one step per decision, for self-play training."""

    def __init__(
        self,
        scenario_ids: Optional[list[str]] = None,
        *,
        defects: Any = None,
        turn_cap: int = 20,
        max_decisions: int = 150,
        seed_base: int = 10_000,
    ):
        self.scenario_ids = list(scenario_ids or SCENARIO_SLOTS)
        self.defects = defects
        self.turn_cap = turn_cap
        self.max_decisions = max_decisions
        self.seed_base = seed_base
        self.rng = np.random.default_rng(0)
        self.engine: Optional[BattleEngine] = None
        self.decisions = 0
        self._processed = 0

    def reset(
        self, *, seed: Optional[int] = None, scenario_id: Optional[str] = None, battle_seed: Optional[int] = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if scenario_id is None:
            scenario_id = self.scenario_ids[int(self.rng.integers(len(self.scenario_ids)))]
        if battle_seed is None:
            battle_seed = self.seed_base + int(self.rng.integers(1_000_000))
        self.engine = BattleEngine(scenario_id, battle_seed, self.defects, self.turn_cap)
        self.decisions = 0
        self._processed = len(self.engine.log)
        return encode_observation(self.engine.state), self._info()

    def step(self, index: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        engine = self.engine
        if engine is None:
            raise RuntimeError("call reset() first")
        action = decode_action(engine.state, index)
        if action is None or not legal_mask(engine.state)[index]:
            raise ValueError(f"illegal action index {index}")

        team = engine.state.characters[action.actor_id].team
        engine.take_action(action)
        self.decisions += 1
        records = engine.log[self._processed :]
        self._processed = len(engine.log)
        reward = combat_reward(engine, records, team)

        terminated = engine.state.finished
        truncated = not terminated and self.decisions >= self.max_decisions
        obs = np.zeros(OBS_SIZE, dtype=np.float32) if terminated else encode_observation(engine.state)
        info = self._info()
        info["team_flip"] = (
            False
            if terminated
            else engine.state.characters[engine.state.current_actor_id()].team != team
        )
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        engine = self.engine
        if engine is None or engine.state.finished:
            return np.zeros(ACTION_SIZE, dtype=bool)
        return legal_mask(engine.state, engine.legal_actions())

    def _info(self) -> dict:
        return {"action_mask": self.action_masks(), "decisions": self.decisions}
