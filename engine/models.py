"""Plain, serializable data model for the battle engine.

Everything here is a Pydantic model so the whole BattleState can be dumped
to JSON at any point (for logging, replay, and the differential oracle) and
reconstructed from it exactly. Nothing here does game logic -- that lives in
effects.py, resolution.py, rules.py.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Element(str, Enum):
    FIRE = "fire"
    ICE = "ice"
    LIGHTNING = "lightning"
    NEUTRAL = "neutral"


class TargetType(str, Enum):
    SELF = "self"
    ONE_ENEMY = "one_enemy"
    ONE_ALLY = "one_ally"
    ALL_ENEMIES = "all_enemies"


class StatusType(str, Enum):
    BURN = "burn"
    POISON = "poison"
    SHIELD = "shield"
    REGEN = "regen"
    CHILL = "chill"
    WEAKEN = "weaken"
    STUN = "stun"


# Statuses that do NOT stack -- a fresh application refreshes the existing
# entry (new magnitude, new duration) instead of adding a second one. Burn
# is the one deliberate exception in this engine: each source keeps its own
# entry so tick damage sums across sources (see effects.tick_burn).
NON_STACKING_STATUSES = frozenset(st for st in StatusType if st != StatusType.BURN)

Team = Literal["party", "enemy"]

# --- Scheduler constants (action-value turn order) ---------------------
#
# A character's action_value is an ABSOLUTE clock reading -- the elapsed_av
# at which they next act -- not a countdown. Whoever holds the minimum acts
# next; after acting, their action_value becomes elapsed_av + one personal
# cycle (BASE_ACTION_VALUE / speed). This is deliberately additive, never a
# per-tick decrement applied to every character: decrementing N-1 other
# characters' counters on every turn-slot is where float drift comes from,
# and it is avoided entirely by only ever adding an exact constant to one
# character's own last-known-exact value.
#
# BASE_ACTION_VALUE / CYCLE_AV are both 10000 / 100, so a speed-100
# character (the default, and the reference speed) acts exactly once per
# CYCLE_AV of elapsed clock -- matching Honkai: Star Rail's convention, and
# chosen so that turn_cap keeps meaning "this many rounds" without every
# call site needing to change (see BattleState.round_number).
BASE_ACTION_VALUE = 10000.0
CYCLE_AV = 100.0

# Scenario speeds must be chosen so BASE_ACTION_VALUE / speed is exactly
# representable in float64 (multiples of 0.5) -- e.g. from
# {80, 100, 125, 160, 200, 250} -> {125, 100, 80, 62.5, 50, 40} -- so that
# repeated addition never drifts and equal-speed ties are bit-exact. Ties
# are still resolved on a quantized key (below), as a second line of
# defense for any speed outside that palette.
_AV_TIE_QUANT = 1_000_000


class StatusApplication(BaseModel):
    """What an ability's effect attaches to its target, declaratively."""

    status_type: StatusType
    magnitude: float
    duration: int  # turns; for Shield this is also the decay-by-turn-count N


class Ability(BaseModel):
    name: str
    element: Element
    energy_cost: int = 0
    cooldown: int = 0  # 0 = no cooldown (e.g. Basic Attack)
    target_type: TargetType
    base_power: float = 0.0  # 0 for pure-utility abilities (buff/debuff only)
    applies: Optional[StatusApplication] = None
    is_basic_attack: bool = False  # guaranteed fallback, see defect B08
    is_revive: bool = False  # target_type must be ONE_ALLY; see resolution.py
    revive_hp_fraction: float = 0.5


class StatusEffect(BaseModel):
    """One live instance of a status on a character.

    Burn is the only status with more than one simultaneous instance per
    character (one per applying source) -- see NON_STACKING_STATUSES.
    For Shield, `magnitude` is the remaining absorption pool and
    `turns_remaining` is the N-turn decay clock; it expires on whichever
    hits zero first (see effects.py).
    """

    status_type: StatusType
    magnitude: float
    turns_remaining: int
    source_id: str


class Character(BaseModel):
    id: str
    name: str
    team: Team
    element: Element
    max_hp: float
    hp: float
    max_energy: float
    energy: float
    energy_regen: float
    abilities: list[Ability]
    statuses: list[StatusEffect] = Field(default_factory=list)
    cooldowns: dict[str, int] = Field(default_factory=dict)  # ability name -> turns left
    alive: bool = True
    speed: float = 100.0
    action_value: float = 0.0  # absolute clock reading of this character's next turn; see build_battle_state

    def ability_by_name(self, name: str) -> Optional[Ability]:
        return next((a for a in self.abilities if a.name == name), None)


class Action(BaseModel):
    actor_id: str
    ability_name: str
    target_id: Optional[str] = None  # None only legal for target_type in {self, all_enemies}


class BattleState(BaseModel):
    """The complete, plain-serializable state of one battle."""

    scenario_id: str
    seed: int
    round_number: int = 0
    step_count: int = 0
    turn_order: list[str]  # tie-break order only, now -- see current_actor_id
    characters: dict[str, Character]
    turn_cap: int = 30
    finished: bool = False
    outcome: Optional[Literal["party_win", "enemy_win", "turn_cap"]] = None
    elapsed_av: float = 0.0  # the scheduler clock: the acting character's own action_value at selection
    opening_av: float = 0.0  # elapsed_av at battle start; round_number is measured from here

    def _tie_break_key(self, char_id: str) -> tuple[int, int]:
        return (round(self.characters[char_id].action_value * _AV_TIE_QUANT), self.turn_order.index(char_id))

    def current_actor_id(self) -> str:
        """The character with the lowest action_value -- everyone in
        turn_order is a candidate regardless of alive/dead, since a dead
        character still consumes a turn-slot (see resolution.resolve_pre)."""
        return min(self.turn_order, key=self._tie_break_key)

    def turn_forecast(self, n: int = 6) -> list[tuple[str, float]]:
        """Pure lookahead: the next `n` selections among currently-alive
        characters, assuming no deaths, revives, or speed changes happen
        in between -- planning information for the agent, not a guarantee
        about what will actually happen."""
        projected = {cid: c.action_value for cid, c in self.characters.items() if c.alive}
        speeds = {cid: self.characters[cid].speed for cid in projected}
        forecast: list[tuple[str, float]] = []
        for _ in range(n):
            if not projected:
                break
            key = lambda cid: (round(projected[cid] * _AV_TIE_QUANT), self.turn_order.index(cid))
            next_id = min(projected, key=key)
            forecast.append((next_id, projected[next_id]))
            projected[next_id] += BASE_ACTION_VALUE / speeds[next_id]
        return forecast

    def alive_on_team(self, team: Team) -> list[Character]:
        return [c for c in self.characters.values() if c.team == team and c.alive]
