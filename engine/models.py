"""Pydantic data model for the battle. Serializable; no game logic."""

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


# Re-applying these refreshes the entry. Burn stacks one entry per source.
NON_STACKING_STATUSES = frozenset(st for st in StatusType if st != StatusType.BURN)

Team = Literal["party", "enemy"]

# Scheduler (HSR-style). action_value is an absolute clock reading of a
# character's next turn; the lowest acts next, then adds 10000 / speed.
# Only ever adding to one value (never decrementing all) avoids float drift.
# Speed 100 acts once per CYCLE_AV, so turn_cap still counts rounds.
BASE_ACTION_VALUE = 10000.0
CYCLE_AV = 100.0

# Scenario speeds come from {80, 100, 125, 160, 200, 250} so 10000 / speed
# is exact. Ties compare on a quantized key as a safety net.
_AV_TIE_QUANT = 1_000_000


class StatusApplication(BaseModel):
    """What an ability's effect attaches to its target, declaratively."""

    status_type: StatusType
    magnitude: float
    duration: int  # turns


class Ability(BaseModel):
    name: str
    element: Element
    energy_cost: int = 0
    cooldown: int = 0  # 0 = no cooldown (e.g. Basic Attack)
    target_type: TargetType
    base_power: float = 0.0  # 0 for pure-utility abilities (buff/debuff only)
    applies: Optional[StatusApplication] = None
    is_basic_attack: bool = False  # guaranteed fallback, see defect B08
    is_revive: bool = False  # target_type must be ONE_ALLY
    revive_hp_fraction: float = 0.5


class StatusEffect(BaseModel):
    """One status on a character. For Shield, magnitude is the remaining pool."""

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
    action_value: float = 0.0  # clock reading of this character's next turn

    def ability_by_name(self, name: str) -> Optional[Ability]:
        return next((a for a in self.abilities if a.name == name), None)


class Action(BaseModel):
    actor_id: str
    ability_name: str
    target_id: Optional[str] = None  # None for self / all_enemies abilities


class BattleState(BaseModel):
    """The complete, plain-serializable state of one battle."""

    scenario_id: str
    seed: int
    round_number: int = 0
    step_count: int = 0
    turn_order: list[str]  # tie-break order only
    characters: dict[str, Character]
    turn_cap: int = 30
    finished: bool = False
    outcome: Optional[Literal["party_win", "enemy_win", "turn_cap"]] = None
    elapsed_av: float = 0.0  # scheduler clock
    opening_av: float = 0.0  # clock at battle start; rounds count from here

    def _tie_break_key(self, char_id: str) -> tuple[int, int]:
        return (round(self.characters[char_id].action_value * _AV_TIE_QUANT), self.turn_order.index(char_id))

    def current_actor_id(self) -> str:
        """Lowest action_value acts next. Dead characters still take (skipped) slots."""
        return min(self.turn_order, key=self._tie_break_key)

    def turn_forecast(self, n: int = 6) -> list[tuple[str, float]]:
        """The next `n` living actors, assuming nobody dies or revives."""
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
