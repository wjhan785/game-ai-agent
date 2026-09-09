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
    turn_order: list[str]
    turn_pointer: int = 0
    characters: dict[str, Character]
    turn_cap: int = 30
    finished: bool = False
    outcome: Optional[Literal["party_win", "enemy_win", "turn_cap"]] = None

    def current_actor_id(self) -> str:
        return self.turn_order[self.turn_pointer]

    def alive_on_team(self, team: Team) -> list[Character]:
        return [c for c in self.characters.values() if c.team == team and c.alive]
