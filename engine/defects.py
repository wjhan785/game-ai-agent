"""The eleven seeded defects, as switches -- not baked in.

Every defect is a boolean flag, defaulting to False (clean/correct
behavior). Each flag guards exactly one branch in effects.py, rules.py,
resolution.py, or scenarios.py; grep the flag name to find its branch.
This is what lets golden tests assert correct semantics against clean
mode, lets the oracle attribute a divergence to a single flag by
ablation, and lets ANSWER_KEY.md be generated from this file (via
scripts/generate_answer_key.py) instead of drifting from it -- each
field's `description` below is that generator's source of truth, not
just a comment.

Semantics of each flag are documented here; the numbering (B01..B11)
matches ANSWER_KEY.md and RESULTS.md throughout the project.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DefectFlags(BaseModel):
    shield_absorbs_dot: bool = Field(
        default=False,
        description=(
            "Shield absorbs Poison/Burn tick damage instead of only direct "
            "ability hits. Clean: DoT ticks bypass shield entirely."
        ),
    )

    burn_stacks_multiply: bool = Field(
        default=False,
        description=(
            "Two Burn stacks from different sources, both freshly applied "
            "in the same resolution step, combine multiplicatively at tick "
            "time instead of additively. Clean: burn tick damage is always "
            "the sum of every active Burn instance's magnitude."
        ),
    )

    chill_no_floor: bool = Field(
        default=False,
        description=(
            "Chill's energy-regen reduction has no floor: neither the "
            "summed Chill percentage nor the resulting energy delta is "
            "clamped, so energy can go negative when a strong Chill "
            "reduction exceeds 100%. Clean: total Chill is capped at 100% "
            "and the regen result is clamped to >= 0."
        ),
    )

    poison_reads_weakened_value: bool = Field(
        default=False,
        description=(
            "A target that is both Weakened and Poisoned has its "
            "percentage-of-max-HP Poison tick computed against the "
            "Weaken-adjusted value instead of the target's own raw max_hp. "
            "Clean: Poison always reads the target's unmodified max_hp, "
            "independent of any other status the target is carrying."
        ),
    )

    stun_adds_instead_of_refresh: bool = Field(
        default=False,
        description=(
            "Applying Stun to an already-stunned target adds a second "
            "status entry (and its duration) instead of refreshing the "
            "existing one in place. Clean: Stun is non-stacking; a fresh "
            "application replaces the remaining duration."
        ),
    )

    cooldown_decrement_before_check: bool = Field(
        default=False,
        description=(
            "Cooldowns are decremented an EXTRA time before the "
            "legal-move check runs for that turn, on top of the normal "
            "end-of-turn decrement, so a cooldown drains twice as fast as "
            "intended and an ability becomes usable one of the actor's own "
            "turns earlier than it should. Clean: cooldowns are "
            "decremented exactly once per own-turn, as end-of-turn "
            "bookkeeping, strictly after that turn's legal-move check has "
            "already run."
        ),
    )

    regen_survives_death: bool = Field(
        default=False,
        description=(
            "Regen is not cleared when its owner dies, so if the owner is "
            "later revived by a separate effect, the stale Regen resumes "
            "ticking using its leftover duration. Clean: all periodic "
            "statuses are cleared at the moment of death."
        ),
    )

    enemy_no_basic_fallback: bool = Field(
        default=False,
        description=(
            "An enemy with every real ability on cooldown or unaffordable "
            "has no basic-attack fallback, so it registers zero legal "
            "actions forever instead of the occasional single-turn skip. "
            "Clean: every character (including enemies) always has a "
            "0-cost, 0-cooldown Basic Attack as a guaranteed fallback."
        ),
    )

    elemental_multiplier_applied_twice: bool = Field(
        default=False,
        description=(
            "The elemental advantage multiplier is applied twice: once "
            "inside the ability's own damage calculation, and again in "
            "the general post-hoc damage-modifier pass. Clean: elemental "
            "advantage is applied exactly once, in the general modifier "
            "pass only."
        ),
    )

    zero_shield_not_removed: bool = Field(
        default=False,
        description=(
            "A Shield whose absorption pool has reached zero is not "
            "removed from the status list, so later has_status(SHIELD) "
            "checks still return True. Clean: a Shield is stripped the "
            "moment its magnitude reaches zero (or its turn-count decay "
            "expires)."
        ),
    )

    tie_break_by_character_id: bool = Field(
        default=False,
        description=(
            "When two or more characters are tied for the next turn (equal "
            "action value), the tie is broken by sorting character id "
            "(e1, e2, e3, p1, p2, p3) instead of the scenario's declared "
            "turn order (p1, e1, p2, e2, p3, e3, ...). Clean: ties are "
            "always broken by the declared turn order, so a full round at "
            "equal speed proceeds party-then-enemy, alternating."
        ),
    )

    def enabled(self) -> list[str]:
        return [name for name, value in self.__dict__.items() if value is True]


ALL_DEFECT_IDS = {
    "B01": "shield_absorbs_dot",
    "B02": "burn_stacks_multiply",
    "B03": "chill_no_floor",
    "B04": "poison_reads_weakened_value",
    "B05": "stun_adds_instead_of_refresh",
    "B06": "cooldown_decrement_before_check",
    "B07": "regen_survives_death",
    "B08": "enemy_no_basic_fallback",
    "B09": "elemental_multiplier_applied_twice",
    "B10": "zero_shield_not_removed",
    "B11": "tie_break_by_character_id",
}


INVARIANT_VISIBLE = {"B03", "B05", "B06", "B07", "B08", "B10"}
INVARIANT_INVISIBLE = {"B01", "B02", "B04", "B09", "B11"}


def single_flag(defect_id: str) -> DefectFlags:
    """DefectFlags with exactly one defect enabled -- for ablation tests
    and for the oracle's per-flag attribution."""
    field = ALL_DEFECT_IDS[defect_id]
    return DefectFlags(**{field: True})
