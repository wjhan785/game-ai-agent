"""Elements: Fire > Ice > Lightning > Fire. Neutral is always x1.0."""

from engine.models import Element

ADVANTAGE = 1.5
DISADVANTAGE = 1.0 / 1.5
NEUTRAL_MULT = 1.0

_BEATS = {
    Element.FIRE: Element.ICE,
    Element.ICE: Element.LIGHTNING,
    Element.LIGHTNING: Element.FIRE,
}


def elemental_multiplier(attacker: Element, defender: Element) -> float:
    if attacker == Element.NEUTRAL or defender == Element.NEUTRAL:
        return NEUTRAL_MULT
    if _BEATS.get(attacker) == defender:
        return ADVANTAGE
    if _BEATS.get(defender) == attacker:
        return DISADVANTAGE
    return NEUTRAL_MULT
