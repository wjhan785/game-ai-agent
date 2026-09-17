"""Shared test helpers and fixtures."""
from __future__ import annotations

import pytest

from engine.content import AEGIS_WARD, make_character
from engine.engine import BattleEngine
from engine.models import Element
from engine import scenarios as scenarios_module


def advance_to_actor(engine: BattleEngine, actor_id: str) -> None:
    """Submits each intervening actor's first legal action until it is
    `actor_id`'s turn. Scenarios carry varied per-character speeds (see
    engine/scenarios.py), so a test that needs a SPECIFIC character's
    turn -- regardless of whatever the scenario's own speed balance makes
    that character's natural position -- drives forward to it this way
    rather than assuming any particular actor goes first.

    Only safe when no OTHER actor's filler action can disturb what the
    test is about to check -- e.g. an enemy's first-legal Basic Attack
    targets the first alive party member, which is often the very
    character a test is trying to keep pristine (see
    shield_test_scenario for the case where that matters)."""
    while engine.state.current_actor_id() != actor_id:
        if engine.state.finished or not engine.legal_actions():
            raise AssertionError(f"battle ended or stuck before {actor_id!r}'s turn came up")
        engine.take_action(engine.legal_actions()[0])


SHIELD_TEST_SCENARIO_ID = "_T_SHIELD_VS_BASIC"


def _build_shield_test_scenario(seed, defects):
    party = [make_character("p1", "Warden", "party", Element.NEUTRAL, 110, 100, 22, [AEGIS_WARD], speed=100)]
    enemy = [make_character("e1", "Attacker", "enemy", Element.FIRE, 95, 90, 20, [], speed=100)]
    return party + enemy, ["p1", "e1"]


@pytest.fixture
def shield_test_scenario():
    """A 1v1 -- Warden (Aegis Ward) vs a lone Fire attacker (Basic Attack
    only) -- for tests that need an exact, undisturbed shield-absorption
    figure: no other character exists to spend a filler turn hitting the
    shield before the test's own explicit action does."""
    scenarios_module.SCENARIOS[SHIELD_TEST_SCENARIO_ID] = _build_shield_test_scenario
    try:
        yield SHIELD_TEST_SCENARIO_ID
    finally:
        del scenarios_module.SCENARIOS[SHIELD_TEST_SCENARIO_ID]
