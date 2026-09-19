"""Shared test helpers and fixtures."""

import pytest

from engine.content import AEGIS_WARD, make_character
from engine.engine import BattleEngine
from engine.models import Element
from engine import scenarios as scenarios_module


def advance_to_actor(engine: BattleEngine, actor_id: str) -> None:
    """Play first-legal actions until it is `actor_id`'s turn. Only safe when
    the filler actions can't disturb what the test checks."""
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
    """1v1 Warden vs a Fire attacker, for exact shield arithmetic with no
    filler turns in between."""
    scenarios_module.SCENARIOS[SHIELD_TEST_SCENARIO_ID] = _build_shield_test_scenario
    try:
        yield SHIELD_TEST_SCENARIO_ID
    finally:
        del scenarios_module.SCENARIOS[SHIELD_TEST_SCENARIO_ID]
