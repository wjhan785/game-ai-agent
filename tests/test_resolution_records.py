"""TurnRecord bookkeeping, as opposed to game state: the record is what the
agent's tool surface reports, so a wrong record field is a false-positive
generator even when the underlying state is correct."""

from engine.engine import BattleEngine
from engine.models import Action
from tests.conftest import advance_to_actor


def test_disadvantaged_hit_on_unshielded_target_records_no_absorption():
    # S2: p1 Pyra (fire) Ember Slash (18 power) into e2 Skirmisher1
    # (lightning) -- fire is at a disadvantage, x(1/1.5) -> 12 damage.
    eng = BattleEngine("S2", seed=0)
    advance_to_actor(eng, "p1")
    rec = eng.take_action(Action(actor_id="p1", ability_name="Ember Slash", target_id="e2"))
    assert rec.after["e2"].statuses == []
    assert abs(rec.ability_effect.per_target_damage["e2"] - 12.0) < 1e-9
    assert rec.ability_effect.per_target_shield_absorbed == {}


def test_shielded_hit_records_exactly_the_pool_consumed(shield_test_scenario):
    # Warden shields (30 pool), then e1's Basic Attack (8, x1.0) is fully absorbed.
    eng = BattleEngine(shield_test_scenario, seed=0)
    eng.take_action(Action(actor_id="p1", ability_name="Aegis Ward", target_id="p1"))
    rec = eng.take_action(Action(actor_id="e1", ability_name="Basic Attack", target_id="p1"))
    assert rec.ability_effect.per_target_damage["p1"] == 0.0
    assert abs(rec.ability_effect.per_target_shield_absorbed["p1"] - 8.0) < 1e-9
    shield = next(s for s in rec.after["p1"].statuses if s.status_type.value == "shield")
    assert abs(shield.magnitude - 22.0) < 1e-9
