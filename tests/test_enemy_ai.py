"""The enemy AI: encoding, env, reward, network, training, checkpoints."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

import numpy as np
import torch

from enemy_ai.env import (
    ACTION_SIZE,
    CHARACTER_SLOTS,
    OBS_SIZE,
    SCENARIO_SLOTS,
    CombatEnv,
    action_index,
    action_parts,
    combat_reward,
    decode_action,
    encode_action,
    encode_observation,
    legal_mask,
)
from enemy_ai.policy import (
    ActorCritic,
    TrainConfig,
    choose_action,
    head_to_head,
    load_checkpoint,
    save_checkpoint,
    train,
)
from engine.engine import BattleEngine
from engine.rules import list_legal_actions

SCENARIOS = list(SCENARIO_SLOTS)


# --- Encoding -------------------------------------------------------------


def test_action_index_round_trips():
    seen = set()
    for index in range(ACTION_SIZE):
        slot, target = action_parts(index)
        assert action_index(slot, target) == index
        seen.add((slot, target))
    assert len(seen) == ACTION_SIZE


def test_action_parts_rejects_out_of_range():
    with pytest.raises(ValueError):
        action_parts(ACTION_SIZE)


def test_character_slots_cover_every_scenario():
    for scenario_id in SCENARIOS:
        engine = BattleEngine(scenario_id, 1, None, 20)
        assert set(engine.state.characters) == set(CHARACTER_SLOTS)
        for c in engine.state.characters.values():
            assert len(c.abilities) <= 3


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_observation_is_fixed_width_and_finite(scenario_id):
    for seed in (1, 7, 12345):
        obs = encode_observation(BattleEngine(scenario_id, seed, None, 20).state)
        assert obs.shape == (OBS_SIZE,)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_mask_matches_the_legal_move_checker(scenario_id):
    engine = BattleEngine(scenario_id, 3, None, 20)
    rng = np.random.default_rng(0)
    while not engine.state.finished:
        legal = list_legal_actions(engine.state)
        if not legal:
            break
        mask = legal_mask(engine.state)
        assert mask.sum() == len(legal)
        for action in legal:
            assert mask[encode_action(engine.state, action)]
        for index in np.flatnonzero(mask):
            assert decode_action(engine.state, int(index)) in legal
        engine.take_action(decode_action(engine.state, int(rng.choice(np.flatnonzero(mask)))))


def test_decode_returns_none_for_an_empty_ability_slot():
    engine = BattleEngine("S1", 1, None, 20)
    actor = engine.state.characters[engine.state.current_actor_id()]
    if len(actor.abilities) < 3:
        assert decode_action(engine.state, action_index(len(actor.abilities), "e1")) is None


# --- Reward and env -------------------------------------------------------


def test_combat_reward_favours_the_acting_team():
    engine = BattleEngine("S1", 1, None, 20)
    team = engine.state.characters[engine.state.current_actor_id()].team
    other = "enemy" if team == "party" else "party"
    attack = next(
        a for a in engine.legal_actions()
        if a.target_id is not None and engine.state.characters[a.target_id].team == other
    )
    before = len(engine.log)
    engine.take_action(attack)
    records = engine.log[before:]
    assert combat_reward(engine, records, team) > 0
    assert combat_reward(engine, records, other) < 0


def test_env_reports_a_team_flip_when_the_side_to_act_changes():
    env = CombatEnv(["S1"])
    _, info = env.reset(scenario_id="S1", battle_seed=1)
    rng = np.random.default_rng(0)
    for _ in range(40):
        team = env.engine.state.characters[env.engine.state.current_actor_id()].team
        _, _, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(info["action_mask"]))))
        if terminated or truncated:
            assert info["team_flip"] is False
            break
        next_team = env.engine.state.characters[env.engine.state.current_actor_id()].team
        assert info["team_flip"] == (next_team != team)


def test_env_rejects_an_illegal_action():
    env = CombatEnv(["S1"])
    _, info = env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(int(np.flatnonzero(~info["action_mask"])[0]))


def test_env_is_deterministic_for_a_fixed_scenario_and_seed():
    def rollout():
        env = CombatEnv(["S3"])
        obs, info = env.reset(scenario_id="S3", battle_seed=42)
        rng = np.random.default_rng(0)
        trace = [obs.copy()]
        for _ in range(10):
            obs, _, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(info["action_mask"]))))
            trace.append(obs.copy())
            if terminated or truncated:
                break
        return trace

    for a, b in zip(rollout(), rollout(), strict=True):
        assert np.array_equal(a, b)


# --- Network and training -------------------------------------------------


def test_untrained_model_is_uniform_over_legal_moves():
    model = ActorCritic()
    mask = legal_mask(BattleEngine("S1", 1, None, 20).state)
    logits, _ = model(torch.zeros(1, OBS_SIZE), torch.as_tensor(mask).unsqueeze(0))
    probs = torch.softmax(logits, dim=-1)[0].detach().numpy()
    assert probs[~mask].sum() == pytest.approx(0.0, abs=1e-6)
    assert probs[mask] == pytest.approx(np.full(mask.sum(), 1.0 / mask.sum()), abs=1e-6)


def test_greedy_act_picks_the_top_legal_move():
    model = ActorCritic(hidden=32)
    with torch.no_grad():
        model.policy.bias.copy_(torch.arange(ACTION_SIZE, dtype=torch.float32))
    engine = BattleEngine("S1", 1, None, 20)
    mask = legal_mask(engine.state)
    index, _, _ = model.act(encode_observation(engine.state), mask, greedy=True)
    assert index == int(np.flatnonzero(mask).max())


def test_choose_action_is_always_legal():
    model = ActorCritic(hidden=32)
    generator = torch.Generator().manual_seed(0)
    engine = BattleEngine("S4", 2, None, 20)
    for _ in range(30):
        if engine.state.finished or not engine.legal_actions():
            break
        action = choose_action(model, engine, greedy=False, generator=generator)
        assert action in engine.legal_actions()
        engine.take_action(action)


def test_checkpoint_round_trip(tmp_path):
    model = ActorCritic(hidden=32)
    with torch.no_grad():
        model.policy.weight.add_(0.01)
    loaded = load_checkpoint(save_checkpoint(model, tmp_path / "ckpt.pt"))
    assert loaded.hidden == 32
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_tiny_training_run_updates_weights():
    config = TrainConfig(total_steps=64, num_envs=2, rollout=8, epochs=1, minibatches=2, scenario_ids=["S1", "S2"])
    before = ActorCritic().state_dict()["trunk.0.weight"].clone()
    model, stats = train(config, progress=None)
    assert stats["steps"] >= 64
    assert not torch.equal(before, model.state_dict()["trunk.0.weight"])


def test_head_to_head_tallies_every_game():
    result = head_to_head(ActorCritic(hidden=32), model_team="party", scenario_ids=["S1", "S2"], seeds=[1, 2], turn_cap=10)
    assert result["games"] == 4
    assert result["win"] + result["loss"] + result["draw"] == 4
