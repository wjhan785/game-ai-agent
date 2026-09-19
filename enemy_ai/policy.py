"""The enemy AI: network, PPO self-play training, and a check against
random play. Trained on the clean build, not the seeded bugs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from enemy_ai.env import (
    ACTION_SIZE,
    OBS_SIZE,
    SCENARIO_SLOTS,
    CombatEnv,
    decode_action,
    encode_observation,
    legal_mask,
    random_action,
)
from engine.engine import BattleEngine
from engine.models import Action

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "enemy_policy.pt"
MASK_FILL = -1e8


# --- Network ------------------------------------------------------------


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic. The policy head starts at zero, so an
    untrained model picks uniformly among legal moves."""

    def __init__(self, obs_size: int = OBS_SIZE, action_size: int = ACTION_SIZE, hidden: int = 256):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, action_size)
        self.value = nn.Linear(hidden, 1)
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        logits = self.policy(features).masked_fill(~mask, MASK_FILL)
        return logits, self.value(features).squeeze(-1)

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        generator: Optional[torch.Generator] = None,
        greedy: bool = False,
    ):
        """Pick one masked action: sampled, or the top one if `greedy`.
        Returns (index, log-prob, value)."""
        with torch.no_grad():
            logits, value = self(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0),
            )
            probs = torch.softmax(logits, dim=-1)
            if greedy:
                index = int(torch.argmax(probs, dim=-1).item())
            else:
                index = int(torch.multinomial(probs, 1, generator=generator).item())
            return index, float(torch.log(probs[0, index] + 1e-12)), float(value.item())


def choose_action(
    model: ActorCritic,
    engine: BattleEngine,
    *,
    greedy: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Action:
    """The model's move for whoever is acting now."""
    mask = legal_mask(engine.state, engine.legal_actions())
    index, _, _ = model.act(encode_observation(engine.state), mask, generator, greedy=greedy)
    action = decode_action(engine.state, index)
    assert action is not None
    return action


def save_checkpoint(model: ActorCritic, path: str | Path, meta: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_size": model.obs_size,
            "action_size": model.action_size,
            "hidden": model.hidden,
            "meta": meta or {},
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path = DEFAULT_CHECKPOINT) -> ActorCritic:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = ActorCritic(blob["obs_size"], blob["action_size"], blob["hidden"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


# --- Training -----------------------------------------------------------


@dataclass
class TrainConfig:
    total_steps: int = 300_000
    num_envs: int = 8
    rollout: int = 128
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    turn_cap: int = 20
    max_decisions: int = 150
    seed: int = 0
    train_seed_base: int = 10_000
    torch_threads: int = 1  # measured: more threads are slower on tensors this small
    scenario_ids: list[str] = field(default_factory=lambda: list(SCENARIO_SLOTS))


def train(
    config: TrainConfig,
    *,
    defects: Any = None,
    progress: Optional[Callable[[str], None]] = print,
) -> tuple[ActorCritic, dict]:
    """Self-play PPO: one policy plays both teams, trained to win."""
    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)

    envs = [
        CombatEnv(
            config.scenario_ids,
            defects=defects,
            turn_cap=config.turn_cap,
            max_decisions=config.max_decisions,
            seed_base=config.train_seed_base,
        )
        for _ in range(config.num_envs)
    ]
    states = [env.reset(seed=config.seed * 1000 + i) for i, env in enumerate(envs)]
    obs = np.stack([s[0] for s in states])
    masks = np.stack([env.action_masks() for env in envs])

    batch = config.num_envs * config.rollout
    updates = max(1, config.total_steps // batch)
    minibatch_size = max(1, batch // config.minibatches)
    outcomes: list[Optional[str]] = []
    steps_done = 0

    for update in range(updates):
        buf_obs = np.zeros((config.rollout, config.num_envs, OBS_SIZE), dtype=np.float32)
        buf_mask = np.zeros((config.rollout, config.num_envs, ACTION_SIZE), dtype=bool)
        buf_action = np.zeros((config.rollout, config.num_envs), dtype=np.int64)
        buf_logp = np.zeros((config.rollout, config.num_envs), dtype=np.float32)
        buf_value = np.zeros((config.rollout, config.num_envs), dtype=np.float32)
        buf_reward = np.zeros((config.rollout, config.num_envs), dtype=np.float32)
        buf_done = np.zeros((config.rollout, config.num_envs), dtype=np.float32)
        buf_sign = np.ones((config.rollout, config.num_envs), dtype=np.float32)

        for t in range(config.rollout):
            buf_obs[t] = obs
            buf_mask[t] = masks
            for i, env in enumerate(envs):
                index, logp, value = model.act(obs[i], masks[i], generator)
                buf_action[t, i], buf_logp[t, i], buf_value[t, i] = index, logp, value
                next_obs, reward, terminated, truncated, info = env.step(index)
                buf_reward[t, i] = reward
                buf_done[t, i] = 1.0 if (terminated or truncated) else 0.0
                buf_sign[t, i] = -1.0 if info.get("team_flip") else 1.0
                if terminated or truncated:
                    outcomes.append(env.engine.state.outcome)
                    next_obs, _ = env.reset()
                obs[i] = next_obs
                masks[i] = env.action_masks()
            steps_done += config.num_envs

        with torch.no_grad():
            _, last_value = model(
                torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(masks, dtype=torch.bool)
            )
            last_value = last_value.numpy()

        # Negamax GAE: the next state's value counts against us when the
        # other team acts next.
        advantages = np.zeros_like(buf_reward)
        running = np.zeros(config.num_envs, dtype=np.float32)
        for t in reversed(range(config.rollout)):
            next_value = last_value if t == config.rollout - 1 else buf_value[t + 1]
            not_done = 1.0 - buf_done[t]
            sign = buf_sign[t]
            delta = buf_reward[t] + config.gamma * sign * next_value * not_done - buf_value[t]
            running = delta + config.gamma * config.gae_lambda * not_done * sign * running
            advantages[t] = running
        returns = advantages + buf_value

        flat_obs = torch.as_tensor(buf_obs.reshape(-1, OBS_SIZE))
        flat_mask = torch.as_tensor(buf_mask.reshape(-1, ACTION_SIZE))
        flat_action = torch.as_tensor(buf_action.reshape(-1))
        flat_logp = torch.as_tensor(buf_logp.reshape(-1))
        flat_adv = torch.as_tensor(advantages.reshape(-1))
        flat_ret = torch.as_tensor(returns.reshape(-1))

        indices = np.arange(batch)
        for _ in range(config.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch, minibatch_size):
                mb = indices[start : start + minibatch_size]
                logits, values = model(flat_obs[mb], flat_mask[mb])
                log_probs = torch.log_softmax(logits, dim=-1)
                chosen = log_probs.gather(1, flat_action[mb].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(chosen - flat_logp[mb])

                adv = flat_adv[mb]
                if adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = -torch.min(
                    ratio * adv, torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * adv
                ).mean()
                value_loss = ((values - flat_ret[mb]) ** 2).mean()
                entropy = -(log_probs.exp() * log_probs).masked_fill(~flat_mask[mb], 0.0).sum(-1).mean()

                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        if progress is not None and (update % 10 == 0 or update == updates - 1):
            recent = outcomes[-200:]
            decisive = sum(1 for o in recent if o in ("party_win", "enemy_win"))
            progress(
                f"[update {update + 1}/{updates}] steps={steps_done} episodes={len(outcomes)} "
                f"decisive={100 * decisive / max(len(recent), 1):.0f}% (last {len(recent)})"
            )

    stats = {
        "steps": steps_done,
        "updates": updates,
        "episodes": len(outcomes),
        "outcomes": {o: outcomes.count(o) for o in sorted(set(map(str, outcomes)))},
    }
    return model, stats


# --- Strength check -----------------------------------------------------


def head_to_head(
    model: ActorCritic,
    *,
    model_team: str,
    scenario_ids: list[str],
    seeds: list[int],
    defects: Any = None,
    turn_cap: int = 20,
    greedy: bool = False,
    sample_seed: int = 0,
) -> dict:
    """The model plays `model_team`, uniform-random plays the other side."""
    generator = torch.Generator().manual_seed(sample_seed)
    rng = np.random.default_rng(sample_seed)
    tally = {"win": 0, "loss": 0, "draw": 0}
    for scenario_id in scenario_ids:
        for seed in seeds:
            engine = BattleEngine(scenario_id, seed, defects, turn_cap)
            while not engine.state.finished:
                legal = engine.legal_actions()
                if not legal:
                    break
                actor = engine.state.characters[engine.state.current_actor_id()]
                if actor.team == model_team:
                    action = choose_action(model, engine, greedy=greedy, generator=generator)
                else:
                    action = random_action(engine.state, rng, legal)
                engine.take_action(action)
            winner = {"party_win": "party", "enemy_win": "enemy"}.get(engine.state.outcome or "")
            if winner is None:
                tally["draw"] += 1
            elif winner == model_team:
                tally["win"] += 1
            else:
                tally["loss"] += 1
    games = sum(tally.values())
    return {**tally, "games": games, "win_rate": tally["win"] / games if games else 0.0}
