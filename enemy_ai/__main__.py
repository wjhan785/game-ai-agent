"""Train or check the enemy AI.

    python -m enemy_ai train                  # writes enemy_ai/enemy_policy.pt
    python -m enemy_ai check                  # vs uniform-random, both sides
"""

import argparse
import json
from pathlib import Path

from enemy_ai.env import SCENARIO_SLOTS
from enemy_ai.policy import DEFAULT_CHECKPOINT, TrainConfig, head_to_head, load_checkpoint, save_checkpoint, train


def _train(args: argparse.Namespace) -> None:
    config = TrainConfig(total_steps=args.steps, seed=args.seed, scenario_ids=args.scenarios)
    model, stats = train(config)
    path = save_checkpoint(model, args.out, {"steps": stats["steps"], "seed": args.seed, "build": "clean"})
    print(json.dumps(stats, indent=2))
    print(f"wrote {path}")


def _check(args: argparse.Namespace) -> None:
    model = load_checkpoint(args.checkpoint)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    for team in ("party", "enemy"):
        r = head_to_head(model, model_team=team, scenario_ids=args.scenarios, seeds=seeds, greedy=args.greedy)
        print(f"model as {team:<5}: win {r['win']:>3}  loss {r['loss']:>3}  draw {r['draw']:>3}  "
              f"win_rate {100 * r['win_rate']:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or check the enemy AI.")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="self-play training on the clean build")
    t.add_argument("--steps", type=int, default=300_000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--scenarios", nargs="+", default=list(SCENARIO_SLOTS))
    t.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    t.set_defaults(func=_train)

    c = sub.add_parser("check", help="win rate vs uniform-random, from both sides")
    c.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    c.add_argument("--seeds", type=int, default=20)
    c.add_argument("--seed-start", type=int, default=1)
    c.add_argument("--greedy", action="store_true", help="top move instead of sampling")
    c.add_argument("--scenarios", nargs="+", default=list(SCENARIO_SLOTS))
    c.set_defaults(func=_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
