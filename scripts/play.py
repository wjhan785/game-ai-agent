"""Play a battle by hand through the agent's own tool surface.

Usage:
    conda run -n learning python scripts/play.py S1
    conda run -n learning python scripts/play.py S5 --defects B08
    conda run -n learning python scripts/play.py S2 --full-build --seed 7
    conda run -n learning python scripts/play.py S1 --manual-enemy

You play the party and the enemy AI plays the enemy. --enemy-sample makes
it less predictable; --manual-enemy lets you play both teams.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.tools import ToolDispatcher  # noqa: E402
from engine.defects import ALL_DEFECT_IDS, DefectFlags  # noqa: E402
from engine.engine import BattleEngine  # noqa: E402
from engine.scenarios import SCENARIOS  # noqa: E402


def build_defects(defect_ids: list[str], full_build: bool) -> DefectFlags:
    if full_build:
        return DefectFlags(**{field: True for field in ALL_DEFECT_IDS.values()})
    return DefectFlags(**{ALL_DEFECT_IDS[d]: True for d in defect_ids})


def print_state(dispatcher: ToolDispatcher) -> None:
    state = dispatcher.get_state()
    print(f"\n-- round {state['round_number']}, step {state['step_count']} --")
    for c in state["characters"]:
        if not c["alive"]:
            print(f"  {c['id']} {c['name']} ({c['team']}): dead")
            continue
        statuses = ", ".join(f"{s['type']}:{s['magnitude']}/{s['turns_remaining']}t" for s in c["statuses"]) or "-"
        cds = ", ".join(f"{k}:{v}" for k, v in c["cooldowns"].items()) or "-"
        print(
            f"  {c['id']} {c['name']} ({c['team']}, {c['element']}): "
            f"HP {c['hp']:.1f}/{c['max_hp']:.1f}  EN {c['energy']:.1f}/{c['max_energy']:.1f}  "
            f"SPD {c['speed']:.0f} (AV {c['action_value']:.1f})  "
            f"statuses: {statuses}  cooldowns: {cds}"
        )
    forecast = ", ".join(f"{f['id']}@{f['action_value']:.0f}" for f in state["forecast"])
    print(f"  next up: {forecast}")


def choose_action(dispatcher: ToolDispatcher) -> dict | None:
    options = dispatcher.list_legal_actions()
    actor_id = options["actor_id"]
    if actor_id is None:
        return None
    print(f"\n{actor_id}'s turn. Legal actions:")
    for i, opt in enumerate(options["options"]):
        target = f" -> {opt.get('target_name', opt['target_id'])}" if opt.get("target_id") else ""
        dmg = f", expected dmg {opt['declared_expected_damage']}" if "declared_expected_damage" in opt else ""
        applies = f", applies {opt['applies']['status_type']}" if "applies" in opt else ""
        print(
            f"  [{i}] {opt['ability_name']}{target}  "
            f"(cost {opt['energy_cost']}, cd {opt['cooldown']}, {opt['element']}{dmg}{applies})"
        )
    while True:
        raw = input("Pick a number ('s' = full state, 'q' = quit): ").strip().lower()
        if raw == "q":
            raise SystemExit(0)
        if raw == "s":
            print(json.dumps(dispatcher.get_state(), indent=2))
            continue
        if raw.isdigit() and 0 <= int(raw) < len(options["options"]):
            chosen = options["options"][int(raw)]
            return {
                "actor_id": actor_id,
                "ability_name": chosen["ability_name"],
                "target_id": chosen["target_id"],
                "reasoning": "human",
            }
        print("Not a valid choice, try again.")


def load_enemy_ai(path: str):
    """(model, generator), or (None, None) with a note if it can't load."""
    try:
        import torch

        from enemy_ai.policy import load_checkpoint
    except ImportError:
        print("Enemy AI needs torch (pip install -e \".[rl]\"); you'll control both teams.")
        return None, None
    if not Path(path).exists():
        print(f"No enemy AI at {path} (train one: python -m enemy_ai train); you'll control both teams.")
        return None, None
    return load_checkpoint(path), torch.Generator().manual_seed(0)


def enemy_ai_action(model, engine: BattleEngine, greedy: bool, generator) -> dict:
    """The enemy AI's move, as take_action kwargs."""
    from enemy_ai.policy import choose_action

    action = choose_action(model, engine, greedy=greedy, generator=generator)
    return {
        "actor_id": action.actor_id,
        "ability_name": action.ability_name,
        "target_id": action.target_id,
        "reasoning": "enemy_ai",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario_id", choices=sorted(SCENARIOS))
    parser.add_argument("--seed", type=int, default=random.randint(0, 1_000_000))
    parser.add_argument("--turn-cap", type=int, default=20)
    parser.add_argument("--defects", nargs="*", default=[], choices=sorted(ALL_DEFECT_IDS), metavar="B01..B11")
    parser.add_argument("--full-build", action="store_true", help="enable every seeded defect at once")
    parser.add_argument("--manual-enemy", action="store_true", help="control the enemy team yourself")
    parser.add_argument("--enemy-ai", default=str(ROOT / "enemy_ai" / "enemy_policy.pt"), metavar="CHECKPOINT",
                        help="enemy AI checkpoint (default: enemy_ai/enemy_policy.pt)")
    parser.add_argument("--enemy-sample", action="store_true", help="enemy AI samples moves instead of always the top one")
    args = parser.parse_args()

    model = generator = None
    if not args.manual_enemy:
        model, generator = load_enemy_ai(args.enemy_ai)

    defects = build_defects(args.defects, args.full_build)
    engine = BattleEngine(args.scenario_id, args.seed, defects, args.turn_cap)
    dispatcher = ToolDispatcher(engine)

    print(f"Scenario {args.scenario_id}, seed {args.seed}, turn cap {args.turn_cap}")
    print(f"Defects enabled: {', '.join(defects.enabled()) or 'none (clean build)'}")
    if model is not None:
        print(f"Enemy team: enemy AI ({'sampling' if args.enemy_sample else 'top move'})")

    while not engine.state.finished:
        print_state(dispatcher)
        actor_id = engine.state.current_actor_id()
        if model is not None and engine.state.characters[actor_id].team == "enemy":
            choice = enemy_ai_action(model, engine, not args.enemy_sample, generator)
            target = f" -> {choice['target_id']}" if choice["target_id"] else ""
            print(f"\n{actor_id} (enemy AI) uses {choice['ability_name']}{target}")
        else:
            choice = choose_action(dispatcher)
            if choice is None:
                break
        print(json.dumps(dispatcher.take_action(**choice), indent=2))

    print(f"\nBattle finished: {engine.state.outcome}")


if __name__ == "__main__":
    main()
