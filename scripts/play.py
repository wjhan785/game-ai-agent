"""Play one battle by hand, turn by turn, through the same surface the LLM
agent uses -- agent.tools.ToolDispatcher's get_state / list_legal_actions /
take_action -- so what you see here (the state, the legal options, the
declared-vs-observed diff after each action) is exactly what the agent
sees, not a separate debug view.

Usage:
    conda run -n learning python scripts/play.py S1
    conda run -n learning python scripts/play.py S5 --defects B08
    conda run -n learning python scripts/play.py S2 --full-build --seed 7
"""
from __future__ import annotations

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario_id", choices=sorted(SCENARIOS))
    parser.add_argument("--seed", type=int, default=random.randint(0, 1_000_000))
    parser.add_argument("--turn-cap", type=int, default=20)
    parser.add_argument("--defects", nargs="*", default=[], choices=sorted(ALL_DEFECT_IDS), metavar="B01..B11")
    parser.add_argument("--full-build", action="store_true", help="enable every seeded defect at once")
    args = parser.parse_args()

    defects = build_defects(args.defects, args.full_build)
    engine = BattleEngine(args.scenario_id, args.seed, defects, args.turn_cap)
    dispatcher = ToolDispatcher(engine)

    print(f"Scenario {args.scenario_id}, seed {args.seed}, turn cap {args.turn_cap}")
    print(f"Defects enabled: {', '.join(defects.enabled()) or 'none (clean build)'}")

    while not engine.state.finished:
        print_state(dispatcher)
        choice = choose_action(dispatcher)
        if choice is None:
            break
        print(json.dumps(dispatcher.take_action(**choice), indent=2))

    print(f"\nBattle finished: {engine.state.outcome}")


if __name__ == "__main__":
    main()
