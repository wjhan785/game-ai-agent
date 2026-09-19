"""Run the random baseline against the full build, then write the bug report.

    python -m eval.random_baseline --run-name random --actions 844
"""

import argparse
import json

from baselines.random_explorer import RandomConfig, run_random_campaign
from enemy_ai.env import SCENARIO_SLOTS
from eval.bug_report import generate_report
from eval.pilot import full_build


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the random baseline against the buggy build, then report.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--actions", type=int, default=800, help="total action budget")
    parser.add_argument("--turn-cap", type=int, default=20)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_SLOTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-root", default="logs/runs")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    config = RandomConfig(
        run_name=args.run_name, actions=args.actions, turn_cap=args.turn_cap,
        scenarios=args.scenarios, seed=args.seed,
    )
    summary = run_random_campaign(
        config, runs_root=args.runs_root, defects=full_build(), progress=lambda line: print(line, flush=True)
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2), flush=True)
    if not args.no_report:
        print(f"wrote {generate_report(f'{args.runs_root}/{args.run_name}')}", flush=True)


if __name__ == "__main__":
    main()
