"""Run an LLM campaign against the full build (every defect on), then report.

    python -m eval.pilot --run-name pilot --episodes 25 --max-spend 0.75
    python -m eval.pilot --run-name greedy --method greedy_llm --episodes 25
"""

import argparse
import json

from agent.llm import Cassette, SpendTracker
from agent.outer_loop import SCENARIO_NOTES, SessionConfig, run_session
from baselines.greedy_llm import greedy_llm_config
from engine.defects import ALL_DEFECT_IDS, DefectFlags
from eval.bug_report import generate_report


def full_build() -> DefectFlags:
    return DefectFlags(**{field: True for field in ALL_DEFECT_IDS.values()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LLM campaign (full agent or Greedy-LLM) against the buggy build, then report.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--method", choices=["full_agent", "greedy_llm"], default="full_agent")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--turn-cap", type=int, default=20)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_NOTES))
    parser.add_argument("--no-gating", action="store_true")
    parser.add_argument("--mode", choices=["live", "record", "replay"], default="record")
    parser.add_argument("--max-spend", type=float, default=0.75, help="cap on THIS run's spend, USD")
    parser.add_argument("--allow-peak", action="store_true")
    parser.add_argument("--runs-root", default="logs/runs")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue an existing run up to --episodes")
    args = parser.parse_args()

    cassette = Cassette() if args.mode in ("record", "replay") else None
    spend = SpendTracker(max_session_spend=args.max_spend) if args.mode != "replay" else None
    make = greedy_llm_config if args.method == "greedy_llm" else SessionConfig
    config = make(
        run_name=args.run_name,
        episodes=args.episodes,
        turn_cap=args.turn_cap,
        gating=not args.no_gating,
        allowed_scenarios=args.scenarios,
    )
    summary = run_session(
        config, runs_root=args.runs_root, defects=full_build(), mode=args.mode,
        cassette=cassette, spend=spend, allow_peak=args.allow_peak,
        progress=lambda line: print(line, flush=True), resume=args.resume,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2), flush=True)
    if spend is not None:
        print(f"this run: ${spend.session_usd:.4f}   cumulative (results/spend.json): ${spend.total_usd:.4f}", flush=True)
    if not args.no_report:
        print(f"wrote {generate_report(f'{args.runs_root}/{args.run_name}')}", flush=True)


if __name__ == "__main__":
    main()
