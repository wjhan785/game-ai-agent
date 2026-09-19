"""Grades a finished campaign against the oracle and writes the bug report.

Per defect: triggered (leave-one-out), active steps, and invariant detection.
Each agent flag is triaged against defects active in the steps it saw:
  strong            wording fits and it names an affected character
  weak              wording fits, no character overlap
  unmatched         a defect was active nearby but the wording fits none
  no defect nearby  likely a false positive
This is a heuristic: it says what a flag is plausibly about, not that it's right.
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent.ledger import Ledger, invariant_kind
from engine.defects import ALL_DEFECT_IDS, INVARIANT_INVISIBLE, INVARIANT_VISIBLE, DefectFlags
from engine.episode_log import read_episode_jsonl
from engine.models import BattleState
from engine.resolution import TurnRecord
from oracle.triggers import StepActivity, TriggerResult, leave_one_out_triggers, stepwise_activity
from replay.generate import generate_replay_html

SHORT_NAMES = {
    "B01": "Shield absorbs Poison/Burn ticks",
    "B02": "Burn stacks multiply instead of add",
    "B03": "Chill drives energy negative",
    "B04": "Poison computed off Weaken-adjusted HP",
    "B05": "Stun duplicates instead of refreshing",
    "B06": "Cooldown drains twice per turn",
    "B07": "Regen survives death",
    "B08": "Enemy has no Basic Attack fallback",
    "B09": "Elemental multiplier applied twice",
    "B10": "Zero-pool Shield never removed",
    "B11": "Turn-order ties broken by character id, not declared order",
}

INVARIANT_SIGNATURES: dict[str, re.Pattern] = {
    "B03": re.compile(r"energy went negative"),
    "B05": re.compile(r"simultaneous stun entries"),
    "B06": re.compile(r"cooldown remaining"),
    "B07": re.compile(r"dead but still carrying regen"),
    "B08": re.compile(r"no-progress"),
    "B10": re.compile(r"zero-magnitude shield"),
}

# Every group must contribute at least one term to the flag's text.
FLAG_KEYWORDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "B01": (("shield",), ("burn", "poison", "tick", "dot", "damage-over-time", "damage over time")),
    "B02": (("burn",), ("two", "both", "multiple", "stack", "sum", "+", "additive", "multipl", "sources", "instances", "combined")),
    "B03": (("energy",), ("negative", "below 0", "below zero", "under 0", "less than 0", "chill")),
    "B04": (("poison",), ("weaken", "max hp", "max_hp", "percent", "%")),
    "B05": (("stun",), ("duplicate", "two stun", "stack", "second", "extend", "longer", "refresh", "multiple")),
    "B06": (("cooldown",),),
    "B07": (("regen",), ("dead", "death", "died", "reviv")),
    "B08": (("basic attack", "no legal", "stuck", "no action", "no-progress", "skipped", "skips"),),
    "B09": (("multiplier", "elemental", "advantage", "tooltip", "1.5", "2.25"),),
    "B10": (("shield",), ("zero", ":0/", " 0/", "empty", "depleted", "drained", "exhausted", "still present", "not removed", "remain")),
    "B11": (("turn order", "tie", "tied", "wrong actor", "acted out of order", "acted early", "skipped its turn", "went before", "went out of turn"),),
}

OBSERVATION_WINDOW_STEPS = 10


@dataclass
class EpisodeGrade:
    episode_id: int
    path: Path
    scenario: str
    seed: int
    outcome: Optional[str]
    reproduces: bool
    triggers: dict[str, TriggerResult]
    activity: StepActivity
    invariant_hits: dict[str, list[int]]
    violation_kinds: set[str]
    unmapped_violation_kinds: list[str]
    names: dict[str, str]


def build_from_header(header: dict) -> DefectFlags:
    return DefectFlags(**{name: True for name in header.get("defects_enabled", [])})


def grade_episode(episode_id: int, path: Path) -> EpisodeGrade:
    header, turns, footer = read_episode_jsonl(path)
    records = [TurnRecord.model_validate(t) for t in turns]
    initial_state = BattleState.model_validate(header["initial_state"])
    build = build_from_header(header)
    report = leave_one_out_triggers(header["scenario_id"], header["seed"], header["turn_cap"], records, build)
    activity = stepwise_activity(initial_state, records, build)

    hits: dict[str, list[int]] = {did: [] for did in INVARIANT_SIGNATURES}
    kinds: set[str] = set()
    unmapped: set[str] = set()
    for rec in records:
        for v in rec.invariant_violations:
            kinds.add(invariant_kind(v))
            mapped = [did for did, pat in INVARIANT_SIGNATURES.items() if pat.search(v)]
            for did in mapped:
                hits[did].append(rec.step_count)
            if not mapped:
                unmapped.add(invariant_kind(v))

    return EpisodeGrade(
        episode_id=episode_id,
        path=path,
        scenario=header["scenario_id"],
        seed=header["seed"],
        outcome=footer.get("outcome"),
        reproduces=report.reproduces,
        triggers=report.triggers,
        activity=activity,
        invariant_hits={k: v for k, v in hits.items() if v},
        violation_kinds=kinds,
        unmapped_violation_kinds=sorted(unmapped),
        names={cid: c.name for cid, c in initial_state.characters.items()},
    )


def _keywords_match(defect_id: str, text: str) -> bool:
    return all(any(term in text for term in group) for group in FLAG_KEYWORDS[defect_id])


def classify_flag(flag: dict, grade: EpisodeGrade) -> dict:
    # Description only: audit evidence would match nearly any defect.
    step = flag["turn"]
    text = flag["description"].lower()
    mentioned = {cid for cid in grade.names if re.search(rf"\b{cid}\b", text)}
    mentioned |= {cid for cid, name in grade.names.items() if name.lower() in text}

    active_nearby: list[str] = []
    matches: list[dict] = []
    for did, steps in grade.activity.steps.items():
        window = [k for k in steps if step - OBSERVATION_WINDOW_STEPS <= k <= step]
        if not window:
            continue
        active_nearby.append(did)
        if not _keywords_match(did, text):
            continue
        affected = set().union(*(set(grade.activity.entities[did].get(k, [])) for k in window))
        matches.append({"defect": did, "strength": "strong" if mentioned & affected else "weak", "steps": window})

    matches.sort(key=lambda m: (m["strength"] != "strong", m["defect"]))
    if matches:
        verdict = "strong" if matches[0]["strength"] == "strong" else "weak"
    elif active_nearby:
        verdict = "unmatched"
    else:
        verdict = "no defect nearby"
    return {"verdict": verdict, "matches": matches, "active_nearby": sorted(active_nearby)}


def _rel(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix()


def generate_report(run_dir: str | Path, *, results_root: str | Path = "results", replay_root: str | Path = "replay/out") -> Path:
    run_dir = Path(run_dir)
    run_name = run_dir.name
    out_dir = Path(results_root) / run_name
    replay_dir = Path(replay_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(run_dir / "ledger.db")
    episodes = ledger.episodes()
    agent_flags = ledger.flags(source="agent")
    hypotheses = ledger.hypotheses()
    ledger.close()
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8")) if (run_dir / "session.json").exists() else {}

    grades: dict[int, EpisodeGrade] = {}
    replay_links: dict[int, Path] = {}
    for e in episodes:
        path = Path(e["log_path"])
        if not path.exists():
            continue
        grades[e["id"]] = grade_episode(e["id"], path)
        replay_links[e["id"]] = generate_replay_html(path, replay_dir / f"{path.stem}.html")

    classified = [(f, classify_flag(f, grades[f["episode"]])) for f in agent_flags if f["episode"] in grades]

    per_defect = {}
    for did in ALL_DEFECT_IDS:
        triggered_eps = [g.episode_id for g in grades.values() if g.triggers.get(did) and g.triggers[did].triggered]
        per_defect[did] = {
            "triggered": triggered_eps,
            "active_steps": sum(len(g.activity.steps.get(did, [])) for g in grades.values()),
            "invariant_eps": [g.episode_id for g in grades.values() if did in g.invariant_hits],
            "strong_eps": sorted({f["episode"] for f, c in classified if any(m["defect"] == did and m["strength"] == "strong" for m in c["matches"])}),
            "weak_eps": sorted({f["episode"] for f, c in classified if any(m["defect"] == did and m["strength"] == "weak" for m in c["matches"])}),
        }

    lines: list[str] = []
    w = lines.append
    w(f"# Bug report: `{run_name}`")
    w("")
    w(f"Generated by `eval/bug_report.py` from `{run_dir.as_posix()}/`. Grading uses the differential oracle "
      "(`oracle/triggers.py`) after the campaign finished; the agent never saw any of it.")
    w("")

    triggered = [d for d in ALL_DEFECT_IDS if per_defect[d]["triggered"]]
    inv_detected = [d for d in ALL_DEFECT_IDS if per_defect[d]["invariant_eps"]]
    agent_strong = [d for d in ALL_DEFECT_IDS if per_defect[d]["strong_eps"]]
    agent_any = [d for d in ALL_DEFECT_IDS if per_defect[d]["strong_eps"] or per_defect[d]["weak_eps"]]
    invisible_hit = [d for d in agent_strong if d in INVARIANT_INVISIBLE]
    w("## Headline")
    w("")
    w(f"- **{len(triggered)} of {len(ALL_DEFECT_IDS)}** seeded defects were triggered at least once ({', '.join(triggered) or 'none'}).")
    w(f"- The invariant checker detected **{len(inv_detected)}** ({', '.join(inv_detected) or 'none'}).")
    w(f"- The agent's flags strongly match **{len(agent_strong)}** ({', '.join(agent_strong) or 'none'}); "
      f"strong or weak: {len(agent_any)} ({', '.join(agent_any) or 'none'}).")
    w(f"- Of the {len(INVARIANT_INVISIBLE)} defects no invariant can see ({', '.join(sorted(INVARIANT_INVISIBLE))}), the agent strongly matched "
      f"**{len(invisible_hit)}** ({', '.join(invisible_hit) or 'none'}).")
    verdicts = Counter(c["verdict"] for _, c in classified)
    w(f"- Agent flags: {len(classified)} total -- strong {verdicts['strong']}, weak {verdicts['weak']}, "
      f"unmatched {verdicts['unmatched']}, no defect nearby (likely false positive) {verdicts['no defect nearby']}.")
    w("")

    w("## Run summary")
    w("")
    scen = Counter(e["scenario"] for e in episodes)
    w(f"- Build under test: {', '.join(sorted(_enabled_ids(grades)))} enabled")
    w(f"- Episodes: {len(episodes)} ({', '.join(f'{k}x{v}' for k, v in sorted(scen.items()))}); actions: {sum(e['actions_used'] for e in episodes)}")
    if session:
        w(f"- Method: {session.get('method', 'full_agent')}")
        d = session.get("decisions", {})
        total = sum(d.values()) or 1
        w("- Decisions: " + ", ".join(f"{k} {v} ({100 * v / total:.0f}%)" for k, v in d.items()))
        w(f"- LLM calls: planner {session.get('planner_calls')}, tactical {session.get('inner_calls')}; "
          f"failed after retries: {session.get('failed_calls')}; planner fallbacks: {session.get('planner_fallbacks')}")
        rate = session.get("cache_hit_rate")
        w(f"- Model spend: ${session.get('cost_usd', 0):.4f}; tokens in {session.get('tokens_in')}, out {session.get('tokens_out')}; "
          f"cache hit rate {'n/a' if rate is None else f'{100 * rate:.1f}%'}")
        if session.get("aborted_reason"):
            w(f"- **Aborted:** {session['aborted_reason']}")
        if session.get("final_summary"):
            w(f"- Planner's closing summary: _{session['final_summary']}_")
    bad = [g.episode_id for g in grades.values() if not g.reproduces or g.activity.inconsistent_steps]
    w(f"- Oracle self-check: {'all episodes replay exactly' if not bad else f'episodes {bad} did not replay exactly -- grading for them is unreliable'}")
    w("")

    w("## Defects: triggered vs detected")
    w("")
    w("| ID | Defect | Invariant-visible | Episodes triggered | Steps active | Invariant detected (episodes) | Agent flag match: strong / weak (episodes) |")
    w("|---|---|---|---|---|---|---|")
    for did in ALL_DEFECT_IDS:
        p = per_defect[did]
        w(f"| {did} | {SHORT_NAMES[did]} | {'yes' if did in INVARIANT_VISIBLE else '**no**'} | "
          f"{len(p['triggered'])} | {p['active_steps']} | {len(p['invariant_eps'])} | "
          f"{len(p['strong_eps'])} / {len(p['weak_eps'])} |")
    w("")

    w("## Agent flags")
    w("")
    if not classified:
        w("_The agent raised no flags._")
    for f, c in classified:
        w(f"### Flag {f['id']} -- episode {f['episode']}, step {f['turn']}, severity {f['severity']}")
        w("")
        w(f"> {f['description']}")
        w("")
        if f.get("evidence"):
            w(f"Evidence: {f['evidence']}")
            w("")
        if c["matches"]:
            desc = "; ".join(
                f"{m['defect']} ({SHORT_NAMES[m['defect']]}) -- {m['strength']}, active at steps {m['steps']}" for m in c["matches"]
            )
            w(f"Oracle triage: **{c['verdict']}** -- {desc}")
        elif c["active_nearby"]:
            w(f"Oracle triage: **unmatched** -- active in the preceding {OBSERVATION_WINDOW_STEPS} steps: "
              f"{', '.join(c['active_nearby'])}, but the flag's wording fits none of them. Needs manual adjudication.")
        else:
            w(f"Oracle triage: **no defect nearby** -- nothing was active in the preceding {OBSERVATION_WINDOW_STEPS} steps; likely a false positive.")
        links = []
        if f.get("trace_path"):
            links.append(f"repro `{Path(f['trace_path']).as_posix()}`")
        if f["episode"] in replay_links:
            links.append(f"[replay]({_rel(replay_links[f['episode']], out_dir)})")
        if links:
            w("")
            w(" · ".join(links))
        w("")

    w("## Hypotheses (from the agent's ledger)")
    w("")
    if not hypotheses:
        w("_None recorded._")
    else:
        w("| # | Status | Hypothesis | Notes |")
        w("|---|---|---|---|")
        for h in hypotheses:
            notes = (h["notes"] or "").replace("\n", " ").replace("|", "/")
            w(f"| {h['id']} | {h['status']} | {h['text'].replace('|', '/')} | {notes} |")
    w("")

    w("## Invariant violations observed")
    w("")
    kinds: Counter = Counter()
    for g in grades.values():
        kinds.update(g.violation_kinds)
    if not kinds:
        w("_None._")
    else:
        w("| Kind | Episodes | Produced by |")
        w("|---|---|---|")
        for kind, n in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0])):
            mapped = [did for did, pat in INVARIANT_SIGNATURES.items() if pat.search(kind)]
            w(f"| {kind} | {n} | {', '.join(mapped) or 'unmapped'} |")
    w("")

    w("## Episodes")
    w("")
    w("| # | Scenario | Seed | Goal | Outcome | Actions | Defects triggered | Replay |")
    w("|---|---|---|---|---|---|---|---|")
    for e in episodes:
        g = grades.get(e["id"])
        trig = ", ".join(d for d, t in (g.triggers.items() if g else []) if t.triggered)
        goal = (e["plan"].get("goal") or "(free exploration)").replace("\n", " ").replace("|", "/")
        if len(goal) > 140:
            goal = goal[:137] + "..."
        link = f"[html]({_rel(replay_links[e['id']], out_dir)})" if e["id"] in replay_links else ""
        w(f"| {e['id']} | {e['scenario']} | {e['seed']} | {goal} | {e['outcome']} | {e['actions_used']} | {trig} | {link} |")
    w("")

    w("## How to read this")
    w("")
    w("- The campaign ran against one build with every seeded defect enabled at once, the way QA meets a real build: "
      "nobody tells the tester which bugs are in it.")
    w("- **Triggered** is decided by leave-one-out replay: the recorded actions replayed with every defect except one; "
      "a divergence means that defect changed this trace's outcome. **Steps active** re-runs each step from its logged "
      "state with and without the defect. Both are identical procedures for any exploration method.")
    w(f"- **Detected by the invariant checker** means a violation of the kind that defect produces. {len(INVARIANT_VISIBLE)} defects can be "
      f"seen that way; {len(INVARIANT_INVISIBLE)} produce wrong numbers but legal states, and only a reasoning agent can report those.")
    w("- Flag matching is a heuristic triage (active nearby + wording + named character), not adjudication. Unmatched "
      "flags and weak matches need a human verdict (eval/adjudicate.py, week 5).")
    w("- This is one campaign with no baselines yet; it describes what this agent did, not how it compares to "
      "random or scripted exploration.")
    w("")

    report_path = out_dir / "bug_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "grades.json").write_text(
        json.dumps(
            {
                "per_defect": per_defect,
                "flags": [{"flag": f, "triage": c} for f, c in classified],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def _enabled_ids(grades: dict[int, EpisodeGrade]) -> set[str]:
    ids: set[str] = set()
    for g in grades.values():
        ids |= set(g.triggers)
    return ids


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write the bug report for a finished campaign run.")
    parser.add_argument("run_dir", help="e.g. logs/runs/pilot")
    args = parser.parse_args()
    print(f"wrote {generate_report(args.run_dir)}")


if __name__ == "__main__":
    main()
