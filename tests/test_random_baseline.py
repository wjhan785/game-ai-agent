"""The random baseline: budget, determinism, and that the bug report grades it."""

import json

import numpy as np

from agent.ledger import Ledger
from baselines.random_explorer import RandomConfig, run_random_campaign
from engine.engine import BattleEngine
from engine.episode_log import read_episode_jsonl
from enemy_ai.env import random_action
from eval.bug_report import generate_report
from eval.pilot import full_build


def _run(tmp_path, name="rand", **kw):
    config = RandomConfig(run_name=name, **{"actions": 120, **kw})
    return run_random_campaign(config, runs_root=tmp_path, defects=full_build(), progress=lambda _: None)


def test_random_action_is_legal():
    engine = BattleEngine("S1", 3)
    rng = np.random.default_rng(0)
    for _ in range(30):
        if engine.state.finished:
            break
        legal = engine.legal_actions()
        action = random_action(engine.state, rng, legal)
        assert action in legal
        engine.take_action(action)


def test_budget_is_exact(tmp_path):
    summary = _run(tmp_path)
    assert summary["actions"] == 120
    ledger = Ledger(tmp_path / "rand" / "ledger.db")
    assert sum(e["actions_used"] for e in ledger.episodes()) == 120
    ledger.close()


def test_same_seed_same_campaign(tmp_path):
    a = _run(tmp_path, "a", seed=4)
    b = _run(tmp_path, "b", seed=4)
    for key in ("episodes_completed", "actions", "invariant_flags", "coverage_items"):
        assert a[key] == b[key]
    for n in range(1, a["episodes_completed"] + 1):
        _, turns_a, _ = read_episode_jsonl(tmp_path / "a" / "episodes" / f"ep_{n:04d}.jsonl")
        _, turns_b, _ = read_episode_jsonl(tmp_path / "b" / "episodes" / f"ep_{n:04d}.jsonl")
        assert [t["action"] for t in turns_a] == [t["action"] for t in turns_b]


def test_cycles_scenarios(tmp_path):
    _run(tmp_path, scenarios=["S1", "S5"], actions=200)
    ledger = Ledger(tmp_path / "rand" / "ledger.db")
    scenarios = [e["scenario"] for e in ledger.episodes()]
    ledger.close()
    assert scenarios[:2] == ["S1", "S5"]


def test_bug_report_grades_it(tmp_path):
    _run(tmp_path, actions=150)
    path = generate_report(tmp_path / "rand", results_root=tmp_path / "results", replay_root=tmp_path / "replay")
    text = path.read_text(encoding="utf-8")
    assert "Method: random" in text
    assert "random 150 (100%)" in text
    data = json.loads((tmp_path / "rand" / "session.json").read_text(encoding="utf-8"))
    assert data["method"] == "random" and data["cost_usd"] == 0.0
