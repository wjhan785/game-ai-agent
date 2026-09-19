"""eval/bug_report.py: flag triage on a hand-built grade, and a full report
generated from a tiny offline campaign against the full build."""

import json
from pathlib import Path

from agent.ledger import Ledger
from agent.llm import Cassette
from agent.outer_loop import SessionConfig, run_session
from eval.bug_report import EpisodeGrade, classify_flag, generate_report, grade_episode
from eval.pilot import full_build
from oracle.triggers import StepActivity


def _grade() -> EpisodeGrade:
    return EpisodeGrade(
        episode_id=1, path=Path("x"), scenario="S1", seed=1, outcome=None, reproduces=True, triggers={},
        activity=StepActivity(
            steps={"B09": [10], "B06": [3]},
            entities={"B09": {10: ["e1"]}, "B06": {3: ["p2"]}},
            inconsistent_steps=[],
        ),
        invariant_hits={}, violation_kinds=set(), unmapped_violation_kinds=[],
        names={"p1": "Warden", "p2": "Blade", "p3": "Bolt", "e1": "Cinderfang", "e2": "Ashclaw", "e3": "Grunt"},
    )


def _flag(turn, description, evidence=""):
    return {"id": 1, "episode": 1, "turn": turn, "description": description, "evidence": evidence}


def test_strong_match_needs_activity_wording_and_the_affected_character():
    c = classify_flag(_flag(12, "Cinderfang took more than the tooltip damage", "expected 12 (8 x 1.5), saw 18"), _grade())
    assert c["verdict"] == "strong"
    assert c["matches"][0]["defect"] == "B09"


def test_weak_match_when_the_named_character_was_not_affected():
    c = classify_flag(_flag(12, "p3 took more than the tooltip damage"), _grade())
    assert c["verdict"] == "weak"


def test_unmatched_when_wording_fits_no_active_defect():
    c = classify_flag(_flag(12, "burn ticked twice on e2"), _grade())
    assert c["verdict"] == "unmatched"
    assert c["active_nearby"] == ["B06", "B09"]


def test_evidence_text_never_creates_a_match():
    c = classify_flag(_flag(12, "Cinderfang's energy looks off", "audit: tooltip 12 ok, x1.5 advantage ok"), _grade())
    assert c["verdict"] == "unmatched"


def test_no_defect_nearby_is_a_likely_false_positive():
    c = classify_flag(_flag(40, "Cinderfang took more than the tooltip damage"), _grade())
    assert c["verdict"] == "no defect nearby"


def test_report_from_an_offline_campaign(tmp_path):
    run_session(
        SessionConfig(run_name="r", episodes=2, turn_cap=4, allowed_scenarios=["S1", "S5"]),
        runs_root=tmp_path / "runs", defects=full_build(), mode="replay",
        cassette=Cassette(dir_path=tmp_path / "empty"), progress=lambda _: None,
    )
    run_dir = tmp_path / "runs" / "r"

    # Plant one agent flag at a step where the doubled elemental multiplier
    # was actually active, naming a character it affected.
    ledger = Ledger(run_dir / "ledger.db")
    ep = ledger.episodes()[0]
    grade = grade_episode(ep["id"], Path(ep["log_path"]))
    step = grade.activity.steps["B09"][0]
    who = grade.activity.entities["B09"][step][0]
    ledger.add_flag(episode=ep["id"], turn=step, source="agent",
                    description=f"{who} took more damage than the tooltip", evidence="elemental multiplier looks doubled")
    ledger.close()

    report = generate_report(run_dir, results_root=tmp_path / "results", replay_root=tmp_path / "replays")
    text = report.read_text(encoding="utf-8")
    assert "## Defects: triggered vs detected" in text
    assert "all episodes replay exactly" in text
    assert "Oracle triage: **strong** -- B09" in text
    assert (tmp_path / "replays" / "r" / "ep_0001.html").exists()
    grades = json.loads((tmp_path / "results" / "r" / "grades.json").read_text())
    assert grades["per_defect"]["B08"]["triggered"] == [2]  # the S5 episode
