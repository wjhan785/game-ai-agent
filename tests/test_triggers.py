"""oracle/triggers.py: leave-one-out trigger detection and one-step
activity on traces from builds with several defects enabled at once.

The strongest check here is agreement between the two procedures: they
share no code path beyond the engine itself (one replays whole
trajectories, the other re-runs single steps from logged state), so if
they agree on the first step each defect acted, both are likely right."""
from __future__ import annotations

from engine.defects import ALL_DEFECT_IDS, DefectFlags, single_flag
from engine.engine import BattleEngine
from engine import scenarios as scenarios_module
from oracle.triggers import CONSTRUCTION_TIME_DEFECTS, leave_one_out_triggers, stepwise_activity
from tests.test_oracle import _make_b06_1v1_scenario

FULL_BUILD = DefectFlags(**{f: True for f in ALL_DEFECT_IDS.values()})


def _run_first_legal(scenario_id, seed, build, turn_cap=20):
    eng = BattleEngine(scenario_id, seed, build, turn_cap)
    initial = eng.state.model_copy(deep=True)
    while not eng.state.finished and eng.legal_actions():
        eng.take_action(eng.legal_actions()[0])
    return eng, initial


def test_full_build_traces_replay_exactly_and_step_consistently():
    for sid in ("S1", "S2", "S3", "S4", "S5", "S6"):
        eng, initial = _run_first_legal(sid, 5, FULL_BUILD)
        report = leave_one_out_triggers(sid, 5, 20, eng.log, FULL_BUILD)
        assert report.reproduces, sid
        assert stepwise_activity(initial, eng.log, FULL_BUILD).inconsistent_steps == [], sid


def test_leave_one_out_and_stepwise_agree_on_first_activity():
    for sid in ("S1", "S2", "S3", "S4", "S5", "S6"):
        eng, initial = _run_first_legal(sid, 5, FULL_BUILD)
        triggers = leave_one_out_triggers(sid, 5, 20, eng.log, FULL_BUILD).triggers
        activity = stepwise_activity(initial, eng.log, FULL_BUILD)
        for did, t in triggers.items():
            if did in CONSTRUCTION_TIME_DEFECTS:
                continue
            first_active = activity.steps[did][0] if activity.steps[did] else None
            assert (t.step if t.triggered else None) == first_active, (sid, did, t, first_active)


def test_construction_time_defect_is_triggered_and_active_on_its_skips():
    eng, initial = _run_first_legal("S5", 1, FULL_BUILD, turn_cap=30)
    assert leave_one_out_triggers("S5", 1, 30, eng.log, FULL_BUILD).triggers["B08"].triggered
    steps = stepwise_activity(initial, eng.log, FULL_BUILD).steps["B08"]
    assert steps and all(eng.log[k].skipped_reason == "no_legal_actions" for k in steps)


def test_clean_trace_triggers_nothing_it_does_not_contain():
    eng, initial = _run_first_legal("S1", 2, DefectFlags())
    report = leave_one_out_triggers("S1", 2, 20, eng.log, DefectFlags())
    assert report.reproduces and report.triggers == {}
    assert stepwise_activity(initial, eng.log, DefectFlags()).steps == {}


def test_stepwise_finds_the_early_cooldown_use():
    scenarios_module.SCENARIOS["_T_B06_STEP"] = _make_b06_1v1_scenario()
    try:
        build = single_flag("B06")
        eng = BattleEngine("_T_B06_STEP", 0, build, 30)
        initial = eng.state.model_copy(deep=True)
        for _ in range(6):
            legal = eng.legal_actions()
            big = [a for a in legal if a.ability_name == "Big Hit"]
            eng.take_action(big[0] if big else legal[0])
        activity = stepwise_activity(initial, eng.log, build)
        early_uses = [
            r.step_count for r in eng.log
            if r.action and r.action.ability_name == "Big Hit" and r.before["atk"].cooldowns.get("Big Hit", 0) > 0
        ]
        assert early_uses and set(early_uses) <= set(activity.steps["B06"])
        assert all(activity.entities["B06"][k] == ["atk"] for k in early_uses)
    finally:
        del scenarios_module.SCENARIOS["_T_B06_STEP"]
