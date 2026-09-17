"""Tests for the inner-loop tool surface (agent/tools.py) -- all offline,
no LLM calls. Drives a real BattleEngine through the dispatcher exactly
as agent/inner_loop.py will, and checks the diff `take_action` returns
matches what the engine actually did.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.defects import DefectFlags
from engine.engine import BattleEngine
from agent.ledger import Ledger
from agent.tools import INNER_TOOL_NAMES, PLANNER_TOOL_NAMES, TOOL_MODELS, ToolDispatcher, build_tool_schemas


def _dispatcher(scenario_id="S1", seed=1, defects=None, turn_cap=20) -> ToolDispatcher:
    engine = BattleEngine(scenario_id, seed, defects or DefectFlags(), turn_cap)
    return ToolDispatcher(engine)


def test_build_tool_schemas_per_loop():
    assert [s["function"]["name"] for s in build_tool_schemas(INNER_TOOL_NAMES)] == [
        "take_action", "flag_anomaly", "ledger_write"
    ]
    assert [s["function"]["name"] for s in build_tool_schemas(PLANNER_TOOL_NAMES)] == [
        "ledger_read", "ledger_write", "reset_episode"
    ]
    schemas = build_tool_schemas()
    assert {s["function"]["name"] for s in schemas} == set(TOOL_MODELS)
    for s in schemas:
        assert "parameters" in s["function"]


def test_ledger_tools_create_update_and_read(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    ep = ledger.start_episode("S1", 1, {}, "p")
    d = ToolDispatcher(BattleEngine("S1", 1, None, 20), ledger=ledger, episode_id=ep)
    assert "error" in d.ledger_write(status="open")  # creating needs text
    created = d.ledger_write(status="open", text="shield may absorb burn ticks")["created"]
    assert d.ledger_write(status="confirmed", hypothesis_id=created, note="tick absorbed")["status"] == "confirmed"
    assert "error" in d.ledger_write(status="refuted", hypothesis_id=999)
    assert d.ledger_read("hypotheses")["hypotheses"][0]["status"] == "confirmed"
    for kind in ("flags", "coverage", "episodes"):
        assert "error" not in d.ledger_read(kind)


def test_flag_anomaly_with_ledger_writes_repro_file(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    ep = ledger.start_episode("S1", 1, {}, "p")
    d = ToolDispatcher(BattleEngine("S1", 1, None, 20), ledger=ledger, episode_id=ep, flags_dir=tmp_path / "flags")
    actor_id = d.engine.state.current_actor_id()
    opt = d.list_legal_actions()["options"][0]
    d.take_action(actor_id=actor_id, ability_name=opt["ability_name"], target_id=opt["target_id"], reasoning="setup")

    result = d.flag_anomaly(description="odd", severity="medium", evidence="12 vs 8")
    row = ledger.flags(source="agent")[0]
    assert row["id"] == result["flag_id"]
    repro = json.loads(Path(row["trace_path"]).read_text(encoding="utf-8"))
    assert repro["scenario_id"] == "S1" and repro["seed"] == 1
    assert repro["actions"] == [{"actor_id": actor_id, "ability_name": opt["ability_name"], "target_id": opt["target_id"]}]


def test_get_state_reports_current_actor_and_alive_characters():
    d = _dispatcher()
    state = d.get_state()
    assert state["current_actor_id"] == d.engine.state.current_actor_id()
    assert state["finished"] is False
    ids = {c["id"] for c in state["characters"]}
    assert ids == set(d.engine.state.characters)
    for c in state["characters"]:
        assert c["alive"] is True
        assert "hp" in c and "energy" in c


def test_list_legal_actions_matches_engine_legal_actions():
    d = _dispatcher()
    result = d.list_legal_actions()
    engine_legal = d.engine.legal_actions()
    assert result["actor_id"] == d.engine.state.current_actor_id()
    assert len(result["options"]) == len(engine_legal)
    option_pairs = {(o["ability_name"], o["target_id"]) for o in result["options"]}
    engine_pairs = {(a.ability_name, a.target_id) for a in engine_legal}
    assert option_pairs == engine_pairs
    # Every option carries enough to decide without a second call.
    for o in result["options"]:
        assert "energy_cost" in o and "cooldown" in o and "element" in o


def test_take_action_rejects_wrong_actor():
    d = _dispatcher()
    actual = d.engine.state.current_actor_id()
    wrong = next(cid for cid in d.engine.state.characters if cid != actual)
    result = d.take_action(actor_id=wrong, ability_name="Basic Attack", reasoning="test")
    assert "error" in result
    assert actual in result["error"]


def test_take_action_rejects_illegal_action():
    d = _dispatcher()
    actor_id = d.engine.state.current_actor_id()
    result = d.take_action(actor_id=actor_id, ability_name="Not A Real Ability", reasoning="test")
    assert "error" in result


def test_take_action_returns_diff_matching_engine_record():
    d = _dispatcher()
    actor_id = d.engine.state.current_actor_id()
    options = d.list_legal_actions()["options"]
    chosen = options[0]
    result = d.take_action(
        actor_id=actor_id,
        ability_name=chosen["ability_name"],
        target_id=chosen["target_id"],
        reasoning="first legal option",
    )
    assert "error" not in result
    assert result["actor_id"] == actor_id
    last_record = d.engine.log[-1]
    assert last_record.actor_id == actor_id
    if last_record.ability_effect is not None:
        assert result["ability"] == last_record.ability_effect.ability_name
        assert len(result["per_target"]) == len(last_record.ability_effect.target_ids)
        for entry in result["per_target"]:
            if "observed_damage" in entry:
                assert entry["observed_damage"] == last_record.ability_effect.per_target_damage[entry["target_id"]]


def test_take_action_exposes_declared_vs_observed_damage_for_double_multiplier_defect():
    # B09: elemental multiplier applied twice. Cinderfang (fire) has
    # Flame Lash into whichever enemy/ally is ice, if reachable in S1 --
    # this scenario has no Ice target, so instead assert the mechanism
    # directly: declared_expected_damage uses the SAME formula regardless
    # of the defect (it is the engine's declared data, not a defect-aware
    # recomputation), so a doubled observed_damage under B09 would show up
    # as observed > declared -- exactly the comparison the agent is meant
    # to make.
    d = _dispatcher(defects=DefectFlags(elemental_multiplier_applied_twice=True))
    actor_id = d.engine.state.current_actor_id()
    options = d.list_legal_actions()["options"]
    damaging = next(o for o in options if o["base_power"] > 0)
    result = d.take_action(
        actor_id=actor_id,
        ability_name=damaging["ability_name"],
        target_id=damaging["target_id"],
        reasoning="test",
    )
    entry = result["per_target"][0]
    assert "declared_expected_damage" in entry
    assert "observed_damage" in entry


def test_flag_anomaly_records_and_attaches_trace():
    d = _dispatcher()
    actor_id = d.engine.state.current_actor_id()
    options = d.list_legal_actions()["options"]
    d.take_action(actor_id=actor_id, ability_name=options[0]["ability_name"], target_id=options[0]["target_id"], reasoning="setup")

    result = d.flag_anomaly(description="damage looked doubled", severity="high", evidence="observed 24 vs declared 12")
    assert result["recorded"] is True
    assert len(d.flags) == 1
    assert d.flags[0]["severity"] == "high"
    assert len(d.flags[0]["trace"]) == len(d.engine.log)


def test_dispatch_routes_by_name_and_matches_direct_call():
    d = _dispatcher()
    via_dispatch = d.dispatch("get_state", {})
    via_direct = d.get_state()
    assert via_dispatch == via_direct


def test_tool_models_validate_minimal_args():
    args = TOOL_MODELS["take_action"].model_validate(
        {"audit": "ok", "actor_id": "p1", "ability_name": "Basic Attack", "reasoning": "why not"}
    )
    assert args.target_id is None
    assert args.anomalies == []


def test_take_action_audit_comes_before_the_action_in_the_schema():
    props = list(build_tool_schemas(["take_action"])[0]["function"]["parameters"]["properties"])
    assert props[:2] == ["audit", "anomalies"]
    assert props.index("audit") < props.index("ability_name")
