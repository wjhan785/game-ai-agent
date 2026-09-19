"""Episode JSONL -> static HTML replay (Jinja2 template in replay/templates/).
Shows every turn's diff, bars and violations next to the declared abilities."""

from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from engine.defects import ALL_DEFECT_IDS
from engine.episode_log import read_episode_jsonl

_FIELD_TO_DEFECT_ID = {field: defect_id for defect_id, field in ALL_DEFECT_IDS.items()}


def _defect_labels(field_names: list[str]) -> list[str]:
    """'enemy_no_basic_fallback' -> 'B08 enemy_no_basic_fallback', so the
    replay is readable against both ANSWER_KEY.md and the source flag."""
    return [
        f"{_FIELD_TO_DEFECT_ID.get(name, '?')} {name}" for name in field_names
    ]

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "episode.html.jinja"


def _pct(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * value / maximum))


def _build_character_legend(initial_state: dict) -> dict[str, dict[str, Any]]:
    legend: dict[str, dict[str, Any]] = {}
    for cid, c in initial_state["characters"].items():
        legend[cid] = {
            "id": cid,
            "name": c["name"],
            "team": c["team"],
            "element": c["element"],
            "max_hp": c["max_hp"],
            "max_energy": c["max_energy"],
            "speed": c.get("speed", 100.0),
            "abilities": c["abilities"],
        }
    return legend


def _snapshot_view(
    cid: str, snap: dict, legend: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    meta = legend.get(cid, {})
    max_hp = meta.get("max_hp", snap["hp"] or 1)
    max_energy = meta.get("max_energy", snap["energy"] or 1)
    return {
        "id": cid,
        "name": meta.get("name", cid),
        "team": meta.get("team", "?"),
        "hp": snap["hp"],
        "max_hp": max_hp,
        "hp_pct": _pct(snap["hp"], max_hp),
        "energy": snap["energy"],
        "max_energy": max_energy,
        "energy_pct": _pct(snap["energy"], max_energy),
        "alive": snap["alive"],
        "statuses": snap["statuses"],
        "cooldowns": {k: v for k, v in snap["cooldowns"].items() if v > 0},
        "speed": meta.get("speed", 100.0),
        "action_value": snap.get("action_value", 0.0),
    }


def _build_turn_view(
    turn: dict, legend: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    actor_id = turn["actor_id"]
    action = turn.get("action")
    ability_effect = turn.get("ability_effect")

    action_view: Optional[dict[str, Any]] = None
    if action is not None:
        target_id = action.get("target_id")
        action_view = {
            "ability_name": action["ability_name"],
            "target_id": target_id,
            "target_name": legend.get(target_id, {}).get("name", target_id) if target_id else None,
        }
        if ability_effect is not None:
            action_view["target_names"] = [
                legend.get(t, {}).get("name", t) for t in ability_effect.get("target_ids", [])
            ]

    after_snapshots = [
        _snapshot_view(cid, snap, legend) for cid, snap in turn.get("after", {}).items()
    ]
    # Stable, deterministic ordering for the per-turn roster table.
    after_snapshots.sort(key=lambda s: (s["team"], s["id"]))

    return {
        "index": turn["index"],
        "round_number": turn["round_number"],
        "elapsed_av": turn.get("elapsed_av"),
        "actor_id": actor_id,
        "actor_name": legend.get(actor_id, {}).get("name", actor_id),
        "skipped_reason": turn.get("skipped_reason"),
        "action": action_view,
        "agent_reasoning": turn.get("agent_reasoning"),
        "agent_audit": turn.get("agent_audit"),
        "decision_mode": turn.get("decision_mode"),
        "agent_flags": turn.get("agent_flags") or [],
        "dot_hot_damage": turn.get("dot_hot_damage") or {},
        "energy_regen_delta": turn.get("energy_regen_delta") or {},
        "ability_effect": ability_effect,
        "deaths": [legend.get(d, {}).get("name", d) for d in turn.get("deaths", [])],
        "invariant_violations": turn.get("invariant_violations") or [],
        "after": after_snapshots,
    }


def build_view_model(header: dict, turns: list[dict], footer: dict) -> dict[str, Any]:
    legend = _build_character_legend(header["initial_state"])
    initial_roster = sorted(legend.values(), key=lambda c: (c["team"], c["id"]))
    return {
        "meta": {
            "method": header.get("method"),
            "scenario_id": header["scenario_id"],
            "seed": header["seed"],
            "defects_enabled": _defect_labels(header.get("defects_enabled") or []),
            "turn_cap": header.get("turn_cap"),
            "finished": footer.get("finished"),
            "outcome": footer.get("outcome"),
            "step_count": footer.get("step_count"),
            "total_turns": len(turns),
        },
        "roster": initial_roster,
        "turns": [_build_turn_view(t, legend) for t in turns],
    }


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "jinja"]),
    )


def generate_replay_html(jsonl_path: str | Path, out_path: str | Path) -> Path:
    header, turns, footer = read_episode_jsonl(jsonl_path)
    view_model = build_view_model(header, turns, footer)

    template = _env().get_template(TEMPLATE_NAME)
    html = template.render(**view_model)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render an episode JSONL log as a static HTML replay.")
    parser.add_argument("jsonl_path", help="Path to the episode .jsonl file")
    parser.add_argument("out_path", help="Path to write the .html replay to")
    args = parser.parse_args()

    out = generate_replay_html(args.jsonl_path, args.out_path)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
