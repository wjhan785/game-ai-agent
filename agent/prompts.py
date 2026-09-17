"""What the model sees: the rules specification, roster formatting, and
compact per-turn observations, shared by the inner (tactical) and outer
(planning) loops.

RULES_SPEC is the game's design specification -- the document a QA tester
tests against. It describes every mechanic at the same level of detail
(the plan's engine specification plus the fixed turn pipeline), because a
spec that went quiet on some mechanics would make those untestable by
reasoning, and one that dwelt on some would be steering. Everything the
agent reports is a claim that observed behavior contradicts this text or
an ability's declared data.

Observations pair DECLARED numbers (an ability's power and element, a
status's magnitude) with OBSERVED ones (HP deltas, tick amounts, the
statuses actually present). The arithmetic of differencing is done here;
judging whether a difference is legitimate (Weaken, Shield, a death) or a
contradiction is the model's job.
"""
from __future__ import annotations

import json
from typing import Optional

from engine.elements import elemental_multiplier
from engine.models import BattleState, Character, StatusEffect
from engine.resolution import TurnRecord

RULES_SPEC = """GAME RULES (the design specification under test)

Battle
- Two teams of three: party p1-p3 and enemy e1-e3. Turn order is driven by speed, not fixed: each character has a speed stat and an action value (AV), AV = 10000 / speed, and whoever holds the lowest AV acts next -- a faster character takes MORE turns over the same stretch of battle, not just earlier ones. Equal AV (a tie) is broken by the declared order p1, e1, p2, e2, p3, e3. After a character's turn-slot, its AV becomes (the clock reading that selected it) + 10000 / its own speed -- one full personal cycle later, regardless of any other character's speed.
- 100 AV of elapsed clock is one cycle; the round number is how many whole cycles have elapsed since battle start, and the battle ends when a team is fully dead or the round cap (in cycles) is reached.
- A dead character still occupies its place in the schedule: its turn-slot is skipped (not removed), and it is rescheduled exactly like a living one, since Revive can bring it back with an AV already waiting.
- Each character's turn-slot resolves in this order:
  1. A dead character's turn-slot is skipped.
  2. Start-of-turn ticks on the acting character: Poison, then Burn, then Regen.
  3. If the ticks brought HP to 0, the character dies and the turn-slot ends.
  4. Energy regeneration.
  5. A stunned character's turn-slot ends here, with no action.
  6. The character uses one legal ability.
  7. End of turn: each of the acting character's cooldown counters drops by 1 -- including a counter set during this same turn -- and each status the acting character holds loses 1 turn of duration -- including one it gave itself during this same turn; a status at 0 turns left expires. Statuses on other characters are untouched until their own end of turn.

Abilities
- Declared data: element, energy cost, cooldown, target type (self / one_enemy / one_ally / all_enemies), base power, and optionally a status applied (type, magnitude, duration in turns).
- An ability is legal only if the user has at least its energy cost, its cooldown counter is 0, and a valid target exists. Using it spends the energy cost and sets its cooldown counter to the declared cooldown. The counter drops by 1 at the end of each of the user's own turn-slots (stunned ones included), starting with the turn it was used: a cooldown-2 ability reads 1 at the end of the turn it was used and 0 at the end of the next, so it is unavailable for one of the user's turns; cooldown 1 means usable again on the user's next turn; cooldown 3 means unavailable for two.
- Every character has Basic Attack (0 cost, 0 cooldown), so a living, unstunned character always has a legal action.
- A status applied by an ability lands only if its target is still alive after the hit.

Direct-hit damage
- damage = base power x elemental multiplier x (1 - attacker's Weaken magnitude, if the attacker is Weakened). The target's Shield then absorbs up to its remaining pool, and the rest is subtracted from HP (HP never below 0).
- Elemental multiplier (ability element vs target character's element): Fire beats Ice, Ice beats Lightning, Lightning beats Fire. Attacking an element you beat is advantage, x1.5: fire->ice, ice->lightning, lightning->fire. Attacking an element that beats yours is disadvantage, x(1/1.5) = x0.667: ice->fire, lightning->ice, fire->lightning. Same element, or Neutral on either side: x1.0.

Statuses
- Burn: each tick deals the Burn's magnitude as damage. Every Burn application adds its own separate instance; a tick deals the sum of all Burn instances on the character.
- Poison: each tick deals magnitude x the poisoned character's max HP (0.08 = 8% of max HP).
- Regen: each tick heals its magnitude, never above max HP.
- Shield: a pool that absorbs direct ability-hit damage only; Poison and Burn ticks are not absorbed. Removed when its pool reaches 0 or its duration runs out.
- Chill: reduces the character's energy regeneration by magnitude x 100% (1.2 = 120%); the total reduction is capped at 100%.
- Weaken: reduces the character's own outgoing direct-hit damage by magnitude x 100%.
- Stun: the character's turn-slots are skipped while it is stunned.
- Every status except Burn is non-stacking: a character holds at most one instance of each, and reapplying replaces its magnitude and duration.

Resources and death
- HP stays within [0, max HP]; energy stays within [0, max energy]. Energy regeneration happens at step 4, before the action: regeneration = the character's energy regen x (1 - Chill reduction), but never above max energy, so a character already at max energy regenerates nothing. Energy at the end of a turn-slot = energy at its start + regeneration - the cost of the ability used.
- At 0 HP a character dies: all its statuses are removed at once; it takes no turns and cannot be targeted, except by Revive.
- Revive returns a dead ally to life at 50% of max HP."""

OBSERVATION_FORMAT = """Observation format
- Statuses are written type:magnitude/turns_left/source, e.g. burn:6/2/e1.
- tooltip_damage = base power x elemental multiplier, i.e. before the attacker's Weaken and the target's Shield. The test harness computes it from the declared data and the multiplier table above: it is the reference to audit against, not an observation of the game. damage_taken is the HP actually lost; shield_absorbed is what the target's Shield took.
- In a turn-slot, start_of_turn lists the statuses the actor held when its ticks fired, and the ticks they produced; energy is the actor's energy start, max, regen, spent and end for that turn-slot; actor_end_of_turn is the actor after its action and end-of-turn countdown. Turn-slots marked "new" happened since your last decision.
- The current state lists each living character's spd (speed) and av (action value) alongside hp/en/statuses/cooldowns, plus a forecast: the next several actors in order (id and the AV each will act at), assuming no death, revive, or speed change between now and then -- a planning aid, not a promise."""

AUDIT_CHECKLIST = """Audit checklist (every mechanic, every time)
- Direct hits: damage_taken + shield_absorbed = tooltip_damage x (1 - attacker's Weaken)?
- Ticks: Burn = sum of Burn magnitudes held; Poison = magnitude x the character's max HP; Regen = magnitude, capped at max HP.
- Energy: regen = min(energy regen x (1 - Chill, capped at 100%), max - start); end = start + regen - spent; never below 0.
- Statuses: applied with the declared magnitude and duration; non-stacking ones refreshed, not duplicated; each loses 1 turn at the end of its holder's own turn (the same turn, if the holder gave it to itself); Shield gone at 0 pool; all statuses gone on death.
- Cooldowns: set to the declared cooldown on use, then drop by 1 at the end of that same turn and of each later own turn; usable only at 0.
- Turns: the living character with the lowest AV acts next (ties break by the declared order p1, e1, p2, e2, p3, e3); after acting, its AV becomes the clock reading that selected it plus 10000/its speed; the dead still occupy their place in the schedule and skip; the stunned skip too; every living, unstunned character has a legal action.
A MISMATCH is a definite contradiction you can show with the numbers given. A value at a cap or floor, or a number you cannot reconstruct from what is shown, is not one; record a suspicion like that with ledger_write instead."""


def _num(x: float) -> float | int:
    r = round(x, 2)
    return int(r) if r == int(r) else r


def fmt_status(s: StatusEffect) -> str:
    return f"{s.status_type.value}:{_num(s.magnitude)}/{s.turns_remaining}/{s.source_id}"


def fmt_statuses(statuses: list[StatusEffect]) -> list[str]:
    return [fmt_status(s) for s in statuses]


def dumps(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"))


def format_character(c: Character) -> str:
    head = (
        f"{c.id} {c.name} | {c.team} | {c.element.value} | spd {_num(c.speed)} | HP {_num(c.max_hp)} | "
        f"EN {_num(c.max_energy)} (+{_num(c.energy_regen)}/turn)"
    )
    lines = [head]
    for a in c.abilities:
        line = (
            f"  - {a.name}: {a.element.value}, cost {a.energy_cost}, cd {a.cooldown}, "
            f"{a.target_type.value}, power {_num(a.base_power)}"
        )
        if a.applies is not None:
            line += f", applies {a.applies.status_type.value} {_num(a.applies.magnitude)} for {a.applies.duration}t"
        if a.is_revive:
            line += f", revives at {int(a.revive_hp_fraction * 100)}% HP"
        lines.append(line)
    return "\n".join(lines)


def format_roster(state: BattleState) -> str:
    return "\n".join(format_character(state.characters[cid]) for cid in sorted(state.characters, key=_team_order))


def _team_order(cid: str) -> tuple[int, str]:
    return (0 if cid.startswith("p") else 1, cid)


def _tooltip_damage(state: BattleState, actor_id: str, ability_name: str, target_id: str) -> Optional[float]:
    actor = state.characters.get(actor_id)
    target = state.characters.get(target_id)
    if actor is None or target is None:
        return None
    ability = actor.ability_by_name(ability_name)
    if ability is None or ability.base_power <= 0:
        return None
    return ability.base_power * elemental_multiplier(ability.element, target.element)


def _energy_breakdown(record: TurnRecord, state: BattleState, start: float, end: float) -> dict:
    """start + regen - spent = end, each term stated, so checking energy is
    arithmetic on given numbers rather than reconstructing a timeline."""
    actor = state.characters[record.actor_id]
    out: dict = {"start": _num(start), "max": _num(actor.max_energy), "regen": _num(record.energy_regen_delta.get(record.actor_id, 0.0))}
    if record.action is not None:
        ability = actor.ability_by_name(record.action.ability_name)
        out["spent"] = ability.energy_cost if ability is not None else None
    out["end"] = _num(end)
    return out


def summarize_record(record: TurnRecord, state: BattleState) -> dict:
    """One finished turn-slot, compactly: the actor's start-of-turn ticks
    (with the statuses it held when they fired), its action, and for each
    target the declared tooltip damage next to what actually happened."""
    actor = record.actor_id
    ev: dict = {"step": record.step_count, "round": record.round_number, "actor": actor}
    before = record.before.get(actor)
    after = record.after.get(actor)

    ticks = record.dot_hot_damage.get(actor)
    if before is not None and (ticks or before.statuses):
        start: dict = {"statuses": fmt_statuses(before.statuses), "hp": _num(before.hp)}
        if ticks:
            start["ticks"] = {k: _num(v) for k, v in ticks.items()}
        ev["start_of_turn"] = start
    if record.skipped_reason:
        ev["skipped"] = record.skipped_reason
    if before is not None and after is not None and record.skipped_reason != "dead":
        ev["energy"] = _energy_breakdown(record, state, before.energy, after.energy)

    if record.action is not None:
        ev["action"] = f"{record.action.ability_name} -> {record.action.target_id or '-'}"
    effect = record.ability_effect
    if effect is not None:
        effects = []
        for tid in effect.target_ids:
            t_before, t_after = record.before.get(tid), record.after.get(tid)
            entry: dict = {"target": tid}
            tooltip = _tooltip_damage(state, actor, effect.ability_name, tid)
            if tooltip is not None:
                entry["tooltip_damage"] = _num(tooltip)
            if tid in effect.per_target_damage:
                entry["damage_taken"] = _num(effect.per_target_damage[tid])
            if tid in effect.per_target_shield_absorbed:
                entry["shield_absorbed"] = _num(effect.per_target_shield_absorbed[tid])
            if t_before is not None and t_after is not None:
                entry["hp"] = [_num(t_before.hp), _num(t_after.hp)]
                entry["statuses_after"] = fmt_statuses(t_after.statuses)
            effects.append(entry)
        ev["effects"] = effects
        if effect.applied_status:
            ev["applied"] = effect.applied_status
        if effect.revived:
            ev["revived"] = effect.revived

    if record.deaths:
        ev["deaths"] = record.deaths
    if after is not None and record.skipped_reason != "dead":
        end: dict = {"hp": _num(after.hp), "statuses": fmt_statuses(after.statuses)}
        if after.cooldowns:
            end["cooldowns"] = dict(after.cooldowns)
        ev["actor_end_of_turn"] = end
    if record.invariant_violations:
        ev["invariant_violations"] = list(record.invariant_violations)
    return ev


MAX_NEW_EVENTS = 12


def recent_events(
    log: list[TurnRecord], state: BattleState, window: int, audit_from: Optional[int] = None
) -> list[dict]:
    """The last `window` informative turn-slots -- extended, if needed, to
    every turn-slot from step `audit_from` on (up to MAX_NEW_EVENTS), each
    marked `"new": true`, so nothing played since the model last decided
    goes unseen. A dead character's skipped turn carries nothing unless the
    invariant checker objected to it."""
    kept = [r for r in log if r.skipped_reason != "dead" or r.invariant_violations]
    shown = kept[-window:]
    if audit_from is not None:
        new = [r for r in kept if r.step_count >= audit_from][-MAX_NEW_EVENTS:]
        if len(new) > len(shown):
            shown = new
    events = [summarize_record(r, state) for r in shown]
    if audit_from is not None:
        for ev in events:
            if ev["step"] >= audit_from:
                ev["new"] = True
    return events


def pending_turn(pending: Optional[TurnRecord], state: BattleState) -> Optional[dict]:
    """The current decision's turn-slot so far: what ticked at the start of
    it, against the statuses held when the ticks fired."""
    if pending is None:
        return None
    actor = pending.actor_id
    before = pending.before.get(actor)
    now = state.characters[actor]
    if before is None:
        return {"actor": actor}
    out: dict = {
        "actor": actor,
        "statuses_before_ticks": fmt_statuses(before.statuses),
        "hp": [_num(before.hp), _num(now.hp)],
        "energy": {
            "start": _num(before.energy),
            "max": _num(now.max_energy),
            "regen": _num(pending.energy_regen_delta.get(actor, 0.0)),
            "now": _num(now.energy),
        },
    }
    ticks = pending.dot_hot_damage.get(actor)
    if ticks:
        out["ticks"] = {k: _num(v) for k, v in ticks.items()}
    return out


def compact_state(state: BattleState) -> dict:
    chars = {}
    for cid in sorted(state.characters, key=_team_order):
        c = state.characters[cid]
        if not c.alive:
            chars[cid] = "dead"
            continue
        entry: dict = {
            "hp": f"{_num(c.hp)}/{_num(c.max_hp)}",
            "en": f"{_num(c.energy)}/{_num(c.max_energy)}",
            "spd": _num(c.speed),
            "av": _num(c.action_value),
        }
        if c.statuses:
            entry["st"] = fmt_statuses(c.statuses)
        live_cds = {k: v for k, v in c.cooldowns.items() if v > 0}
        if live_cds:
            entry["cd"] = live_cds
        chars[cid] = entry
    forecast = [{"id": cid, "av": _num(av)} for cid, av in state.turn_forecast(6)]
    return {"round": state.round_number, "step": state.step_count, "characters": chars, "forecast": forecast}


def compact_legal(legal_options: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for o in legal_options:
        grouped.setdefault(o["ability_name"], []).append(o["target_id"] or "-")
    return grouped
