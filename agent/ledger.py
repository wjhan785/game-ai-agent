"""The exploration ledger: the agent's external memory across episodes, in
SQLite, so it survives outside any one context window and any one process.

Five tables, per the project plan:
  episodes    -- what was run, under which plan, what it cost, what came of it
  hypotheses  -- concrete, testable suspicions and their current status
  flags       -- every reported anomaly: the agent's own `flag_anomaly`
                 calls (source='agent') and the invariant checker's
                 violations (source='invariant'), side by side
  visits      -- (state_sig, action_sig) pairs seen, for novelty gating
  coverage    -- status-effect combinations seen, for the coverage curve
                 and for pointing the planner at untested interaction seams

`adjudication` and `bug_id` on `flags` are grading-time columns: nothing in
the agent ever writes or reads them (every agent-facing read below selects
explicit columns), and grading only happens after a sweep is finished.

State signatures are deliberately LOSSY -- per character: alive, an HP
bucket, an energy bucket, and the set of status types held (with a 1 / 2+
stack bucket), plus whose turn it is. An exact signature would make every
state unique and the visits table worthless for novelty.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from engine.models import BattleState, StatusType
from engine.resolution import CharacterSnapshot, TurnRecord

HYPOTHESIS_STATUSES = ("open", "testing", "confirmed", "refuted", "inconclusive")
ALL_STATUS_TYPES = tuple(sorted(st.value for st in StatusType))

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    scenario TEXT NOT NULL,
    seed INTEGER NOT NULL,
    plan TEXT NOT NULL DEFAULT '{}',
    outcome TEXT,
    actions_used INTEGER NOT NULL DEFAULT 0,
    llm_calls INTEGER NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '{}',
    log_path TEXT
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    created_ep INTEGER,
    resolved_ep INTEGER,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY,
    episode INTEGER NOT NULL,
    turn INTEGER NOT NULL,
    source TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT,
    evidence TEXT,
    trace_path TEXT,
    adjudication TEXT,
    bug_id TEXT
);
CREATE TABLE IF NOT EXISTS visits (
    state_sig TEXT NOT NULL,
    action_sig TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    first_episode INTEGER NOT NULL,
    PRIMARY KEY (state_sig, action_sig)
);
CREATE TABLE IF NOT EXISTS coverage (
    combo_sig TEXT PRIMARY KEY,
    first_seen_ep INTEGER NOT NULL,
    first_seen_action_idx INTEGER NOT NULL
);
"""

_FLAG_COLUMNS = "id, episode, turn, source, description, severity, evidence, trace_path"


# --- Signatures ---------------------------------------------------------


def _bucket(value: float, maximum: float) -> str:
    if value < 0:
        return "neg"
    if value == 0 or maximum <= 0:
        return "0"
    pct = value / maximum
    if pct < 0.25:
        return "<25"
    if pct < 0.50:
        return "<50"
    if pct < 0.75:
        return "<75"
    return "hi"


def state_signature(state: BattleState) -> str:
    chars = []
    for cid in sorted(state.characters):
        c = state.characters[cid]
        counts = Counter(s.status_type.value for s in c.statuses)
        statuses = sorted([t, "1" if n == 1 else "2+"] for t, n in counts.items())
        chars.append([cid, c.alive, _bucket(c.hp, c.max_hp), _bucket(c.energy, c.max_energy), statuses])
    payload = {"scenario": state.scenario_id, "turn": state.current_actor_id(), "chars": chars}
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def action_signature(ability_name: str, target_id: Optional[str]) -> str:
    return f"{ability_name}->{target_id or '-'}"


def _status_set(snap: CharacterSnapshot) -> tuple[str, ...]:
    return tuple(sorted({s.status_type.value for s in snap.statuses}))


def record_coverage_combos(record: TurnRecord, ability_element: Optional[str]) -> set[str]:
    """Coverage items one turn-slot exercised: every set of status types
    simultaneously held by one character at either end of the turn-slot
    (`set:burn+shield`), any status held as two or more instances at once
    (`stack:burn`), plus the (applied status, ability element) pair if the
    action applied a status (`pair:burn|fire`)."""
    combos: set[str] = set()
    for snap in list(record.before.values()) + list(record.after.values()):
        types = _status_set(snap)
        if types:
            combos.add("set:" + "+".join(types))
        counts = Counter(s.status_type.value for s in snap.statuses)
        combos.update(f"stack:{t}" for t, n in counts.items() if n >= 2)
    effect = record.ability_effect
    if effect is not None and effect.applied_status and ability_element:
        combos.add(f"pair:{effect.applied_status}|{ability_element}")
    return combos


def status_pairs_from_combos(combo_sigs: Iterable[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for sig in combo_sigs:
        if not sig.startswith("set:"):
            continue
        types = sig[len("set:"):].split("+")
        pairs.update(itertools.combinations(sorted(types), 2))
    return pairs


ALL_STATUS_PAIRS = tuple(itertools.combinations(ALL_STATUS_TYPES, 2))


# --- Ledger ---------------------------------------------------------------


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # --- episodes ---

    def start_episode(self, scenario: str, seed: int, plan: dict, log_path: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO episodes (scenario, seed, plan, log_path) VALUES (?, ?, ?, ?)",
            (scenario, seed, json.dumps(plan, sort_keys=True), log_path),
        )
        return int(cur.lastrowid)

    def finish_episode(
        self,
        episode_id: int,
        *,
        outcome: Optional[str],
        actions_used: int,
        llm_calls: int,
        tokens_in: int,
        tokens_out: int,
        cache_hit_tokens: int,
        cost_usd: float,
        summary: dict,
    ) -> None:
        self._conn.execute(
            "UPDATE episodes SET outcome=?, actions_used=?, llm_calls=?, tokens_in=?, tokens_out=?, "
            "cache_hit_tokens=?, cost_usd=?, summary=? WHERE id=?",
            (
                outcome,
                actions_used,
                llm_calls,
                tokens_in,
                tokens_out,
                cache_hit_tokens,
                cost_usd,
                json.dumps(summary, sort_keys=True),
                episode_id,
            ),
        )

    def episodes(self) -> list[dict]:
        rows = self._rows("SELECT * FROM episodes ORDER BY id")
        for r in rows:
            r["plan"] = json.loads(r["plan"])
            r["summary"] = json.loads(r["summary"])
        return rows

    def total_actions(self) -> int:
        row = self._conn.execute("SELECT COALESCE(SUM(actions_used), 0) FROM episodes").fetchone()
        return int(row[0])

    # --- hypotheses ---

    def add_hypothesis(self, text: str, created_ep: Optional[int], status: str = "open", notes: str = "") -> int:
        if status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"unknown hypothesis status {status!r}")
        cur = self._conn.execute(
            "INSERT INTO hypotheses (text, status, created_ep, notes) VALUES (?, ?, ?, ?)",
            (text, status, created_ep, notes),
        )
        return int(cur.lastrowid)

    def update_hypothesis(self, hypothesis_id: int, status: str, note: str, episode: Optional[int]) -> bool:
        if status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"unknown hypothesis status {status!r}")
        row = self._conn.execute("SELECT notes FROM hypotheses WHERE id=?", (hypothesis_id,)).fetchone()
        if row is None:
            return False
        notes = row["notes"]
        if note:
            prefix = f"[ep {episode}] " if episode is not None else ""
            notes = f"{notes}\n{prefix}{note}".strip()
        resolved_ep = episode if status in ("confirmed", "refuted", "inconclusive") else None
        self._conn.execute(
            "UPDATE hypotheses SET status=?, notes=?, resolved_ep=? WHERE id=?",
            (status, notes, resolved_ep, hypothesis_id),
        )
        return True

    def hypotheses(self, statuses: Optional[Iterable[str]] = None) -> list[dict]:
        if statuses is None:
            return self._rows("SELECT * FROM hypotheses ORDER BY id")
        statuses = tuple(statuses)
        marks = ",".join("?" for _ in statuses)
        return self._rows(f"SELECT * FROM hypotheses WHERE status IN ({marks}) ORDER BY id", statuses)

    def hypothesis_exists(self, hypothesis_id: int) -> bool:
        return self._conn.execute("SELECT 1 FROM hypotheses WHERE id=?", (hypothesis_id,)).fetchone() is not None

    # --- flags ---

    def add_flag(
        self,
        *,
        episode: int,
        turn: int,
        source: str,
        description: str,
        severity: Optional[str] = None,
        evidence: Optional[str] = None,
        trace_path: Optional[str] = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO flags (episode, turn, source, description, severity, evidence, trace_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (episode, turn, source, description, severity, evidence, trace_path),
        )
        return int(cur.lastrowid)

    def set_flag_trace(self, flag_id: int, trace_path: str) -> None:
        self._conn.execute("UPDATE flags SET trace_path=? WHERE id=?", (trace_path, flag_id))

    def flags(self, *, source: Optional[str] = None, episode: Optional[int] = None) -> list[dict]:
        clauses, params = [], []
        if source is not None:
            clauses.append("source=?")
            params.append(source)
        if episode is not None:
            clauses.append("episode=?")
            params.append(episode)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._rows(f"SELECT {_FLAG_COLUMNS} FROM flags {where} ORDER BY id", tuple(params))

    # --- visits (novelty) ---

    def seen(self, state_sig: str) -> bool:
        return self._conn.execute("SELECT 1 FROM visits WHERE state_sig=? LIMIT 1", (state_sig,)).fetchone() is not None

    def record_visit(self, state_sig: str, action_sig: str, episode: int) -> None:
        self._conn.execute(
            "INSERT INTO visits (state_sig, action_sig, count, first_episode) VALUES (?, ?, 1, ?) "
            "ON CONFLICT(state_sig, action_sig) DO UPDATE SET count = count + 1",
            (state_sig, action_sig, episode),
        )

    def visit_stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT state_sig), COUNT(*), COALESCE(SUM(count), 0) FROM visits"
        ).fetchone()
        return {"distinct_states": int(row[0]), "distinct_state_actions": int(row[1]), "total_visits": int(row[2])}

    # --- coverage ---

    def record_coverage(self, combo_sig: str, episode: int, action_idx: int) -> bool:
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO coverage (combo_sig, first_seen_ep, first_seen_action_idx) VALUES (?, ?, ?)",
            (combo_sig, episode, action_idx),
        )
        return cur.rowcount == 1

    def coverage(self) -> list[dict]:
        return self._rows("SELECT * FROM coverage ORDER BY first_seen_action_idx, combo_sig")

    # --- planner digest ---

    def digest(self, *, max_flags: int = 15, max_resolved: int = 15) -> dict:
        """Everything the planner needs at an episode boundary, bounded so
        the planning prompt doesn't grow without limit over a long sweep."""
        episodes = self.episodes()
        per_scenario = Counter(e["scenario"] for e in episodes)

        last = None
        if episodes:
            e = episodes[-1]
            last = {
                "id": e["id"],
                "scenario": e["scenario"],
                "seed": e["seed"],
                "plan": e["plan"],
                "outcome": e["outcome"],
                "actions_used": e["actions_used"],
                "summary": e["summary"],
            }

        active = [
            {"id": h["id"], "text": h["text"], "status": h["status"], "notes": h["notes"]}
            for h in self.hypotheses(("open", "testing"))
        ]
        resolved = [
            {"id": h["id"], "text": h["text"], "status": h["status"], "notes": h["notes"]}
            for h in self.hypotheses(("confirmed", "refuted", "inconclusive"))
        ][-max_resolved:]

        agent_flags = [
            {"id": f["id"], "episode": f["episode"], "severity": f["severity"], "description": f["description"]}
            for f in self.flags(source="agent")
        ][-max_flags:]

        invariant_kinds: dict[str, dict] = {}
        for f in self.flags(source="invariant"):
            kind = invariant_kind(f["description"])
            entry = invariant_kinds.setdefault(kind, {"kind": kind, "episodes": set(), "first_episode": f["episode"]})
            entry["episodes"].add(f["episode"])
        invariant_summary = [
            {"kind": v["kind"], "episodes": len(v["episodes"]), "first_episode": v["first_episode"]}
            for v in sorted(invariant_kinds.values(), key=lambda v: (v["first_episode"], v["kind"]))
        ]

        combos = [c["combo_sig"] for c in self.coverage()]
        seen_pairs = status_pairs_from_combos(combos)
        return {
            "episodes_run": len(episodes),
            "actions_total": self.total_actions(),
            "episodes_per_scenario": dict(sorted(per_scenario.items())),
            "last_episode": last,
            "hypotheses_active": active,
            "hypotheses_resolved_recent": resolved,
            "agent_flags_recent": agent_flags,
            "invariant_violation_kinds": invariant_summary,
            "coverage": {
                "status_pairs_seen_together": ["+".join(p) for p in sorted(seen_pairs)],
                "status_pairs_never_seen_together": [
                    "+".join(p) for p in ALL_STATUS_PAIRS if p not in seen_pairs
                ],
                "applied_status_by_element_seen": sorted(c[len("pair:"):] for c in combos if c.startswith("pair:")),
                "statuses_seen_as_multiple_instances": sorted(c[len("stack:"):] for c in combos if c.startswith("stack:")),
            },
        }


_PAREN_WITH_NUMBER = re.compile(r"\s*\([^)]*\d[^)]*\)")


def invariant_kind(description: str) -> str:
    """Normalizes an invariant-violation message to its kind, so the same
    violation on different characters or with different numbers counts
    once: 'e1: energy went negative (-5.0)' -> 'energy went negative'."""
    text = description.split(":", 1)[1] if ":" in description else description
    text = _PAREN_WITH_NUMBER.sub("", text)
    words = [w for w in text.split() if not any(ch.isdigit() for ch in w)]
    return " ".join(words)
