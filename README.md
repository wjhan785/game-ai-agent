# game-ai-agent

A headless turn-based tactics combat engine with ten deliberately seeded
defects, built to be hunted by an LLM agent that plans what to test next,
keeps memory outside its context window, and calls tools rather than
reasoning freeform over a state dump. Evaluated against two non-LLM
baselines that represent what a studio already gets for free from scripted
QA. 

## Why this project

See [docs/scenario-matrix.md](docs/scenario-matrix.md) for the design
rationale behind the six scenarios, and [ANSWER_KEY.md](ANSWER_KEY.md)
(generated, not hand-written -- see below) for the ten seeded defects. The
short version: most of a combat system's real bugs live at the seams
between independently-simple systems -- a shield that doesn't know it
should ignore damage-over-time, two stacking effects combined the wrong
way, a percentage effect computed against the wrong base. An LLM agent
that plans what to try next, remembers what it's already tried, and calls
tools instead of parsing raw JSON is a natural fit for finding exactly
that kind of bug.

## Setup

```bash
python -m pip install pydantic openai pytest python-dotenv
cp .env.example .env   # then fill in DEEPSEEK_API_KEY yourself; .env is gitignored
bash scripts/install-hooks.sh   # installs a pre-commit hook that blocks committed API keys
```

## Running the tests

```bash
python -m pytest -v
```

33 tests, all green, no API calls (nothing in `engine/` touches the
network). They fall into four files:

- `tests/test_engine_golden.py` -- hand-calculated values against clean
  mode (all ten defect flags off). If these fail, the engine's basic
  semantics are wrong.
- `tests/test_defects.py` -- one test per seeded defect: enabling exactly
  that flag changes the trajectory in the documented direction.
- `tests/test_invariants.py` -- across a scenario x seed grid, clean mode
  never trips the invariant checker, and each of the six invariant-visible
  defects trips it somewhere.
- `tests/test_no_bug_leakage.py` / `tests/test_no_secrets.py` --
  methodological and secret-hygiene guards, described below.

## Architecture (what exists so far)

```
engine/
  models.py       Pydantic state: Character, Ability, StatusEffect, BattleState, Action
  content.py       Ability and character library (data only)
  elements.py       Fire > Ice > Lightning > Fire, x1.5 / x1.0 / x0.67
  effects.py        Status application, stacking rules, tick logic -- most defects live here
  resolution.py      The fixed turn-resolution pipeline (see its docstring for the exact order)
  rules.py           Legal-move checker: energy, cooldown, valid targeting
  defects.py         The ten seeded defects, as switches (all default False)
  invariants.py      Post-action invariant checks, independent of defect flags
  scenarios.py       The six hand-built scenarios, seeded for bounded variety
  engine.py          BattleEngine facade: reset / legal_actions / take_action / state / log
```

**State is plain and serializable.** `BattleEngine.state_dict()` /
`log_dicts()` dump the whole battle (or its turn-by-turn history) to JSON
via Pydantic. Nothing in `engine/` holds unserializable state except the
`BattleEngine` object itself (its internal bookkeeping for what turn-slot
is pending).

**The turn pipeline is fixed and documented once,** in
`resolution.py`'s module docstring -- every other module defers to it.
Most of the ten defects are violations of this exact ordering.

**The agent never needs to submit a no-op.** `BattleEngine` auto-resolves
any turn-slot that doesn't need a real decision (a dead character's turn,
a stunned character's turn, an enemy with zero legal actions -- see defect
B08) and only stops to ask for an action when a choice actually matters.

## The ten seeded defects

Kept behind boolean switches in `engine/defects.py` (all default `False`),
not baked into the engine's only code path -- this is what lets
`tests/test_engine_golden.py` assert *correct* semantics against clean
mode instead of having to encode the bugs, and what lets
`scripts/generate_answer_key.py` regenerate
[ANSWER_KEY.md](ANSWER_KEY.md) from the flags' own docstrings instead of a
hand-maintained doc drifting from the code. Six of the ten are visible to
the invariant checker on their own; four are not -- see ANSWER_KEY.md for
which is which, and RESULTS.md (once it exists) for why that four-bug gap
is the part of this project actually worth measuring.

## Methodological guards

- **`docs/scenario-matrix.md` was committed before `engine/defects.py`
  exists**, and `tests/test_no_bug_leakage.py` checks that ordering
  against git history -- so the scenarios can't have been reverse
  engineered from the answer key.
- **The same test file** also statically checks that nothing under
  `agent/` or `baselines/` imports `engine.defects` or `oracle/`, or
  contains any of the defect IDs or `DefectFlags` field names anywhere in
  source (including string literals, which covers prompt templates).
- **`tests/test_no_secrets.py`** scans the working tree for anything
  shaped like an API key, and (when `DEEPSEEK_API_KEY` is set) for the
  live key's literal value. `scripts/install-hooks.sh` installs a
  pre-commit hook doing the staged-diff version of the same check.

## A note on process

Two real engine bugs (not seeded ones) were caught during Week 1 by the
test suite itself, not by manual review:

1. `BattleEngine.take_action` never called the pipeline's end-of-turn
   step, so cooldowns, status durations, and the turn pointer never
   advanced after a real decision -- caught by a smoke test noticing the
   same character kept acting every turn.
2. An ability that both deals lethal damage and applies a status (e.g.
   Cinder Burn: damage + Burn) re-attached that status to the target
   *after* it had just died, because the status-application step wasn't
   gated on the target still being alive -- caught by a 300-battle random
   regression sweep in clean mode showing invariant violations that
   should have been impossible.

## Roadmap

- R1 -- differential oracle (`oracle/`), episode JSONL logging,
  the replay HTML generator, the LLM provider layer + smoke test, the tool
  interface, a bare single-turn agent with no memory yet.
- R2 -- the exploration ledger (SQLite), the outer planning loop,
  `flag_anomaly` wired to the invariant checker, a first bug report.
- R3 -- both baselines (Random, Systematic), repo goes public
  (after the secret-hygiene checklist), application submitted.
- R4 -- full evaluation sweep across all four methods, metrics,
  charts.
- R5 -- write-up, video, stretch goals or buffer.

Developed with assistance from Claude Code. 
