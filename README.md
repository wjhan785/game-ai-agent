# game-ai-agent

A headless turn-based tactics combat engine with 11 deliberately seeded
defects, built to be hunted by an LLM agent that plans what to test next,
keeps memory outside its context window, and calls tools rather than
reasoning freeform over a state dump.

See [docs/scenario-matrix.md](docs/scenario-matrix.md) for the design
rationale behind the six scenarios, and [ANSWER_KEY.md](ANSWER_KEY.md)
(generated) for the eleven seeded defects.

## Setup

```bash
python -m pip install pydantic openai pytest python-dotenv jinja2
bash scripts/install-hooks.sh   # installs a pre-commit hook that blocks committed API keys
```

Then copy `.env.example` to `.env` (gitignored) and fill in
`DEEPSEEK_API_KEY=`. Alternatively, fill in with your API.

## Running things

```bash
python -m pytest -q                  # the whole suite: offline, no API calls, a few seconds
python -m agent.llm --smoke          # one live tool-calling round trip: key, model, cost accounting
python -m eval.pilot --run-name pilot --episodes 25 --max-spend 0.75   # a full-agent campaign + bug report
python -m eval.bug_report logs/runs/pilot                              # re-grade a finished campaign
python -m replay.generate logs/runs/pilot/episodes/ep_0001.jsonl out.html
```

Every entry point that can spend money takes `--max-spend` (a cap on that
run's spend). Campaigns record every request/response pair to `fixtures/` by default
(`--mode record`); `--mode replay` re-runs them offline at zero cost.

## Architecture

```
engine/         the game: pure, deterministic, serializable
  models.py       Pydantic state: Character, Ability, StatusEffect, BattleState, Action
  content.py      Ability and character library (data only)
  elements.py     Fire > Ice > Lightning > Fire, x1.5 / x1.0 / x0.67
  effects.py      Status application, stacking rules, tick logic -- most defects live here
  resolution.py   The per-turn-slot resolution pipeline (fixed steps; see its docstring) and the
                  action-value scheduler advance (whose turn is next is speed-driven, not fixed)
  rules.py        Legal-move checker: energy, cooldown, valid targeting
  defects.py      The eleven seeded defects, as switches (all default False)
  invariants.py   Post-action invariant checks, independent of defect flags
  scenarios.py    The six hand-built scenarios (each with its own per-character speeds), seeded
                  for bounded variety
  engine.py       BattleEngine facade: reset / legal_actions / take_action / pending_record / log
  episode_log.py  Episode JSONL: header (run metadata + initial state), one line per turn, footer

agent/          the LLM agent -- may not import engine.defects or oracle/ (enforced by a test)
  llm.py          DeepSeek via the OpenAI SDK: forced single tool call, validate/retry, cache-aware
                  cost accounting, spend caps, off-peak gate, cassette record/replay
  prompts.py      What the model sees: the rules spec, rosters, compact per-turn observations
  tools.py        The tool surface for both loops, and the dispatcher around one BattleEngine
  ledger.py       SQLite external memory: episodes, hypotheses, flags, visits, coverage
  inner_loop.py   Per-turn tactical decisions toward the episode goal; flags anomalies
  runner.py       One episode: LLM-call gating, scripted turns, ledger bookkeeping
  outer_loop.py   Per-episode planner and the campaign driver

oracle/         grading only -- never visible to the agent
  differential.py Replay a trace under another build; first point of divergence
  attribute.py    Single-flag ablation: which defect explains a single-defect trace
  triggers.py     Leave-one-out triggers and one-step activity for full-build traces

eval/
  pilot.py        Builds the buggy build (every seeded defect) and runs a campaign against it
  bug_report.py   Grades a campaign: triggered vs detected per defect, triage for every flag

replay/
  generate.py     Episode JSONL -> self-contained HTML replay (Jinja2)
```

**State is plain and serializable.** `BattleEngine.state_dict()` /
`log_dicts()` dump the whole battle (or its turn-by-turn history) to JSON
via Pydantic. Nothing in `engine/` holds unserializable state except the
`BattleEngine` object itself (its internal bookkeeping for what turn-slot
is pending).

**The turn pipeline is fixed and documented once,** in
`resolution.py`'s module docstring -- every other module defers to it.
Most defects are violations of this exact ordering.

**Turn order is speed-driven,** Each character has a `speed`
stat and an action value, AV = 10000 / speed; whoever holds the lowest AV
acts next, so a fast character acts more often than a slow one over the
same stretch of battle -- not just earlier. Ties break by each scenario's
declared order (`p1, e1, p2, e2, p3, e3`). 100 AV of elapsed clock is one
cycle, so `turn_cap` keeps counting "rounds" exactly as before. This
matters beyond flavor: a fast character burns through a debuff's declared
duration in less wall-clock time than a slow one carrying the same
status, and gets its cooldowns back sooner -- exactly the kind of seam
this project's bugs live in. `BattleState.turn_forecast()` gives the
agent (and `scripts/play.py`) a look-ahead at who acts next, for setting
up multi-character interactions deliberately rather than by luck.

**The agent never needs to submit a no-op.** `BattleEngine` auto-resolves
any turn-slot that doesn't need a real decision (a dead character's turn,
a stunned character's turn, an enemy with zero legal actions -- see defect
B08) and only stops to ask for an action when a choice actually matters.

## The eleven seeded defects

Kept behind boolean switches in `engine/defects.py` (all default `False`),
not baked into the engine's only code path. This is what lets
`tests/test_engine_golden.py` assert _correct_ semantics against clean
mode instead of having to encode the bugs, and what lets
`scripts/generate_answer_key.py` regenerate
[ANSWER_KEY.md](ANSWER_KEY.md) from the flags' own docstrings instead of a
hand-maintained doc drifting from the code. The eleventh, B11, seeds a
wrong tie-break in the new speed scheduler (ties resolved by sorting
character id instead of the scenario's declared order), graded the same
way as every other defect: `oracle/triggers.py` treats it as a
construction-time defect, like B08, since it is decided when the battle
is built, not inside any one step. Six of the eleven are visible to
the invariant checker on their own; five are not. See ANSWER_KEY.md for
which is which, and RESULTS.md (once it exists) for why that five-bug gap
is the part of this project actually worth measuring.

## Methodological guards

- **`docs/scenario-matrix.md` was committed before `engine/defects.py`
  exists**, and `tests/test_no_bug_leakage.py` checks that ordering
  against git history, so the scenarios can't have been reverse
  engineered from the answer key.
- **The same test file** also statically checks that nothing under
  `agent/` or `baselines/` imports `engine.defects` or `oracle/`, or
  contains any of the defect IDs or `DefectFlags` field names anywhere in
  source (including string literals, which covers prompt templates).
- **`tests/test_no_secrets.py`** scans the working tree for anything
  shaped like an API key, and (when `DEEPSEEK_API_KEY` is set) for the
  live key's literal value. `scripts/install-hooks.sh` installs a
  pre-commit hook doing the staged-diff version of the same check.

## Roadmap

- ~~R1~~ -- differential oracle, episode JSONL logging, the replay HTML
  generator, the LLM provider layer + smoke test, the tool interface, a
  bare single-turn agent.
- ~~R2~~ -- the exploration ledger (SQLite), the outer planning loop,
  `flag_anomaly` wired to the invariant checker, a first bug report.
- ~~R3~~ -- the baselines (Random, Systematic, and Greedy-LLM: the same agent
  with ledger and planner switched off), repo goes public (after the
  secret-hygiene checklist), application submitted.
- R4 -- full evaluation sweep across all four methods, metrics,
  charts.
- R5 -- write-up, video, stretch goals or buffer.

Developed with assistance from Claude Code.
