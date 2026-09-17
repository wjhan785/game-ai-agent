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
(generated, not hand-written -- see below) for the eleven seeded defects. The
short version: most of a combat system's real bugs live at the seams
between independently-simple systems -- a shield that doesn't know it
should ignore damage-over-time, two stacking effects combined the wrong
way, a percentage effect computed against the wrong base. An LLM agent
that plans what to try next, remembers what it's already tried, and calls
tools instead of parsing raw JSON is a natural fit for finding exactly
that kind of bug -- and the project only means something if it's measured
honestly against what a much simpler script would already find.

## Setup

```bash
python -m pip install pydantic openai pytest python-dotenv jinja2
bash scripts/install-hooks.sh   # installs a pre-commit hook that blocks committed API keys
```

Then create `.env` (gitignored) with `DEEPSEEK_API_KEY=` filled in.
Nothing reads the key from anywhere else -- no literal default, no CLI
argument.

## Running things

```bash
python -m pytest -q                  # the whole suite: offline, no API calls, a few seconds
python -m agent.llm --smoke          # one live tool-calling round trip: key, model, cost accounting
python -m eval.pilot --run-name pilot --episodes 25 --max-spend 0.75   # a full-agent campaign + bug report
python -m eval.bug_report logs/runs/pilot                              # re-grade a finished campaign
python -m replay.generate logs/runs/pilot/episodes/ep_0001.jsonl out.html
```

Every entry point that can spend money takes `--max-spend` (a cap on that
run's spend), refuses DeepSeek's peak-pricing windows unless given
`--allow-peak`, and enforces the project's $15 ceiling regardless.
Campaigns record every request/response pair to `fixtures/` by default
(`--mode record`); `--mode replay` re-runs them offline at zero cost.

## The tests

All offline. The engine tests (hand-calculated golden values in clean
mode, one test per seeded defect, the invariant grid) pin the game's
semantics; the oracle tests check that replay reproduces traces exactly
and that the two independent grading procedures -- whole-trajectory
leave-one-out and one-step counterfactuals -- agree on when every defect
first acted, in every scenario; the agent tests drive the planner, the
tactical loop and whole campaigns through hand-built cassettes keyed on
the real request construction, so they exercise the actual wiring rather
than mocks. `test_no_bug_leakage.py` and `test_no_secrets.py` are the
guards described below.

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

**Turn order is speed-driven, not fixed.** Each character has a `speed`
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
not baked into the engine's only code path -- this is what lets
`tests/test_engine_golden.py` assert *correct* semantics against clean
mode instead of having to encode the bugs, and what lets
`scripts/generate_answer_key.py` regenerate
[ANSWER_KEY.md](ANSWER_KEY.md) from the flags' own docstrings instead of a
hand-maintained doc drifting from the code. The eleventh, B11, seeds a
wrong tie-break in the new speed scheduler (ties resolved by sorting
character id instead of the scenario's declared order) -- graded the same
way as every other defect: `oracle/triggers.py` treats it as a
construction-time defect, like B08, since it is decided when the battle
is built, not inside any one step. Six of the eleven are visible to
the invariant checker on their own; five are not -- see ANSWER_KEY.md for
which is which, and RESULTS.md (once it exists) for why that five-bug gap
is the part of this project actually worth measuring.

## How the agent works

A campaign runs against one build with every seeded defect enabled at once --
the way QA meets a real build, with nobody saying which bugs are in it.
The agent drives both teams: it is a test harness, not a player.

**Two loops, one memory.** The *outer* loop runs at every episode
boundary. It reads a digest of the SQLite ledger -- which pairs of
statuses have never been seen on one character, which hypotheses are open,
what was flagged, what the last episode did -- then works through tools:
`ledger_write` to mark the last episode's hypotheses confirmed, refuted or
inconclusive (with evidence) and to open new ones, `ledger_read` for
detail, and `reset_episode` to start the next episode with a scenario, a
concrete turn-by-turn goal, the characters to focus on, and the hypotheses
under test. The *inner* loop plays each turn toward that goal, and calls
`flag_anomaly` when what it observes contradicts the rules or an
ability's declared data. Every flag carries a reproduction file (scenario,
seed, every action so far), so any report can be replayed to the exact
state it was raised in.

**Deterministic code does the arithmetic; the model does the judgment.**
The model is given the game's design spec (`agent/prompts.py`: every
mechanic at the same level of detail -- the document a QA tester tests
against) and, each turn, compact observations that pair declared numbers
with observed ones: an ability's tooltip damage next to the damage taken
and the shield absorbed, the statuses a character held next to the ticks
they produced. Whether a gap is legitimate (Weaken, Shield, elemental
disadvantage) or a contradiction is the model's call.

**Verify, then act.** The first live shakedown showed the failure mode
plainly: the agent set up exactly the right interactions ("apply a second
Burn to test stacking") and then never checked the numbers they produced.
So `take_action`'s first field is now a required `audit`: before choosing
a move, the model checks every turn-slot played since its last decision
against a generic checklist covering every mechanic (direct hits, ticks,
energy, statuses, cooldowns, turn order), one line each, ok or MISMATCH;
every mismatch goes in `anomalies` and is filed as a flag. The audit is
logged and shown in the replay, so what the agent checked -- not just what
it did -- is inspectable.

The next two shakedowns were about precision rather than recall. Once the
agent audited, it found the doubled elemental multiplier -- but two thirds
of its flags were false, and every false one traced to a spec or an
observation that left room to misread the timeline: energy flagged as
"regen not applied" on characters already at their cap, a cooldown
"one lower than declared" at the end of the turn it was set (the countdown
includes that turn), a disadvantage matchup read as neutral because the
spec only listed who beats whom. None of those were fixed by telling the
model to be careful. They were fixed by making the spec say the timeline
outright, writing out all six elemental matchups, and handing over energy
as `start + regen - spent = end` instead of two snapshots. The model's job
stayed judgment; the observation's job became leaving nothing to
reconstruct.

**Combinatorics is computed, not reasoned.** The same shakedown had the
planner confidently misread the rosters (claiming a scenario could put
Poison and Shield on one character; none can). Which status pairs each
scenario can actually produce now comes from the abilities' declared
target types, and the coverage digest splits never-seen pairs into
"reachable, and where" versus "unreachable".

**Only informative turns reach the model.** A turn goes to the LLM if its
(lossy) state signature has never been visited, or its character is one
the plan focuses on; every other turn takes a seeded scripted action, so a
replay is exact. Budgets across methods are matched on actions taken, and
LLM calls are reported separately.

**Request layout is a cost decision.** DeepSeek caches shared prompt
prefixes automatically, and a cache hit costs ~50x less than a miss. Each
inner request is ordered stable-first -- the system prompt (identical on
every call), the episode brief (identical within an episode), then a
bounded window of recent turns, the current turn, the state and the legal
moves. Cache-hit tokens are logged per episode, so a silent prefix
invalidator shows up as a number rather than a bill.

**Failures are metrics, not crashes.** Every model response must be one
tool call that validates against a Pydantic schema; on a validation error
the error is fed back and the call retried twice, then the turn falls back
to the first legal action and the fallback is counted. The same holds for
the planner: a plan that names an unknown scenario or hypothesis is
rejected with the reason, and a planning phase that never produces a valid
plan falls back to the least-explored scenario.

**Grading is separate from playing.** After a campaign, `eval/bug_report.py`
replays every episode through the oracle: a defect *triggered* if
replaying the trace without it diverges; it was *active* on every step
where re-running that step from its logged state without it changes the
outcome. Each agent flag is then triaged against the defects active in the
steps the agent could see when it raised the flag. The agent never sees
any of this -- the oracle grades, the agent plays.

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

Both are exactly the kind of thing the project's own thesis says an
automated invariant checker is for. Left in this README rather than
quietly fixed and forgotten, because it's a fair example of the
process working as intended before any LLM was involved.

A third turned up in Week 3, while building what the agent observes:
every hit whose damage came in under the ability's base power -- an
elemental disadvantage, a Weakened attacker -- was recorded as *shield
absorption*, even on targets with no shield. Game state was right, so no
invariant or golden test could see it; only the per-hit record was wrong.
But that record is what the agent reads, so it would have generated false
reports on every disadvantaged hit. Absorption is now measured across the
absorption call itself, with a regression test. It's the same lesson as
the four invariant-invisible defects, one level up: wrong numbers in a
legal state are the hard ones.

A fourth showed up designing the speed-based scheduler that replaced the
fixed turn order: the differential oracle's whole correctness argument
(`oracle/differential.py`) rests on comparing `TurnRecord` logs
positionally, which was safe under a fixed order because the schedule
never depended on game state. A speed-driven schedule does -- so the
argument only still holds if each character's action value is itself
part of what gets diffed. It was nearly left out of `CharacterSnapshot`
(hp/energy/alive/statuses/cooldowns felt like the complete list); adding
it is what turns "identical prefixes imply the same next actor" from an
assumption into something the diff actually checks. The companion fix is
in `oracle/triggers.py`'s per-step reconstruction, which used to *derive*
the scheduler pointer from the recorded actor id (so it could never
disagree); restoring real action values instead means the reconstruction
can select a different actor than the log recorded, and now asserts
agreement rather than assuming it, failing into `inconsistent_steps`
instead of silently grading the wrong character.

## Roadmap

- ~~R1~~ -- differential oracle, episode JSONL logging, the replay HTML
  generator, the LLM provider layer + smoke test, the tool interface, a
  bare single-turn agent.
- ~~R2~~ -- the exploration ledger (SQLite), the outer planning loop,
  `flag_anomaly` wired to the invariant checker, a first bug report.
- R3 -- the baselines (Random, Systematic, and Greedy-LLM: the same agent
  with ledger and planner switched off), repo goes public (after the
  secret-hygiene checklist), application submitted.
- R4 -- full evaluation sweep across all four methods, metrics,
  charts.
- R5 -- write-up, video, stretch goals or buffer.