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
python -m pip install pydantic openai pytest python-dotenv jinja2 numpy
python -m pip install torch        # optional: only for the enemy AI (enemy_ai/, scripts/play.py)
bash scripts/install-hooks.sh      # installs a pre-commit hook that blocks committed API keys
```

Then copy `.env.example` to `.env` (gitignored) and fill in
`DEEPSEEK_API_KEY=`. Alternatively, fill in with your API.

## Running things

```bash
python -m pytest -q                  # the whole suite: offline, no API calls, a few seconds
python -m agent.llm --smoke          # one live tool-calling round trip: key, model, cost accounting
python -m eval.pilot --run-name pilot --episodes 25 --max-spend 0.75   # a full-agent campaign + bug report
python -m eval.pilot --run-name greedy --method greedy_llm --episodes 25 # the Greedy-LLM baseline
python -m eval.random_baseline --run-name random --actions 800          # the random baseline (no API calls)
python -m eval.bug_report logs/runs/pilot                              # re-grade a finished campaign
python -m replay.generate logs/runs/pilot/episodes/ep_0001.jsonl out.html
python scripts/play.py S1                                              # play a battle yourself against the enemy AI
python -m enemy_ai train                                               # retrain the enemy AI (~200 s on one CPU thread)
python -m enemy_ai check --greedy                                      # enemy AI win rate vs random play
```

Every entry point that can spend money takes `--max-spend` (a cap on that
run's spend). A campaign stopped by its cap can be continued with
`--resume` under the same run name. Campaigns record every request/response
pair to `fixtures/` by default (`--mode record`); `--mode replay` re-runs
them offline at zero cost.

## Architecture

```
engine/         the game: pure, deterministic, serializable
  models.py       Pydantic state: Character, Ability, StatusEffect, BattleState, Action
  content.py      Ability and character library (data only)
  elements.py     Fire > Ice > Lightning > Fire, x1.5 / x1.0 / x1/1.5
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

baselines/      comparison methods -- same leakage rule as agent/
  random_explorer.py  Uniform-random legal moves for both teams, no LLM
  greedy_llm.py       The full agent with its planner and ledger readback switched off

enemy_ai/       the game's own enemy AI, for human play -- no part in bug testing
  env.py          Fixed-shape observation/action encoding and a self-play training env
  policy.py       PyTorch actor-critic, PPO self-play training, strength check vs random
  enemy_policy.pt The trained checkpoint

oracle/         grading only -- never visible to the agent
  differential.py Replay a trace under another build; first point of divergence
  attribute.py    Single-flag ablation: which defect explains a single-defect trace
  triggers.py     Leave-one-out triggers and one-step activity for full-build traces

eval/
  pilot.py        Builds the buggy build (every seeded defect) and runs an LLM campaign against it
                  (full agent or Greedy-LLM)
  random_baseline.py  The same, for the random baseline
  bug_report.py   Grades a campaign: triggered vs detected per defect, triage for every flag

scripts/
  play.py         Play a battle by hand through the agent's own tool surface; the enemy AI
                  plays the other side (--manual-enemy to play both)

replay/
  generate.py     Episode JSONL -> self-contained HTML replay (Jinja2)
```

**State is plain and serializable:** `BattleEngine.state_dict()` /
`log_dicts()` dump the whole battle (or its turn-by-turn history) to JSON
via Pydantic. Nothing in `engine/` holds unserializable state except the
`BattleEngine` object itself (its internal bookkeeping for what turn-slot
is pending).

**Turn order is speed-driven:** Each character has a `speed`
stat and an action value, AV = 10000 / speed; whoever holds the lowest AV
acts next, so a fast character acts more often than a slow one over the
same stretch of battle. Ties break by each scenario's
declared order (`p1, e1, p2, e2, p3, e3`). 100 AV of elapsed clock is one
cycle. `BattleState.turn_forecast()` gives the
agent (and `scripts/play.py`) a look-ahead at who acts next, for setting
up multi-character interactions deliberately rather than by luck.

**Three methods:** The full agent (planner + ledger +
per-turn loop), **Greedy-LLM** (the same per-turn loop, prompts, tools and
gating, with the planner and ledger readback switched off), and a
**random** explorer. All three drive both teams through the same engine,
write the same ledger and episode logs, and are graded by the same oracle
and bug report, so any difference between them comes from the method.

**An enemy AI for human play:** `enemy_ai/` is a small PPO policy trained
by PyTorch on the clean build, so it learns the game as designed rather
than the seeded bugs. It plays the enemy team in `scripts/play.py`. It
takes no part in bug testing, where the tested method controls both
teams.

**The agent never needs to submit a no-op:** `BattleEngine` auto-resolves
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
which is which, and the results below for why that five-bug gap is the
part of this project actually worth measuring.

## Results

One pilot campaign per method, all against the full build (every defect
enabled at once) with `deepseek-flash`. A defect counts as
**triggered** in an episode when the oracle's leave-one-out replay shows
it changed that trace, and as **detected** when the invariant checker
reported it or an agent flag strongly matched it (the flag names a
character the defect affected, and its wording fits the defect).

|                                         | Random    | Greedy-LLM   | Full agent   |
| --------------------------------------- | --------- | ------------ | ------------ |
| Episodes / actions                      | 21 / 844  | 25 / 751     | 26 / 871     |
| Model spend                             | $0        | $0.34        | $0.96        |
| Defects triggered                       | 11 / 11   | 11 / 11      | 11 / 11      |
| Defects detected                        | 5 / 11    | 11 / 11      | 11 / 11      |
| Invariant-invisible defects detected    | **0 / 5** | **5 / 5**    | **5 / 5**    |
| Agent flags (strong / weak / unmatched) | --        | 200 / 2 / 45 | 186 / 3 / 68 |
| LLM calls (planner / per-turn)          | --        | 0 / 709      | 246 / 861    |
| Prompt cache hit rate                   | --        | 68.3%        | 81.0%        |

Per defect, episodes where it was detected / episodes where it was triggered:

| Defect                                     | Invariant-visible | Random  | Greedy-LLM | Full agent |
| ------------------------------------------ | ----------------- | ------- | ---------- | ---------- |
| B01 Shield absorbs Poison/Burn ticks       | no                | 0 / 1   | 2 / 4      | 3 / 5      |
| B02 Burn stacks multiply instead of add    | no                | 0 / 5   | 7 / 8      | 5 / 6      |
| B03 Chill drives energy negative           | yes               | 0 / 4   | 4 / 4      | 3 / 4      |
| B04 Poison computed off Weaken-adjusted HP | no                | 0 / 3   | 3 / 4      | 3 / 3      |
| B05 Stun duplicates instead of refreshing  | yes               | 2 / 2   | 1 / 1      | 5 / 5      |
| B06 Cooldown drains twice per turn         | yes               | 13 / 21 | 15 / 25    | 19 / 25    |
| B07 Regen survives death                   | yes               | 4 / 4   | 8 / 8      | 6 / 6      |
| B08 Enemy has no Basic Attack fallback     | yes               | 2 / 3   | 3 / 4      | 6 / 7      |
| B09 Elemental multiplier applied twice     | no                | 0 / 21  | 25 / 25    | 24 / 25    |
| B10 Zero-pool Shield never removed         | yes               | 5 / 5   | 15 / 15    | 10 / 10    |
| B11 Turn-order ties broken by id           | no                | 0 / 21  | 1 / 24     | 5 / 25     |

**Random play reaches every defect but misses six of them.** It
triggered all eleven, yet with only the invariant checker to report
them it detected none of the five that produce wrong numbers rather than
impossible states (nor B03, see below). Both LLM methods detected all five, by checking each
turn's observed numbers against the rules and the abilities' declared
data.

**What the planner changes.** The full agent's planner chose scenarios
and set up specific interactions, for example turn-order ties, where
B11 was matched in 5 episodes against 1 for Greedy-LLM. It accounts for
most of the full agent's cost (246 planner calls over 26 episodes).

Full reports, with every flag and a replay link per episode:
`results/{random,greedy_llm,full_agent}/bug_report.md` (generated locally;
`results/` is gitignored).

## Methodological guards

- **`docs/scenario-matrix.md` was committed before `engine/defects.py`
  exists**, and `tests/test_no_bug_leakage.py` checks that ordering
  against git history, so the scenarios can't have been reverse
  engineered from the answer key.
- **The same test file** also statically checks that nothing under
  `agent/`, `baselines/` or `enemy_ai/` imports `engine.defects` or `oracle/`, or
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
- ~~R3~~ -- the baselines (Random, and Greedy-LLM: the same agent with ledger
  and planner switched off), pilot runs of all three methods.

Developed with assistance from Claude Code.
