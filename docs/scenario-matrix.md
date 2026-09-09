# Scenario matrix

This document is written before `engine/defects.py` exists, and that ordering
is deliberate: it is the evidence (visible in git history) that these
scenarios were designed to exercise combat-system interaction axes in
general, not reverse-engineered to hit a specific answer key. See
`tests/test_no_bug_leakage.py` for the automated side of that guard.

## Why a matrix and not "six scenarios that seem fun"

A turn-based tactics combat system's riskiest code is not any single
ability — it's what happens where two independent systems compose:
a shield absorbing the wrong kind of damage, two stacking effects from
different sources, a percentage effect computed against the wrong base,
a control effect colliding with a resource-gate effect, a status outliving
the character it was attached to. Real defects live at these seams. Six
scenarios were chosen to each foreground one seam, using natural party/enemy
archetypes rather than contrived setups, so any given scenario also serves
as a normal playable battle.

## The interaction axes

| Axis | What it stresses |
|---|---|
| Shield × damage-over-time | Does a shield correctly ignore DoT ticks and only intercept direct hits? |
| Multi-source stacking | When two sources apply the same additive effect, does the tick logic combine them correctly? |
| Energy denial | Does energy regeneration under a debuff stay bounded, even when denial stacks from multiple sources? |
| Percentage damage × modifiers | Does a %-of-max-HP effect compute against the target's own baseline, independent of what other modifiers are active? |
| Turn-skip × cooldown timing | Does a control effect that skips a turn interact correctly with cooldown bookkeeping and refresh-not-stack semantics? |
| Death × persistent healing | Do periodic effects get cleared on death, so a revived character doesn't inherit stale state? |

## The six scenarios

| # | Name | Party | Enemies | Axis under test |
|---|---|---|---|---|
| S1 | Bulwark | A shield-granting support + two sustained-damage dealers | Two Fire-heavy DoT appliers | Shield × DoT |
| S2 | Pyroclasm | A balanced 3-character party | Two independent Fire attackers focusing one high-HP target | Multi-source stacking |
| S3 | Deep Freeze | High-energy-cost casters | Ice-heavy control enemies applying energy denial | Energy denial |
| S4 | Attrition | A near-mirror match with Poison and Weaken appliers on both sides | Same | %-damage × modifiers |
| S5 | Lockdown | Low-cooldown, frequent-acting party | Lightning stun-lock enemies with tight ability cooldowns | Turn-skip × cooldown |
| S6 | Last Stand | A Regen/support-heavy party with one revive-capable ability | A single hard-hitting burst enemy | Death × persistent healing |

Each scenario is defined once in `engine/scenarios.py` as a fixed
composition, and takes a `seed` that perturbs starting HP, energy, and
cooldown offsets within documented bounds — enough breadth for repeated
exploration runs without needing an open-ended world.

## Non-goals

This matrix is not exhaustive over the full space of effect combinations —
that space is combinatorially large and most of it is uninteresting. It is
deliberately narrow: six axes, one scenario each, chosen because each maps
to a class of bug that recurs in real combat-system code, not because a
specific defect needed a home.
