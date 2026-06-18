# Catcher Reward Shaping

The engine emits the same sparse `±1` terminal signal for the catcher as it
does for the runner: `+1` when the catcher wins (kill by step, kill by
projectile, or timeout with the runner below `SPECIAL_MAJORITY`), `-1`
otherwise, `0` on every non-terminal step. Unlike the runner, the catcher's
terminal events can happen mid-episode (a kill terminates immediately on
contact), so the credit-assignment problem is shorter and a naive sparse
catcher will reliably learn the *stepping* part of the game.

What the sparse signal does **not** teach is the catcher's actual strategic
job: **area defense**. The bundled heuristic catcher (`agents/heuristic.py`)
scores every action by the resulting Chebyshev distance to the runner —
pure chase, projectiles ignored, no awareness of which specials are
contested. A sparse RL catcher trained against the heuristic runner will
likely arrive at a similar policy, because random shots usually miss and
sparse expectation marks them as wasted turns. The shaping signals below
exist specifically to teach the catcher that

1. **a well-aimed bullet is worth firing** even when it doesn't kill — it
   denies a special the runner is one move from grabbing,
2. **interposing between the runner and contested specials** beats greedy
   chase as the *default* behavior — closing only matters when the runner is
   in striking range, and
3. **the runner's sprint charges are a depletable resource** — when they're
   spent, cornering the runner becomes a closing trap rather than a
   reversible position.

`CatcherRewardShaper` lives alongside `RunnerRewardShaper` in
`rl_agent/reward_shaping.py`. Both subclass a shared `RewardShaper` base
that owns the `_cheb` helper and the public `shape(prev, curr, base)`
method (which short-circuits on `curr.terminated` and otherwise delegates
to the subclass's `_compute`). The environment picks the right shaper at
construction time based on `trainee_role`.

## Tunables (class attributes)

| Attribute                       | Value  | What it controls                                                                                       |
|---------------------------------|--------|--------------------------------------------------------------------------------------------------------|
| `SPECIAL_DEFENSE_COEFF`         | 0.10   | One-shot reward when a freshly fired bullet's ray covers an imminently-capturable special.             |
| `SPECIAL_DEFENSE_NEAR_MAX`      | 3      | Inclusive max Chebyshev distance from catcher along the shoot ray for the "runner reaches in 1" check. |
| `SPECIAL_DEFENSE_FAR_MIN`       | 4      | Inclusive min Chebyshev distance along the shoot ray for the "runner reaches in 2" check.              |
| `SPECIAL_DEFENSE_FAR_MAX`       | 5      | Inclusive max Chebyshev distance along the shoot ray for the "runner reaches in 2" check.              |
| `BULLET_SPAM_PENALTY`           | 0.10   | Flat penalty applied on the shoot turn when the new bullet's ray fails both reach checks.              |
| `SPECIAL_BLOCKING_NEAREST`      | 0.10   | Coefficient on the blocking score for the runner's closest remaining special.                          |
| `SPECIAL_BLOCKING_SECOND`       | 0.05   | Coefficient on the blocking score for the runner's second-closest remaining special.                   |
| `TIME_ADVANTAGE_COEFF`          | 0.005  | Per-step coefficient mirroring the runner's `URGENCY_COEFF` with opposite sign.                        |
| `CHASE_COEFF`                   | 0.20   | Numerator of the heavy close-range proximity bonus (catcher within `DANGER_RADIUS` and closing).       |
| `PROXIMITY_COEFF`               | 0.015  | Numerator of the light ambient proximity bonus (everywhere else).                                      |
| `CORNERING_COEFF`               | 0.015  | Numerator of the steady cornering bonus, scaled by `corner_score · sprint_pressure`.                   |
| `CORNER_SPRINT_BOOST`           | 0.5    | Multiplier on the sprint-pressure ramp. At zero sprints, cornering pays `1 + 0.5 = 1.5×` its base.     |
| `DANGER_RADIUS`                 | 2      | Chebyshev radius inside which the chase branch fires; symmetric with the runner's `DANGER_RADIUS`.     |

Distances are **Chebyshev** throughout, matching the engine's 8-directional
movement (one step in any direction equals one unit of Chebyshev distance).

## The reward function in one expression

For a catcher trainee, given `prev` (state before the trainee's action),
`curr` (state after the trainee's action *and* the opponent's response —
i.e. `self.state` inside `env.step()`), and the engine's `base` reward:

```
shaped = base
       + _special_defense_bonus(prev, curr)         # one-shot per fired bullet
       + _special_blocking_attraction(curr)         # area-defense positioning
       + _time_advantage_bonus(curr)                # stall-pressure mirror
       + _chase_bonus(prev, curr)                   # proximity + cornering
```

Every term short-circuits cleanly to `0` (or near-zero) when its
preconditions don't apply, so the per-step magnitude breakdown below is the
right way to read the calibration.

## Component by component

### Special-defense bonus / bullet-spam penalty — `_special_defense_bonus(prev, curr)`

This is the *only* signal that fires on bullet firing, and it fires
**exactly once per bullet** — on the turn the bullet is spawned. There is
no ongoing reward for in-flight bullets, by design: a defensive shot is a
decision made at firing time, and the credit should be assigned to that
decision rather than smeared across every subsequent turn the bullet
happens to remain in flight.

```
new_bullet = _newly_fired_bullet(prev, curr)
if new_bullet is None:
    return 0.0
(_, (dx, dy)) = new_bullet
(cx, cy) = prev.catcher_pos
remaining = curr.special_squares − curr.captured_squares

if remaining:
    # NEAR scan — runner reachable in 1 turn along the shoot ray
    for k in 1 .. SPECIAL_DEFENSE_NEAR_MAX:
        cell = (cx + k·dx, cy + k·dy)
        if cell out of bounds: break
        if cell ∈ remaining and runner_reaches_in_one(prev, cell):
            return +SPECIAL_DEFENSE_COEFF

    # FAR scan — runner reachable in exactly 2 turns along the shoot ray
    for k in SPECIAL_DEFENSE_FAR_MIN .. SPECIAL_DEFENSE_FAR_MAX:
        cell = (cx + k·dx, cy + k·dy)
        if cell out of bounds: break
        if cell ∈ remaining and runner_reaches_in_exactly_two(prev, cell):
            return +SPECIAL_DEFENSE_COEFF

return -BULLET_SPAM_PENALTY
```

**Newly-fired bullet detection.** No need to thread the action through the
shaper — `_newly_fired_bullet` advances every entry in `prev.projectiles`
one step in its own direction and treats any entry in `curr.projectiles`
not in that expected set as the freshly spawned bullet. When no shoot
happened this turn, the function returns `None` and the bonus short-circuits
to `0`. The detection is symmetric with the engine's logic: the bullet
spawns at `prev.catcher_pos` and is advanced one cell by `_tick_projectiles`
the same turn, so its `curr` position is always `(catcher + dir)` with its
direction equal to the chosen shoot direction.

**Why catcher coords, not bullet coords.** The two scan ranges are
expressed in Chebyshev distance from the catcher, not from the bullet's
post-tick position. This is cleaner for two reasons: (1) the catcher is the
agent making the decision, and (2) the relevant geometry of "the runner
reaches this cell in one or two turns" is most naturally indexed against
the firing point. The constants `NEAR_MAX = 3` and `FAR_MIN/MAX = 4/5` are
chosen to mesh with the runner's reach geometry: a 1-turn runner reach
extends ~3 cells from the runner's current position via sprint, and a
2-turn reach extends ~5 cells. Specials further out than `FAR_MAX = 5`
along the shoot ray are too distant for the runner to credibly capture
before the bullet arrives or the catcher can move.

**Why the spam penalty matches the bonus magnitude.** With both signals at
`0.10` the catcher gets a *symmetric* incentive: a good shot pays the same
as a bad shot costs. A catcher that fires randomly nets zero in
expectation, but the squared variance increases — random shots are
strictly worse than not shooting at all. The catcher learns to fire only
when it has identified a defensive opportunity. If you observe the
catcher under-shooting in training, raise the bonus or lower the penalty
asymmetrically; if you observe spam, do the reverse.

**The `runner_reaches_in_one` / `runner_reaches_in_exactly_two` predicates.**
Both are structural rather than radius-based, because sprint is
cardinal-only with charge and path constraints — a generic Chebyshev
radius would over-count diagonals and under-count cardinal-3 cells.
See `_runner_reaches_in_one` and `_runner_reaches_in_exactly_two` in
`reward_shaping.py` for the exact code; the gist is "enumerate every
legal first move the runner can make and check whether the target is
reachable from there given remaining sprint charges and catcher-blocking."
The reach predicates use `prev` because both the runner's position and
sprint count are unchanged across the catcher's half-turn anyway, and
`prev` is conceptually the state at firing time.

### Special-blocking attraction — `_special_blocking_attraction(curr)`

This is the per-step **area-defense** signal — the catcher's positional
objective. It tries to teach the catcher to interpose between the runner
and the runner's likely next targets.

```
remaining = list(curr.special_squares − curr.captured_squares)
if not remaining: return 0.0
remaining.sort(key=λ s: cheb(s, curr.runner_pos))
targets = remaining[:2]
weights = (SPECIAL_BLOCKING_NEAREST, SPECIAL_BLOCKING_SECOND)
return Σ w · blocking_score(curr.runner_pos, curr.catcher_pos, s)
       for (w, s) in zip(weights, targets)
```

**Target selection.** The two specials are picked by their Chebyshev
distance to the **runner**, with no safety filter. This is deliberate: if
we mirrored the runner's "safe from catcher" filter, we'd create circular
coupling ("I'm rewarded for blocking S only when S is far from me, but the
reward pulls me toward S"). Ranking by runner-distance alone gives the
right adaptation signal — as the runner moves, the top-2 list updates
every step, so the catcher's pull naturally shifts to wherever the runner
is heading next. When the catcher blocks the closest target effectively
and the runner pivots toward a different special, the new closest special
rises in the ranking and the catcher's signal redirects.

**Blocking score.** The function `_blocking_score(runner, catcher, s)`
expresses "how well does the catcher sit between R and S" via the
Chebyshev triangle inequality:

```
rc = cheb(runner, catcher)
cs = cheb(catcher, s)
rs = cheb(runner, s)
slack         = rc + cs − rs            # 0 iff catcher on a shortest R→S path
lead_deficit  = max(0, cs − rs)         # >0 iff runner reaches s strictly first
score         = 1 / (1 + slack + lead_deficit)
```

`slack` measures how far the catcher is "off the path" from the runner to
the special — the triangle inequality guarantees `slack ≥ 0`, with
equality iff the catcher sits on some Chebyshev geodesic from `runner` to
`s`. `lead_deficit` measures how badly the catcher loses the race to
`s` — `0` if the catcher arrives first or ties, positive (and equal to the
gap) if the runner arrives first.

The reciprocal shape `1 / (1 + slack + lead_deficit)` yields a smooth,
bounded score in `(0, 1]`. Peak at `1.0` when the catcher is on-path
**and** arrives at `s` no later than the runner. One cell off path → `0.5`.
Three steps behind on the race → `0.25`. The shape is deliberately gentle
because the runner makes decisions on partial information; we want the
catcher to *prefer* the perfect interpose spot but not be penalized to
zero for being slightly off.

**Additive over the two targets.** A position that partially blocks both
top-2 specials beats a position that perfectly blocks one and ignores the
second. If the top-2 lie in the same direction from the runner, the
catcher's optimal pose maximizes both at once. If they're on opposite
sides, the optimal pose is a compromise — exactly the "split the
difference" reasoning a strategic catcher should learn.

**Magnitudes.** At peak score (both targets perfectly blocked) the per-step
reward is `0.10 + 0.05 = 0.15`, well above any other per-step term in the
shaper. This is intentional and reflects the design decision that area
defense is the catcher's *core* strategy. Lower these values if you find
the catcher refusing to commit to kills even at close range.

### Chase bonus — `_chase_bonus(prev, curr)`

The chase bonus is one function returning a sum of two sub-signals:
**proximity** (with chase-direction gating) and **cornering** (with
sprint-pressure scaling). They're kept in the same function because both
condition on the runner's position and the catcher–runner geometry, and
because they're calibrated together against the blocking magnitude.

```
catcher_move_dist = cheb(curr.catcher_pos, prev.runner_pos)
delta             = catcher_move_dist − cheb(prev.catcher_pos, prev.runner_pos)
current_dist      = cheb(curr.runner_pos, curr.catcher_pos)

if current_dist ≤ DANGER_RADIUS and delta < 0:
    proximity = CHASE_COEFF / current_dist
else:
    proximity = PROXIMITY_COEFF / max(1, current_dist)

rx, ry          = curr.runner_pos
edge_dist       = min(rx, BOARD_SIZE − 1 − rx) + min(ry, BOARD_SIZE − 1 − ry)
corner_score    = 1.0 − edge_dist / (BOARD_SIZE − 1)
sprint_pressure = 1.0 + CORNER_SPRINT_BOOST · (1.0 − curr.sprint_charges / SPRINT_CHARGES)
cornering       = CORNERING_COEFF · corner_score · sprint_pressure

return proximity + cornering
```

**Proximity branch — chase-direction gating.** The strict dual of the
runner's `_catcher_distance_rewarding`. Two regimes:

- **Close-range chase**: `cheb ≤ DANGER_RADIUS = 2` *and* the catcher
  moved closer this half-turn. Reward: `+CHASE_COEFF / cheb`, peak
  `+0.20` at `cheb = 1`. This is the "commit to the kill" signal — the
  catcher gets a strong reward only when it's in striking range *and*
  actually committed by closing.
- **Ambient pull**: anywhere else, including close-range without
  closing motion. Reward: `+PROXIMITY_COEFF / max(1, cheb)`, in
  `[+0.0025, +0.015]` per step. This is a small constant gradient pulling
  the catcher toward the runner everywhere on the board, so the policy
  never fully drifts off into pure-blocking mode.

The chase-direction check uses the catcher's pre-move distance to the
runner's pre-move position vs. the post-move distance to the same target.
Negative delta means the catcher closed in. This is the catcher-side
mirror of the runner's "didn't retreat" check: the catcher has to *act
on* the proximity, not just be passively close, to earn the heavy reward.

**Why these magnitudes.** At `cheb = 1` (committed chase) the proximity
reward is `+0.20`, above the blocking-attraction peak of `+0.15`. So the
strategic transition the catcher learns is: by default, position to
block; once the runner is within `cheb = 1` and the catcher closes, the
chase reward wins. The crossover point sits around `cheb = 1.33` against
blocking max — chase wins at `cheb = 1` only, blocking wins everywhere
else. This preserves the "block-by-default, kill-on-opportunity" intent.

**Cornering branch — area-control with sprint awareness.** The
`corner_score` is `1.0` at any of the four corners of the 7×7 board
(`(0,0), (0,6), (6,0), (6,6)`), `0.0` at the center `(3,3)`, and varies
smoothly elsewhere. Formula:
`corner_score = 1 - (min(rx, 6-rx) + min(ry, 6-ry)) / 6`.

This signal is **steady, not derivative** — the catcher is rewarded for
keeping the runner in a cornered position, not just for pushing the runner
there once. A derivative-only signal would vanish the moment the runner
stops moving, but cornering's whole strategic value is that a cornered
runner has few escape options. The steady form pays while that state
persists, which is exactly the situation we want the catcher to maintain.

**Sprint-pressure multiplier.** `sprint_pressure` is `1.0` when the
runner has all 3 sprint charges and `1.5` when the runner has none. The
linear interpolation means the cornering bonus ramps up smoothly as the
runner's escape resource drains:

| `sprint_charges` | `sprint_pressure` | Max cornering / step |
|------------------|-------------------|----------------------|
| 3 (full)         | 1.0               | `0.015`              |
| 2                | 1.17              | `0.018`              |
| 1                | 1.33              | `0.020`              |
| 0                | 1.5               | `0.023`              |

This is the signal that turns "cornering the runner is mildly good" into
"cornering the runner is *especially* lethal when their sprints are
out." The catcher's policy needs to observe the runner's sprint charges
for this differential signal to actually reach the policy gradient —
channel 3 of the observation carries that information under the catcher
perspective specifically for this purpose.

**Why cornering is applied here and not multiplied into chase.** Keeping
cornering as an independent additive term makes it tunable in isolation
and means the cornering bonus *biases* the chase direction (toward
whichever escape lane closes the corner) without overriding the chase
decision itself. The chase signal is about "should the catcher commit
now"; the cornering signal is about "which direction should the
positioning lean." Composing them additively rather than multiplicatively
keeps those two questions decoupled in the policy gradient.

### Time-advantage bonus — `_time_advantage_bonus(curr)`

The strict dual of the runner's `_urgency_penalty`:

```
shortfall = SPECIAL_MAJORITY − len(curr.captured_squares)
if shortfall ≤ 0: return 0.0
turns_elapsed = curr.turn / TURN_LIMIT
return +TIME_ADVANTAGE_COEFF · shortfall · turns_elapsed
```

Where the runner's urgency penalty grows as the runner falls behind on
captures, the catcher's time-advantage bonus grows by the same magnitude
with opposite sign. The pedagogical purpose: teach the catcher that
**stalling counts as winning** when the runner is short of
`SPECIAL_MAJORITY`. A pure-chase catcher with no shaping might learn to
greedily pursue the runner all 40 turns; this signal rewards the catcher
for letting the clock work in its favor when the runner can't catch up.

When the runner has already reached `SPECIAL_MAJORITY` (shortfall ≤ 0), the
bonus zeros out — there's no advantage to maintain because the runner has
already crossed the win threshold by captures.

**Magnitudes.** Per-step peak is `0.005 · 4 · 1.0 = 0.02` on turn 40
when the runner has captured nothing. Per-episode cumulative ceiling
(linear ramp): `Σ_{t=1}^{40} 0.005 · 4 · (t/40) ≈ 0.41`. Comparable to
the runner's urgency penalty worst case, as intended.

## Why these magnitudes (calibration)

Per-step magnitudes by signal:

| Signal                           | Min       | Max       | Typical range            |
|----------------------------------|-----------|-----------|--------------------------|
| `_special_defense_bonus`         | `-0.10`   | `+0.10`   | `0` (no shoot this turn) |
| `_special_blocking_attraction`   | `0.0`     | `+0.15`   | `+0.05` to `+0.12`       |
| `_chase_bonus` (proximity only)  | `+0.0025` | `+0.20`   | `+0.005` to `+0.015`     |
| `_chase_bonus` (cornering only)  | `0.0`     | `+0.0225` | `+0.003` to `+0.012`     |
| `_time_advantage_bonus`          | `0.0`     | `+0.02`   | `+0.005` to `+0.015`     |

**Per-step ceiling (pathological).** Catcher chasing at `cheb = 1` and
closing (`+0.20` proximity), runner in a corner with zero sprints
(`+0.0225` cornering), both blocking targets at peak score (`+0.15`
blocking), runner on zero captures at turn 40 (`+0.02` time advantage),
and a qualifying shoot this turn (`+0.10` defense): total `≈ +0.49`.
Below the `+1` terminal kill reward — closing the kill remains more
rewarding than maintaining pressure, even at maximum theoretical pressure.

**Per-step floor.** Catcher far from runner (`+0.003` proximity at
`cheb = 6`), runner mid-board (`0` cornering), no blocking opportunity
(`0` blocking — happens when no specials remain), spam shot
(`-0.10` defense): total `≈ -0.10`. Above the `-1` terminal loss.

**Per-episode cumulative.** Worst case for a stalling catcher that wins
on timeout: blocking averages ~`+0.075`/step, cornering ~`+0.01`/step,
proximity ~`+0.005`/step, time-advantage averages ~`+0.005`/step (linear
ramp from 0 to 0.02), defense net ~`0` (catcher shoots sparingly and
well). Sum over 40 steps: `40 · 0.095 ≈ +3.8`. Plus the timeout terminal
reward of `+1`, total `≈ +4.8`. This is several times the terminal
magnitude, but the *gradient direction* still aligns with winning — every
component is calibrated so its sign matches the win/lose verdict, which
is what governs policy convergence in PPO.

If you want a tighter signal-to-terminal ratio, scale all per-step
coefficients uniformly (e.g. by 0.65); the strategic crossovers between
signals are preserved because they depend on coefficient *ratios*, not
absolute magnitudes.

**The terminal-dominance question for shaped catchers** is slightly more
subtle than for the runner. The catcher has two winning conditions: kill
(immediate `+1` mid-episode) and timeout (`+1` at turn 40 with runner
below majority). The shaping is designed so the *gradient* toward both
wins is consistent — chase signals reward kill trajectories, blocking
and time-advantage signals reward timeout trajectories — but a catcher
that learns to stall when it's "winning the timeout race" will rationally
forego risky kill attempts. That's by design: the catcher should prefer
a near-certain timeout to a 50/50 kill attempt, and the magnitudes encode
that preference.

## Integration with the environment

`CatchyRunEnv.__init__` dispatches by `trainee_role`:

```python
if trainee_role == "runner":
    self.reward_shaper = RunnerRewardShaper()
else:
    self.reward_shaper = CatcherRewardShaper()
```

`CatchyRunEnv._shape_reward` is a one-line delegation to whichever shaper
was bound. The base class's `shape()` handles the `curr.terminated`
short-circuit, so neither subclass needs to repeat that guard.

To tune a magnitude, edit the class attribute at the top of
`CatcherRewardShaper`; no environment changes required. Run
`rl_agent/trace_rewards.py` with `trainee_role="catcher"` to confirm
component signs and magnitudes against a trained policy or random actions.
The trace tool reports the four components (`special_defense`,
`special_blocking`, `time_advantage`, `chase`) alongside `base`.
