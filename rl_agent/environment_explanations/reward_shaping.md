# Reward Shaping

The engine itself only emits a terminal reward: `+1` to the winner, `-1` to the
loser, `0` on every non-terminal step. That signal is correct but sparse — the
runner can play for 20 half-turns without ever learning what a "good" move
looks like, because the gradient only arrives at the end. `RewardShaper` sits
between the engine and the agent and adds *shaping* signals: small per-step
nudges that point the policy toward useful behaviour (run away from the
catcher, capture specials, dodge bullets) without overriding the terminal
verdict.

`RewardShaper` lives in `rl_agent/reward_shaping.py`. The environment owns one
instance (`self.reward_shaper`, built in `__init__`) and calls
`reward_shaper.shape(prev, curr, base)` once per `step()`. The class is plain
state: it stores the trainee's role and the tunables, exposes one public
method (`shape`), and keeps each reward component in its own private method
so the components are easy to tune in isolation.

It only shapes for the **runner** trainee. For a catcher trainee, `shape`
short-circuits and returns the engine reward unchanged — every component
below is runner-specific.

## Tunables (class attributes)

All magnitudes live at the top of the class so there is exactly one place to
edit them:

| Attribute                   | Value | What it controls                                                       |
|-----------------------------|-------|------------------------------------------------------------------------|
| `CAPTURE_BONUS`             | 0.1   | Reward per special-square captured this transition.                    |
| `ALIVE_BONUS`               | 0.005 | Flat per-step bonus while the game is ongoing.                         |
| `CATCHER_DISTANCE_COEFF`    | 0.2   | Numerator of the catcher-distance penalty.                             |
| `PROJECTILE_THREAT_COEFF`   | 0.15  | Numerator of the per-projectile threat penalty.                        |
| `ATTRACTION_NEAREST`        | 0.02  | Numerator of the closest-special attraction term.                      |
| `ATTRACTION_SECOND_NEAREST` | 0.01  | Numerator of the second-closest-special attraction term.               |
| `SPRINT_WASTE_PENALTY`      | 0.02  | Flat penalty when the runner sprints while already in the safe zone.   |
| `DANGER_RADIUS`             | 2     | Specials within this Chebyshev distance of the catcher are unsafe.     |
| `SAFE_ZONE_THRESHOLD`       | 2     | Runner is in the safe zone iff `cheb(runner, catcher) > 2`.            |

Distances are **Chebyshev** throughout. Movement on this board is
8-directional, so one step in any direction is exactly one unit of Chebyshev
distance — the metric matches the agent's actual reach per turn.

## The reward function in one expression

For a runner trainee, given `prev` (state before the trainee's action), `curr`
(state after the trainee's action *and* the opponent's response — i.e.
`self.state` inside `env.step()`), and the engine's `base` reward:

```
shaped = base
       + CAPTURE_BONUS  · newly_captured                              # capture bonus
       + ALIVE_BONUS                                                  # alive bonus
       - CATCHER_DISTANCE_COEFF / max(1, cheb(runner, catcher))       # catcher gradient
       - Σ over bullets:  PROJECTILE_THREAT_COEFF / max(1, min(d₁, d₂))   # bullet gradient
       + ATTRACTION_NEAREST       / max(1, cheb(runner, safe₁))       # attraction (closest)
       + ATTRACTION_SECOND_NEAREST/ max(1, cheb(runner, safe₂))       # attraction (2nd closest)
       - SPRINT_WASTE_PENALTY      if sprint_used and in_safe_zone     # sprint waste
```

`safe₁`, `safe₂` are the two closest *uncaptured* specials that lie more than
`DANGER_RADIUS` away from the catcher. `d₁`, `d₂` are the Chebyshev distances
from the runner to a projectile's next cell and the cell after that. Every
divisor is clamped via `max(1, …)`, so the penalty / reward saturates at the
coefficient value rather than blowing up to ∞ at distance 0.

## Component by component

### Capture bonus — `_capture_bonus(prev, curr)`

```
CAPTURE_BONUS · ( |curr.captured_squares| − |prev.captured_squares| )
```

`prev` is from before the trainee moved, `curr` is from after the opponent
replied. Captures can only happen on the runner's turn, so this delta is `0`
or `1` per transition. The bonus is `+0.1` for one capture — small enough
that the terminal `±1` still dominates over a full episode, large enough to
beat any single-step penalty so the runner never refuses a free capture.

### Alive bonus — flat `+ALIVE_BONUS`

Added unconditionally inside `shape`. The runner sees roughly 20 env-steps
per episode, so the cumulative alive bonus tops out at ≈ `+0.1` — well below
the `±1` terminal reward. Its job is to give a small constant positive
gradient on every step the game continues, which speeds up early learning
when most other signals are noisy.

### Catcher-distance penalty — `_catcher_distance_penalty(curr)`

```
-CATCHER_DISTANCE_COEFF / max(1, cheb(curr.runner_pos, curr.catcher_pos))
```

At Chebyshev 1 (catcher adjacent) the penalty is `-0.2` — a stronger negative
than the `+0.1` capture bonus, so the runner is incentivised to break off and
flee even mid-capture-run when the catcher closes. The `max(1, …)` clamp
matters because the actual Chebyshev distance is never 0 in a non-terminal
state (a 0 would mean the catcher caught the runner, which terminates the
episode), but at distance 1 we want the penalty at its full strength.

### Projectile threat penalty — `_projectile_threat_penalty(curr)`

For each in-flight projectile `((px, py), (dx, dy))` in `curr.projectiles`:

```
next_cell  = (px +   dx, py +   dy)
next2_cell = (px + 2·dx, py + 2·dy)
d₁ = cheb(runner_pos, next_cell)
d₂ = cheb(runner_pos, next2_cell)
penalty -= PROJECTILE_THREAT_COEFF / max(1, min(d₁, d₂))
```

The penalty samples the two cells the bullet will land on over the next two
ticks. Using `min(d₁, d₂)` means proximity to *either* upcoming cell triggers
a sharp penalty — the gradient stays steep so the runner reliably learns to
side-step the bullet's path, not just step out of the single next cell.
Penalties from multiple in-flight projectiles sum: a runner sandwiched by
two bullets really is in twice as much danger.

### Special-square attraction — `_special_attraction(curr)`

```
remaining = curr.special_squares − curr.captured_squares
safe      = [s in remaining if cheb(s, catcher) > DANGER_RADIUS]
sort safe by cheb(s, runner) ascending
reward    = ATTRACTION_NEAREST        / max(1, cheb(safe[0], runner))
         + ATTRACTION_SECOND_NEAREST / max(1, cheb(safe[1], runner))   (if a 2nd safe special exists)
```

If no safe specials remain, the term is `0`. The `DANGER_RADIUS` filter is
the same idea used in the previous shaping: never *positively* steer the
runner toward a special that sits in the catcher's strike zone, because the
catcher would just intercept on arrival. At distance 1 the term contributes
at most `0.02 + 0.01 = 0.03` — clearly less than the `+0.1` for actually
landing on the square, so "capture it" always beats "linger near it." Only
the two nearest safe specials count; farther specials add nothing, so the
runner isn't rewarded for being equidistant from many remote specials.

### Sprint-waste penalty — `_sprint_waste_penalty(prev, curr)`

```
sprint_used  = prev.sprint_charges > curr.sprint_charges
in_safe_zone = cheb(runner, catcher) > SAFE_ZONE_THRESHOLD
penalty      = -SPRINT_WASTE_PENALTY  if (sprint_used and in_safe_zone) else 0
```

A sprint consumes one charge; capturing a special on the same turn refills a
charge (net change to `sprint_charges` is 0 in that case). The penalty fires
only when `prev > curr`, so a sprint that's immediately recouped via a
capture is exempt — that's a productive sprint, not a wasted one. "Safe
zone" is `cheb > 2`: any closer and the sprint could be a real flee, which
we don't want to discourage.

Sprint usage is detected by comparing `prev.sprint_charges` to
`curr.sprint_charges`. The opponent (catcher) cannot modify the runner's
sprint counter, so any change between `prev` and `curr` is attributable to
the runner's own action this turn.

## Why the magnitudes are calibrated this way

A few worst-case sums to keep the relative scales straight:

- **Per-step floor.** With catcher adjacent (`-0.2`), one bullet next to the
  runner (`-0.15`), no captures, no safe specials, no waste: shaping ≈
  `-0.345` plus the `+0.005` alive bonus ≈ `-0.34`. Still far above the
  `-1` terminal.
- **Per-step ceiling.** Captured a special (`+0.1`) on a transition where
  the runner is on top of one safe special (`+0.02`) with another at
  distance 1 (`+0.01`), catcher far (penalty ≈ `-0.033`), no bullets, no
  waste: shaping ≈ `+0.097`. Below the `+1` terminal.
- **Per-episode cumulative shaping.** Roughly bounded by
  `20 × (0.005 + 0.02 + 0.01) ≈ +0.7` on the positive side. The terminal
  `±1` is what the policy is ultimately chasing.

Terminal dominance is the property we care about: at no point does the
shaping reward a sequence of moves that loses the game more than it rewards
winning it. Everything else is a gradient for the learner.

## Integration with the environment

`CatchyRunEnv.__init__` builds `self.reward_shaper = RewardShaper(trainee_role)`.
`CatchyRunEnv._shape_reward` is a one-line delegation:

```python
def _shape_reward(self, prev, curr, base):
    return self.reward_shaper.shape(prev, curr, base)
```

`env.step()` keeps the original signature — the `RewardShaper` reads sprint
usage from the state delta, so no action argument needed to be threaded
through. To tune a magnitude, edit the class attribute at the top of
`reward_shaping.py`; no environment changes required.
