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

It only shapes for the **runner** trainee. For a catcher trainee — or any
transition where `curr.terminated` is `True` — `shape` short-circuits and
returns the engine reward unchanged. Every component below is runner-specific
and only meaningful while the game is ongoing.

## Tunables (class attributes)

All magnitudes live at the top of the class so there is exactly one place to
edit them:

| Attribute                   | Value | What it controls                                                                                                    |
|-----------------------------|-------|---------------------------------------------------------------------------------------------------------------------|
| `CAPTURE_BONUS`             | 0.2   | Reward per special-square captured this transition.                                                                 |
| `ALIVE_BONUS`               | 0.005 | Flat per-step bonus while the game is ongoing.                                                                      |
| `CATCHER_DISTANCE_COEFF`    | 0.30  | Heavy-branch numerator: in danger and runner didn't move away from the catcher.                                     |
| `CATCHER_PROXIMITY_COEFF`   | 0.03  | Light-branch numerator: in danger but moved away, or in the safe zone.                                              |
| `PROJECTILE_THREAT_COEFF`   | 0.25  | Numerator of the per-projectile threat penalty.                                                                     |
| `ATTRACTION_NEAREST`        | 0.03  | Numerator of the closest-safe-special attraction term.                                                              |
| `ATTRACTION_SECOND_NEAREST` | 0.01  | Numerator of the second-closest-safe-special attraction term.                                                       |
| `SPRINT_WASTE_PENALTY`      | 0.02  | Flat penalty when the runner sprints while already in the safe zone.                                                |
| `URGENCY_COEFF`             | 0.005 | Per-step penalty per missing special, scaled by fraction of the episode elapsed (zero once the runner reaches `SPECIAL_MAJORITY`). |
| `DANGER_RADIUS`             | 2     | Specials within this Chebyshev distance of the catcher are unsafe; also gates the heavy/light split of the catcher term. |
| `SAFE_ZONE_THRESHOLD`       | 3     | Runner is in the safe zone iff `cheb(runner, catcher) > 3` (used only by sprint waste).                              |

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
       + catcher_term                                                 # see below
       − Σ over bullets:  PROJECTILE_THREAT_COEFF / max(1, min(d₁, d₂))   # bullet gradient
       + ATTRACTION_NEAREST       / max(1, cheb(runner, safe₁))       # attraction (closest)
       + ATTRACTION_SECOND_NEAREST/ max(1, cheb(runner, safe₂))       # attraction (2nd closest)
       − SPRINT_WASTE_PENALTY     if sprint_used and in_safe_zone     # sprint waste
       − URGENCY_COEFF · shortfall · (curr.turn / TURN_LIMIT)         # urgency (shortfall = max(0, SPECIAL_MAJORITY − |captured|))

where catcher_term =
       −CATCHER_DISTANCE_COEFF  / cheb(runner, catcher)   if cheb(runner, catcher) ≤ DANGER_RADIUS
                                                           and runner did not move away from
                                                           the catcher's previous position
       −CATCHER_PROXIMITY_COEFF / cheb(runner, catcher)   otherwise
```

`safe₁`, `safe₂` are the two closest *uncaptured* specials that lie more than
`DANGER_RADIUS` away from the catcher. `d₁`, `d₂` are the Chebyshev distances
from the runner to a projectile's next cell and the cell after that. The
projectile and attraction terms clamp their divisor via `max(1, …)`, so the
penalty / reward saturates at the coefficient value rather than blowing up at
distance 0. The catcher term needs no clamp — in any non-terminal state
`cheb(runner, catcher) ≥ 1` (a `0` means the catcher caught the runner, which
terminates the episode and short-circuits `shape`).

## Component by component

### Capture bonus — `_capture_bonus(prev, curr)`

```
CAPTURE_BONUS · ( |curr.captured_squares| − |prev.captured_squares| )
```

`prev` is from before the trainee moved, `curr` is from after the opponent
replied. Captures can only happen on the runner's turn, so this delta is `0`
or `1` per transition. The bonus is `+0.2` for one capture — small enough
that the terminal `±1` still dominates the win/lose verdict over an episode,
large enough to clearly beat the per-step soft costs (alive bonus,
attraction, light catcher branch) so the runner doesn't pass up a free
capture during normal play.

### Alive bonus — flat `+ALIVE_BONUS`

Added unconditionally inside `shape`. The runner sees roughly 20 env-steps
per episode, so the cumulative alive bonus tops out at ≈ `+0.1` — well below
the `±1` terminal reward. Its job is to give a small constant positive
gradient on every step the game continues, which speeds up early learning
when most other signals are noisy.

### Catcher term — `_catcher_distance_rewarding(curr, prev)`

This term has two strengths depending on (a) whether the runner is in the
danger zone and (b) whether it actively moved away from the catcher this
turn:

```
runner_move_dist = cheb(curr.runner_pos, prev.catcher_pos)
delta_dist       = runner_move_dist − cheb(prev.runner_pos, prev.catcher_pos)
current_dist     = cheb(curr.runner_pos, curr.catcher_pos)

if current_dist ≤ DANGER_RADIUS:
    if delta_dist ≤ 0:          # runner did not increase its distance from where catcher was
        penalty = −CATCHER_DISTANCE_COEFF  / current_dist
    else:                        # runner moved away from catcher's previous position
        penalty = −CATCHER_PROXIMITY_COEFF / current_dist
else:                            # safe zone
    penalty = −CATCHER_PROXIMITY_COEFF     / current_dist
```

`delta_dist` is measured against the catcher's *previous* position, not its
current one. We're scoring the runner on the direction *it* chose, before the
catcher's reply could distort the picture — `delta_dist > 0` means the runner
stepped to a cell farther from where the catcher was when it decided.

The heavy branch (`CATCHER_DISTANCE_COEFF = 0.30`) fires only when the runner
is already in striking range *and* failed to retreat. At Chebyshev 1 that's
`−0.30` — larger in magnitude than the `+0.2` capture bonus, so the runner is
incentivised to flee even mid-capture-run when the catcher closes and the
remaining specials sit in the wrong direction.

The light branch (`CATCHER_PROXIMITY_COEFF = 0.03`) covers the other two
cases. In the danger zone but actively retreating, it keeps a mild downward
pressure without punishing the good choice. In the safe zone it acts as a
constant "the catcher still exists" tax, scaled inversely by distance so it
fades as the runner gets clear (e.g. `−0.005` at cheb 6).

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
two bullets really is in twice as much danger. At distance 1 to either
sampled cell the penalty saturates at `−0.25` per bullet.

### Special-square attraction — `_special_attraction(curr)`

```
remaining = curr.special_squares − curr.captured_squares
safe      = [s in remaining if cheb(s, catcher) > DANGER_RADIUS]
sort safe by cheb(s, runner) ascending
reward    = ATTRACTION_NEAREST        / max(1, cheb(safe[0], runner))
         + ATTRACTION_SECOND_NEAREST / max(1, cheb(safe[1], runner))   (if a 2nd safe special exists)
```

If no safe specials remain, the term is `0`. The `DANGER_RADIUS` filter is
there so the runner is never *positively* steered toward a special inside the
catcher's strike zone — the catcher would just intercept on arrival. At
distance 1 the combined attraction contributes at most `0.03 + 0.01 = 0.04`
— clearly less than the `+0.2` for actually landing on the square, so
"capture it" always beats "linger near it." Only the two nearest safe
specials count; farther specials add nothing, so the runner isn't rewarded
for being equidistant from many remote specials.

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
zone" is `cheb > 3`: any closer and the sprint could be a real flee, which
we don't want to discourage.

Sprint usage is detected by comparing `prev.sprint_charges` to
`curr.sprint_charges`. The opponent (catcher) cannot modify the runner's
sprint counter, so any change between `prev` and `curr` is attributable to
the runner's own action this turn.

### Urgency penalty — `_urgency_penalty(curr)`

```
shortfall     = SPECIAL_MAJORITY − |curr.captured_squares|
if shortfall ≤ 0:
    penalty = 0
else:
    penalty = −URGENCY_COEFF · shortfall · (curr.turn / TURN_LIMIT)
```

The win condition on a timeout is "captured at least `SPECIAL_MAJORITY = 4`
specials." Channel 4 of the observation exposes `turn / TURN_LIMIT`, so the
policy *can* learn urgency on its own, but in practice the gradient from the
sparse terminal reward isn't enough to teach "you're at turn 35 with 2
captures — start hunting." This term injects that signal directly.

The penalty is zero at `turn = 0` and grows linearly with elapsed time,
weighted by how many captures the runner still needs. Once the runner reaches
4 captures it disengages entirely (`shortfall ≤ 0 ⇒ 0`), so a runner that
gets ahead of the curve is never punished for "running down the clock" to
secure the win.

Worst-case magnitudes (with `URGENCY_COEFF = 0.005`):

- **Single step**, shortfall = 4, turn = 38: `−0.005 · 4 · 0.95 ≈ −0.019` —
  negligible against the heavy catcher branch (`−0.30`) or a bullet
  (`−0.25`), so it never dominates immediate threat signals.
- **Cumulative**, shortfall = 4 across all 20 of the runner's steps (a
  runner that captures nothing): `≈ −0.19`. Comfortably above the `−1`
  terminal, so terminal dominance is preserved.

Using `curr` (post-opponent-reply) for `turn` and `captured_squares` is
consistent with how the other shaping terms read state: the agent is being
scored on the situation it ends up in after the half-turn fully plays out.
The `terminated` short-circuit at the top of `shape()` means the urgency
term never fires on the terminal step itself — by then the verdict is
already in.

Note `SAFE_ZONE_THRESHOLD` (3) is one cell looser than `DANGER_RADIUS` (2):
the "danger zone" that gates the catcher term and the attraction filter is
`cheb ≤ 2`, while the "safe zone" that gates the sprint penalty is
`cheb > 3`. The intermediate distance `cheb = 3` is "neither in danger nor
unambiguously safe" — sprinting from there isn't flagged as wasted.

## Why the magnitudes are calibrated this way

A few worst-case sums to keep the relative scales straight:

- **Per-step floor.** Catcher adjacent and runner didn't retreat (heavy
  branch, `−0.30`), one bullet's sampled cell adjacent to the runner
  (`−0.25`), shortfall = 4 at turn 38 (urgency `≈ −0.019`), no captures,
  no safe specials, no waste, alive bonus (`+0.005`): shaping ≈ `−0.56`.
  Still above the `−1` terminal as a single-step signal.
- **Per-step ceiling.** Captured a special (`+0.2`) on a transition where
  the runner is on top of one safe special (`+0.03`) with another at
  distance 1 (`+0.01`), catcher far in the safe zone (light branch
  ≈ `−0.005` at cheb 6), no bullets, no waste, alive (`+0.005`): shaping
  ≈ `+0.24`. Below the `+1` terminal. (The capture takes the runner to
  ≥ 1 captures; once shortfall hits 0 the urgency term silently zeros out.)
- **Per-episode cumulative shaping (excluding capture events).** Roughly
  bounded by `20 × (0.005 + 0.03 + 0.01) ≈ +0.9` from alive + attraction
  alone. Captures add up to `7 × 0.2 = 1.4` more in a perfect run, but a
  7-capture episode also ends in the runner's favor — the shaping is
  rewarding the path the policy should take, not gaming an undeserved win.
- **Per-episode cumulative urgency.** Worst case (shortfall = 4 for the
  entire episode) ≈ `−0.19`; zero in any episode where the runner reaches
  4 captures before turn 0 of the urgency curve has much elapsed. The
  bound is small enough that terminal dominance still holds even in the
  pathological "captures nothing, loses on timeout" episode.

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
