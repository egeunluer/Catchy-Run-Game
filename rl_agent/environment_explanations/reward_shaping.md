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

It only shapes for the **runner**. On any transition where `curr.terminated`
is `True`, `shape` short-circuits and returns the engine reward unchanged.
Every component below is runner-specific and only meaningful while the game is
ongoing.

## Tunables (class attributes)

All magnitudes live at the top of the class so there is exactly one place to
edit them:

| Attribute                   | Value | What it controls                                                                                                    |
|-----------------------------|-------|---------------------------------------------------------------------------------------------------------------------|
| `CAPTURE_BONUS`             | 0.24  | Reward per special-square captured this transition.                                                                 |
| `ALIVE_BONUS`               | 0.005 | Flat per-step bonus while the game is ongoing.                                                                      |
| `CATCHER_DISTANCE_COEFF`    | 0.30  | Heavy-branch numerator: in danger and runner didn't move away from the catcher.                                     |
| `CATCHER_PROXIMITY_COEFF`   | 0.02  | Light-branch numerator: in danger but moved away, or in the safe zone.                                              |
| `PROJECTILE_THREAT_COEFF`   | 0.25  | Flat per-bullet penalty when the runner stands on one of that bullet's next two cells (equal to `UNSAFE_CAPTURE_PENALTY`).            |
| `ATTRACTION_NEAREST`        | 0.01  | Numerator of the closest-safe-special attraction term.                                                              |
| `ATTRACTION_SECOND_NEAREST` | 0.005 | Numerator of the second-closest-safe-special attraction term.                                                       |
| `SPRINT_WASTE_PENALTY`      | 0.02  | Flat penalty when the runner sprints while already in the safe zone.                                                |
| `URGENCY_COEFF`             | 0.005 | Per-step penalty per missing special, scaled by fraction of the episode elapsed (zero once the runner reaches `SPECIAL_MAJORITY`). |
| `UNSAFE_CAPTURE_PENALTY`    | 0.25  | Flat penalty when the runner captures a special at Chebyshev ≤ 1 from where the catcher stood at the start of the half-turn.       |
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
       − Σ over prev bullets:  PROJECTILE_THREAT_COEFF  if curr.runner_pos on bullet's next or 2nd-next cell   # bullet threat
       + ATTRACTION_NEAREST       / max(1, cheb(runner, safe₁))       # attraction (closest)
       + ATTRACTION_SECOND_NEAREST/ max(1, cheb(runner, safe₂))       # attraction (2nd closest)
       − SPRINT_WASTE_PENALTY     if sprint_used and in_safe_zone     # sprint waste
       − URGENCY_COEFF · shortfall · (curr.turn / TURN_LIMIT)         # urgency (shortfall = max(0, SPECIAL_MAJORITY − |captured|))
       − UNSAFE_CAPTURE_PENALTY   if newly_captured > 0 and cheb(curr.runner_pos, prev.catcher_pos) ≤ 1   # unsafe capture (catcher)
       + projectile_threat(prev, curr)   if newly_captured > 0                                            # unsafe capture (bullet) — re-applies the bullet term

where catcher_term =
       −CATCHER_DISTANCE_COEFF  / cheb(runner, catcher)   if cheb(runner, catcher) ≤ DANGER_RADIUS
                                                           and runner did not move away from
                                                           the catcher's previous position
       −CATCHER_PROXIMITY_COEFF / cheb(runner, catcher)   otherwise
```

`safe₁`, `safe₂` are the two closest *uncaptured* specials that lie more than
`DANGER_RADIUS` away from the catcher. The projectile term is now a flat
threshold penalty: it fires `−PROJECTILE_THREAT_COEFF` per bullet **in
`prev.projectiles`** (the bullets in flight when the runner decided) whose next
or second-next cell coincides with `curr.runner_pos`, and contributes nothing
otherwise. The attraction term clamps its divisor via `max(1, …)`, so the
reward saturates at the coefficient value rather than blowing up at distance 0.
The catcher term needs no clamp — in any non-terminal state
`cheb(runner, catcher) ≥ 1` (a `0` means the catcher caught the runner, which
terminates the episode and short-circuits `shape`).

## Component by component

### Capture bonus — `_capture_bonus(prev, curr)`

```
CAPTURE_BONUS · ( |curr.captured_squares| − |prev.captured_squares| )
```

`prev` is from before the trainee moved, `curr` is from after the opponent
replied. Captures can only happen on the runner's turn, so this delta is `0`
or `1` per transition. The bonus is `+0.24` for one capture — small enough
that the terminal `±1` still dominates the win/lose verdict over an episode,
large enough to clearly beat the per-step soft costs (alive bonus,
attraction, light catcher branch) so the runner doesn't pass up a free
capture during normal play. Note that it does **not** outweigh the heavy
catcher branch (`−0.30` at cheb 1) — capturing a special that sits adjacent
to the catcher yields a net negative shaped reward by design, on the
assumption that the catcher would intercept on arrival anyway. The unsafe
capture penalty (see below) reinforces this for the exact `cheb ≤ 1` case
by neutralizing the capture bonus directly, so the heavy distance branch no
longer has to do the work alone on noisy transitions.

### Unsafe capture penalty — `_unsafe_capture_penalty(prev, curr)`

```
newly_captured = |curr.captured_squares| − |prev.captured_squares|
if newly_captured ≤ 0:
    penalty = 0
else:
    penalty  = −UNSAFE_CAPTURE_PENALTY   if cheb(curr.runner_pos, prev.catcher_pos) ≤ 1   # catcher part
    penalty += projectile_threat_penalty(prev, curr)                                       # bullet part
```

This penalty has **two parts**, both gated on a capture having happened this
turn (`newly_captured > 0`).

**Catcher part.** Fires when the runner captures a special that sits adjacent
to where the catcher was *at the start of the half-turn*. The distance is measured
against `prev.catcher_pos`, not `curr.catcher_pos`, because the danger that
matters is the danger at decision time: by the time `shape()` runs the
catcher has already replied, and `curr.catcher_pos` is confounded with the
catcher's policy (it may have stepped closer, shot in another direction, or
been killed mid-bullet — none of those facts should change how we score the
runner's decision).

The `cheb ≤ 1` threshold is the right cut-off. In any non-terminal state
`cheb(runner, catcher) ≥ 1` — a `0` means the catcher already caught the
runner and `shape()` short-circuits — so this fires exactly when the special
the runner just landed on is one step (cardinal or diagonal) from the
catcher's pre-reply position, i.e. inside the catcher's one-turn kill range.

Magnitude (`−0.25`) is chosen to **neutralize** the `+0.24` capture bonus
rather than dominate it. At a `cheb = 1` unsafe capture the immediate shaped
reward is roughly `+0.24` (capture) `− 0.30` (heavy catcher branch)
`+ 0.005` (alive) `− 0.25` (unsafe capture) `≈ −0.305`, with the terminal
`−1` arriving on the next half-turn whenever the catcher actually closes the
kill. The point is that the *intrinsic* incentive to grab the special is
gone — the heavy distance branch then makes the signal cleanly negative,
and the terminal reward does the rest. We deliberately keep this penalty
below the `+1` terminal so a *game-winning* unsafe capture (the runner's 4th
special) is still favored when the terminal reward will pay out.

**Bullet part.** This re-applies `_projectile_threat_penalty(prev, curr)` — the
exact same flat, per-bullet term documented above — but only on a capture turn.
The projectile term *already* fires once inside `_compute` for any step where
the runner stands in a bullet's next-two window; adding it a second time here
means a runner that grabs a special *by walking into* that window eats the
bullet penalty **twice** (`−0.50` for one threatening bullet, stacking per
bullet). The intent mirrors the catcher part: capturing is only worth the
risk when it's safe, so the same danger that costs the runner on an ordinary
step costs it double when it's the price of a grab. We deliberately reuse the
method rather than duplicate the cell math, so the "double" stays a true
doubling of whatever `PROJECTILE_THREAT_COEFF` is set to. Like the standalone
projectile term, the immediate-collision (`next_cell`) case is essentially
always terminal, so in practice this part fires on the `next2_cell` case.

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
`−0.30` — larger in magnitude than the `+0.24` capture bonus, so the runner is
incentivised to flee even mid-capture-run when the catcher closes and the
remaining specials sit in the wrong direction.

The light branch (`CATCHER_PROXIMITY_COEFF = 0.02`) covers the other two
cases. In the danger zone but actively retreating, it keeps a mild downward
pressure without punishing the good choice. In the safe zone it acts as a
constant "the catcher still exists" tax, scaled inversely by distance so it
fades as the runner gets clear (e.g. `≈ −0.0033` at cheb 6).

### Projectile threat penalty — `_projectile_threat_penalty(curr)`

For each in-flight projectile `((px, py), (dx, dy))` in **`prev.projectiles`**:

```
next_cell  = (px +   dx, py +   dy)
next2_cell = (px + 2·dx, py + 2·dy)
penalty   -= PROJECTILE_THREAT_COEFF   if curr.runner_pos ∈ {next_cell, next2_cell}
             0                          otherwise
```

This is a **flat threshold** penalty, deliberately the same shape as the
unsafe-capture penalty: rather than a distance-decaying cost levied on every
bullet in flight, it fires only when the runner moved onto one of the two cells
a bullet will occupy over the next two ticks — i.e. into the bullet's two-step
kill window. A bullet that is on the board but not on the runner's cell costs
nothing, so the runner is no longer taxed merely for the catcher having shot
somewhere on the board.

**Why `prev.projectiles`, not `curr.projectiles`.** This term is the runner's
analogue of the unsafe-capture penalty: it scores the runner's *own decision*,
so it must read the world as it stood at decision time. By the time `shape()`
runs the catcher has already replied, and `curr.projectiles` is doubly
confounded — the pre-existing bullets have advanced two ticks, and a bullet the
catcher *just fired* may now sit on the board. Penalizing the runner for that
fresh shot would punish it for the catcher's response rather than its own move,
since the bullet did not exist when the runner chose where to go. Reading
`prev.projectiles` and checking the runner's chosen destination
(`curr.runner_pos`, which the catcher's reply cannot change) restores the clean
"did the runner step into a bullet that was already in flight?" semantics.

Note the next-cell case is largely terminal: if the runner steps onto a bullet's
immediate next cell, the engine's per-tick `bullet_hit_runner` check catches it
and `shape()` short-circuits on the terminal branch. So in practice this term
mainly fires on `next2_cell` — the runner survived this tick but parked where a
pre-existing bullet is headed. Keeping both cells preserves the "two dangerous
squares" framing and is harmless.

Penalties from multiple in-flight projectiles still sum: a runner whose cell
sits in the next-two window of two different bullets really is in twice as
much danger, so it eats `−0.25` per threatening bullet. The coefficient
(`PROJECTILE_THREAT_COEFF = 0.25`) equals `UNSAFE_CAPTURE_PENALTY` by
design — both are flat "you stepped into a kill square" penalties and share a
magnitude.

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
distance 1 the combined attraction contributes at most `0.01 + 0.005 = 0.015`
— clearly less than the `+0.24` for actually landing on the square, so
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
  branch, `−0.30`), the runner standing on one bullet's next-two cell
  (`−0.25`), shortfall = 4 at turn 38 (urgency `≈ −0.019`), no captures,
  no safe specials, no waste, alive bonus (`+0.005`): shaping ≈ `−0.56`.
  Still above the `−1` terminal as a single-step signal. The unsafe-capture
  penalty can only fire when `newly_captured > 0`, so it never stacks with the
  "no captures" floor — its worst-case stack is with the capture bonus. On a
  cheb-1 unsafe grab with no bullet it lands a net `≈ −0.305` (capture `+0.24`,
  heavy catcher `−0.30`, alive `+0.005`, unsafe-catcher `−0.25`). If that same
  grab also steps into a bullet's next-two window, the bullet term fires twice
  (once in `_compute`, once in the unsafe-capture bullet part) for an extra
  `−0.50` per threatening bullet, taking the one-bullet worst case to `≈ −0.805`.
  With *two* bullets converging on the captured cell the doubled-and-stacked
  bullet penalty alone reaches `−1.0`, so the single-step shaping can dip below
  the `−1` terminal. We accept this: a capture that walks into a two-bullet
  crossfire next to the catcher is a near-certain death the policy *should* be
  strongly steered away from, and it is a transient per-step signal — episode
  return is still anchored by the terminal `±1`, so aggregate terminal dominance
  (never rewarding a losing line over a winning one across the episode) holds.
- **Per-step ceiling.** Captured a special (`+0.24`) on a transition where
  the runner is on top of one safe special (`+0.01`) with another at
  distance 1 (`+0.005`), catcher far in the safe zone (light branch
  ≈ `−0.0033` at cheb 6), no bullets, no waste, alive (`+0.005`): shaping
  ≈ `+0.26`. Below the `+1` terminal. (The capture takes the runner to
  ≥ 1 captures; once shortfall hits 0 the urgency term silently zeros out.)
- **Per-episode cumulative shaping (excluding capture events).** Roughly
  bounded by `20 × (0.005 + 0.01 + 0.005) = +0.4` from alive + attraction
  alone. Captures add up to `7 × 0.24 ≈ +1.68` more in a perfect run, but a
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
