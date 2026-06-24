# Catcher Reward Shaping

The engine emits the same sparse `±1` terminal signal for the catcher as it
does for the runner: `+1` when the catcher wins (kill by step, kill by
projectile, or timeout with the runner below `SPECIAL_MAJORITY`), `-1`
otherwise, `0` on every non-terminal step. Unlike the runner, the catcher's
terminal events can happen mid-episode (a kill terminates immediately on
contact), so the credit-assignment problem is shorter.

This shaper is deliberately **minimal**. An earlier version layered four
dense components (special-blocking attraction, chase/cornering, time
advantage, defensive bullets) on top of the sparse signal. The catcher
learned to farm those per-step terms instead of doing its job — classic
reward hacking. The current design strips the shaping back to the bare
minimum task signals and lets the policy discover the rest:

1. **catch the runner** — by far the biggest reward,
2. **defend the special squares** — hard penalty whenever the runner grabs one,
3. **shoot well** — keep the well-aimed-bullet reward, penalize every other
   shot hard.

Nothing else. No positioning, chasing, or cornering signals. The hypothesis
is that with a strong, clean reward for the actual objective and strong, clean
penalties for the actual failures, the catcher will learn the intermediate
behaviors (interposing, chasing, cornering) on its own rather than being
hand-held into them — and without a dense surface to hack.

`CatcherRewardShaper` lives alongside `RunnerRewardShaper` in
`rl_agent/reward_shaping.py`. Both subclass a shared `RewardShaper` base
that owns the `_cheb` helper and the public `shape(prev, curr, base)`
method. The base `shape()` short-circuits on `curr.terminated` and returns
only `base`; **the catcher overrides `shape()`** so it can add the catch
bonus in the terminal branch (see below). The environment picks the right
shaper at construction time based on `trainee_role`.

## Tunables (class attributes)

| Attribute                  | Value | What it controls                                                                          |
|----------------------------|-------|-------------------------------------------------------------------------------------------|
| `SPECIAL_DEFENSE_COEFF`    | 0.10  | One-shot reward when a freshly fired bullet's ray covers an imminently-capturable special. |
| `SPECIAL_DEFENSE_NEAR_MAX` | 3     | Inclusive max Chebyshev distance along the shoot ray for the "runner reaches in 1" check.   |
| `SPECIAL_DEFENSE_FAR_MIN`  | 4     | Inclusive min Chebyshev distance along the shoot ray for the "runner reaches in 2" check.   |
| `SPECIAL_DEFENSE_FAR_MAX`  | 5     | Inclusive max Chebyshev distance along the shoot ray for the "runner reaches in 2" check.   |
| `BULLET_SPAM_PENALTY`      | 0.30  | Hard flat penalty on the shoot turn when the new bullet's ray fails both reach checks.       |
| `CATCH_BONUS`              | 2.0   | Standout reward (on top of the terminal `+1`) when the catcher actually catches the runner.  |
| `CAPTURED_SQUARE_PENALTY`  | 0.50  | Hard penalty per special the runner captures during the catcher's env-step.                  |

Distances are **Chebyshev** throughout, matching the engine's 8-directional
movement (one step in any direction equals one unit of Chebyshev distance).

## The reward function in two parts

The catcher overrides `shape()` rather than only implementing `_compute()`:

```
shape(prev, curr, base):
    if curr.terminated:
        return base + (CATCH_BONUS if _is_catch(curr) else 0.0)
    return _compute(prev, curr, base)

_compute(prev, curr, base):
    return base
         + _special_defense_bonus(prev, curr)   # one-shot per fired bullet
         + _captured_square_penalty(prev, curr)  # per special the runner grabs
```

The terminal branch is where the catch bonus has to live, because a catch
*is* a terminal event — the game ends the instant the catcher steps onto the
runner or a bullet lands on it. The base `shape()` would discard everything
but `base` on termination, so the catcher's override re-adds the bonus there.

## Component by component

### Catch bonus — `_is_catch(curr)` in the terminal branch

```
_is_catch(curr):
    if curr.winner != "catcher":      return False   # excludes runner wins
    if curr.runner_pos == curr.catcher_pos: return True   # step kill
    return any(p == curr.runner_pos for p, _ in curr.projectiles)  # bullet kill
```

A catcher win comes in two flavors: an **actual catch** (the catcher steps
onto the runner, or a bullet it fired lands on the runner) and a **timeout
win** (40 half-turns elapse with the runner below `SPECIAL_MAJORITY`). Only
the actual catch earns `CATCH_BONUS`. The discriminator is positional and
reconstructable from `curr` alone: a step-kill leaves `runner_pos ==
catcher_pos`; a bullet-kill leaves a projectile sitting on `runner_pos` (the
engine does not remove the bullet that lands). A timeout win has the catcher
elsewhere and no bullet on the runner, so `_is_catch` returns `False`.

**Why timeout wins are excluded.** The whole point of the rework is to make
the catcher *go for the runner*. Rewarding timeout wins as richly as kills
would re-introduce the "stalling counts as winning" behavior that the old
`_time_advantage_bonus` taught — exactly the passivity we're trying to
remove. A timeout win still collects the engine's terminal `+1`; it just
doesn't get the `+2` on top.

**Magnitude.** A catch totals `base (+1) + CATCH_BONUS (+2) = +3`. That is
larger than any plausible per-episode accumulation of the two per-step
signals, so the policy gradient points unambiguously at "end the episode by
catching the runner." This is intentional: the catch is the objective, and
its reward should dominate everything else by design.

### Special-defense bonus / bullet-spam penalty — `_special_defense_bonus(prev, curr)`

Unchanged from the previous design, and the *only* signal that fires on
bullet firing — **exactly once per bullet**, on the turn it is spawned. A
defensive shot is a decision made at firing time; the credit is assigned to
that decision rather than smeared across every turn the bullet stays in
flight.

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

**Newly-fired bullet detection.** `_newly_fired_bullet` advances every entry
in `prev.projectiles` one step in its own direction and treats any entry in
`curr.projectiles` not in that expected set as the freshly spawned bullet.
When no shoot happened this turn the function returns `None` and the bonus
short-circuits to `0`. The bullet spawns at `prev.catcher_pos` and is advanced
one cell the same turn, so its `curr` position is `(catcher + dir)` with
direction equal to the chosen shoot direction.

**Why the spam penalty is now harder than the bonus.** With the bonus at
`0.10` and the penalty at `0.30`, a catcher that fires at random nets `-0.20`
in expectation per shot — firing is *strictly* discouraged unless the shot
clears one of the two reach checks. The asymmetry (vs. the old symmetric
`0.10/0.10`) is deliberate: the rework wants a catcher that shoots rarely and
only with a concrete defensive purpose, so undisciplined shooting is punished
harder than disciplined shooting is rewarded. If you observe the catcher
refusing to shoot even in clear defensive situations, lower the penalty.

**The reach predicates** (`_runner_reaches_in_one`,
`_runner_reaches_in_exactly_two`) are structural rather than radius-based,
because sprint is cardinal-only with charge and path constraints. See the code
in `reward_shaping.py`; the gist is "enumerate every legal first move the
runner can make and check whether the target is reachable from there given
remaining sprint charges and catcher-blocking." They read `prev` because the
runner's position and sprint count are unchanged across the catcher's
half-turn, and `prev` is the state at firing time.

### Captured-square penalty — `_captured_square_penalty(prev, curr)`

The mirror of the runner's `_capture_bonus`, negated and made hard:

```
newly_captured = len(curr.captured_squares) − len(prev.captured_squares)
return -CAPTURED_SQUARE_PENALTY · newly_captured
```

For a catcher trainee, `prev` is the state before the catcher's action and
`curr` is the state after the catcher's action **and** the runner's response.
The runner moves once inside that window, so any special the runner steps onto
shows up as `newly_captured` here. At `0.50` per square and a `SPECIAL_MAJORITY`
of 4, the cumulative pressure of letting the runner win on captures is about
`-2.0` — comparable in scale to the catch bonus, so "stop the captures" reads
as a real objective, not a rounding error.

**Edge case: the 7th capture.** Capturing the 7th special ends the game in the
runner's favor (`terminated=True`), which routes through the terminal branch
of `shape()` — so `_captured_square_penalty` does **not** fire on that final
grab. That's fine: the engine's terminal `-1` already punishes the loss, and
double-counting the last square would not change the gradient direction.

## Calibration summary

| Signal                     | Min     | Max     | When it fires                          |
|----------------------------|---------|---------|----------------------------------------|
| catch bonus (terminal)     | `0`     | `+2.0`  | catcher actually catches the runner    |
| `_special_defense_bonus`   | `-0.30` | `+0.10` | a bullet is fired this turn            |
| `_captured_square_penalty` | `-0.50` | `0`     | runner captures ≥1 special this step   |

Per-step magnitude is dominated by the captured-square penalty (`-0.50`); the
catch bonus (`+2.0`, terminal-only) dominates the episode. Every sign matches
the win/lose verdict, which is what governs PPO convergence. There is no dense
positive per-step surface left to farm — the only repeatable positive signal
is the well-aimed bullet, which is gated behind two structural reach checks.

## Integration with the environment

`CatchyRunEnv.__init__` dispatches by `trainee_role`:

```python
if trainee_role == "runner":
    self.reward_shaper = RunnerRewardShaper()
else:
    self.reward_shaper = CatcherRewardShaper()
```

`CatchyRunEnv._shape_reward` is a one-line delegation to whichever shaper was
bound. Run `rl_agent/trace_rewards.py` with `trainee_role="catcher"` to
confirm component signs and magnitudes; the trace tool reports the two per-step
components (`special_defense`, `captured_square`) alongside `base`. The catch
bonus only appears on terminal kills, so it shows up in the per-episode reward
total rather than the per-step component table.
