# Catcher Reward Shaping

The engine emits the same sparse `±1` terminal signal for the catcher as it
does for the runner: `+1` when the catcher wins (kill by step, kill by
projectile, or timeout with the runner below `SPECIAL_MAJORITY`), `-1`
otherwise, `0` on every non-terminal step. Unlike the runner, the catcher's
terminal events can happen mid-episode (a kill terminates immediately on
contact), so the credit-assignment problem is shorter.

This shaper is deliberately **lean**. An earlier version layered four
dense components (special-blocking attraction, chase/cornering, time
advantage, defensive bullets) on top of the sparse signal as raw per-step
bonuses. The catcher learned to farm those terms instead of doing its job —
classic reward hacking, because a standing per-turn reward accumulates with
episode length and can out-earn the terminal verdict. The current design keeps
only the genuine task signals and routes every dense term through a
**potential function**, so it telescopes and cannot accumulate:

1. **catch the runner** — by far the biggest reward (terminal),
2. **defend the special squares** — a potential term that costs the catcher
   exactly when the runner grabs one,
3. **close the distance** — a potential term that pays the catcher for shrinking
   the Manhattan gap to the runner (cornering pressure),
4. **shoot well** — keep the well-aimed-bullet reward, penalize every other
   shot at least as hard.

The shift from raw per-step bonuses to **potential-based shaping**
(Ng, Harada & Russell 1999) is the core of the rework. A potential term
`F = γ·Φ(s′) − Φ(s)` telescopes over an episode to `γ^T·Φ_T − Φ_0`, a constant
independent of trajectory length, so it provably cannot change the optimal
policy and — crucially here — **cannot out-earn the terminal `±1` no matter how
long the episode runs**. That is what makes "the terminal reward always
dominates" a structural guarantee rather than a coefficient-tuning hope. We use
`γ = 1` in the potential (no need to plumb the RL discount into the shaper),
which makes the per-step semantics exact: `Φ(curr) − Φ(prev)`.

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
| `BULLET_SPAM_PENALTY`      | 0.15  | Hard flat penalty on the shoot turn when the new bullet's ray fails both reach checks.       |
| `CATCH_BONUS`              | 2.0   | Standout reward (on top of the terminal `+1`) when the catcher actually catches the runner.  |
| `SQUARE_POTENTIAL_COEFF`   | 0.03  | Potential weight per uncaptured special; per-capture hit is `-0.03`.                          |
| `DISTANCE_COEFF`           | 0.01  | Potential weight on the Manhattan gap; pays `+0.01` per unit the catcher closes.             |

The shoot-ray reach checks use **Chebyshev** distance, matching the engine's
8-directional movement. The distance *potential* uses **Manhattan** distance on
purpose: it penalizes being off on both axes, which pushes the catcher to align
with and corner the runner rather than merely sharing a row or column. (Note the
catcher moves in Chebyshev, so a diagonal step closes two Manhattan units and
earns `+0.02` that turn.)

## The reward function in two parts

The catcher overrides `shape()` rather than only implementing `_compute()`:

```
shape(prev, curr, base):
    if curr.terminated:
        return base + (CATCH_BONUS if _is_catch(curr) else 0.0)
    return _compute(prev, curr, base)

_compute(prev, curr, base):
    return base
         + (Phi(curr) - Phi(prev))              # potential-based dense shaping
         + _special_defense_bonus(prev, curr)    # one-shot per fired bullet

Phi(s) = SQUARE_POTENTIAL_COEFF * (uncaptured specials)
       - DISTANCE_COEFF         * manhattan(runner, catcher)
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

**Magnitude.** A catch totals `base (+1) + CATCH_BONUS (+2) = +3`. The dense
potential shaping is bounded to `≈ ±0.33` per episode (it telescopes) and the
bullet signal is small and gated, so the policy gradient points unambiguously at
"end the episode by catching the runner." This is intentional: the catch is the
objective, and its reward should dominate everything else by design.

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

**Why the spam penalty is `≥` the bonus.** With the bonus at `0.10` and the
penalty at `0.15`, a catcher that fires at random nets a loss in expectation per
shot — firing is discouraged unless the shot clears one of the two reach checks.
Keeping the penalty at least as large as the bonus also closes the one dense
*positive* surface that is **not** potential-based: a runner camping next to a
threatened special could otherwise let the catcher re-earn `+0.10` every turn.
With `BULLET_SPAM_PENALTY ≥ SPECIAL_DEFENSE_COEFF`, net repeated shooting can't
be farmed, so the terminal-dominates guarantee holds even against a camping
runner. If you observe the catcher refusing to shoot even in clear defensive
situations, lower the penalty (but not below the bonus).

**The reach predicates** (`_runner_reaches_in_one`,
`_runner_reaches_in_exactly_two`) are structural rather than radius-based,
because sprint is cardinal-only with charge and path constraints. See the code
in `reward_shaping.py`; the gist is "enumerate every legal first move the
runner can make and check whether the target is reachable from there given
remaining sprint charges and catcher-blocking." They read `prev` because the
runner's position and sprint count are unchanged across the catcher's
half-turn, and `prev` is the state at firing time.

### Potential-based dense shaping — `Phi(curr) − Phi(prev)`

Both dense signals live in one potential, applied as its per-step difference
(`γ = 1`):

```
Phi(s) = SQUARE_POTENTIAL_COEFF · (uncaptured specials)
       − DISTANCE_COEFF         · manhattan(runner, catcher)
```

For a catcher trainee, `prev` is the state before the catcher's action and
`curr` is the state after the catcher's action **and** the runner's response, so
the difference captures the net effect of the full env-step. It decomposes into
two readable parts:

- **Square defense.** `SQUARE_POTENTIAL_COEFF · (uncaptured)` falls by exactly
  `0.03` for each special the runner grabs this step, so `Φ(curr) − Φ(prev)`
  contributes `−0.03 · newly_captured`. This is the `γ = 1` potential-based form
  of the old captured-square penalty — same gradient at the capture event, but
  with the coefficient *decoupled* from any standing per-turn income. A naive
  "+reward per uncaptured square each turn" would pay the catcher to stall (more
  surviving turns = more income), re-growing the passivity this shaper exists to
  kill; the potential form gives the catcher `≈ 0` on turns where nothing is
  captured and only the `−0.03` hit on the turn one is lost.
- **Cornering.** `−DISTANCE_COEFF · manhattan(runner, catcher)` rises as the gap
  shrinks, so closing one Manhattan unit pays `+0.01` and a diagonal step (which
  closes two) pays `+0.02`. Letting the runner slip away costs the same per unit.

**Why this can't be farmed.** A potential difference telescopes:
`Σ_t (Φ_{t+1} − Φ_t) = Φ_T − Φ_0`. Over any episode the total dense shaping is
just `Φ(last) − Φ(first)`, bounded by `max|Φ|` regardless of how many turns the
game lasts — at most `0.03·7 + 0.01·12 ≈ 0.33` in magnitude. It therefore cannot
accumulate past the terminal `±1`, which is precisely the property that makes the
win/lose verdict dominate the gradient on *every* episode, not just short ones.

**Edge case: the 7th capture.** Capturing the 7th special ends the game in the
runner's favor (`terminated=True`), which routes through the terminal branch of
`shape()` — so the potential difference does **not** fire on that final grab.
That's fine: the engine's terminal `-1` already punishes the loss.

## Calibration summary

| Signal                   | Min      | Max      | When it fires                                |
|--------------------------|----------|----------|----------------------------------------------|
| catch bonus (terminal)   | `0`      | `+2.0`   | catcher actually catches the runner          |
| `_special_defense_bonus` | `-0.15`  | `+0.10`  | a bullet is fired this turn                  |
| square potential delta   | `-0.06`  | `0`      | runner captures ≥1 special this step         |
| distance potential delta | `-0.06`  | `+0.06`  | the Manhattan gap changes (almost every step) |

Per-episode, the catch (`+3` total) dominates by design, and the dense potential
shaping is bounded to `≈ ±0.33` total no matter the episode length — so a
timeout win (`+1`) or a loss (`-1`) always keeps its sign. The only repeatable
positive *event* signal is the well-aimed bullet, gated behind two structural
reach checks and capped below the spam penalty, so there is no dense surface left
to farm.

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
confirm component signs and magnitudes; the trace tool reports the three per-step
components (`special_defense`, `square`, `distance`) alongside `base`, where
`square` and `distance` are the two halves of the potential difference. The catch
bonus only appears on terminal kills, so it shows up in the per-episode reward
total rather than the per-step component table.
