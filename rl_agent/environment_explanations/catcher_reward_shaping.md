# Catcher Reward Shaping

The engine emits the same sparse `±1` terminal signal for the catcher as it
does for the runner: `+1` when the catcher wins (kill by step, kill by
projectile, or timeout with the runner below `SPECIAL_MAJORITY`), `-1`
otherwise, `0` on every non-terminal step. Unlike the runner, the catcher's
terminal events can happen mid-episode (a kill terminates immediately on
contact), so the credit-assignment problem is shorter and a naive sparse
catcher will reliably learn the *stepping* part of the game.

The reason the catcher gets shaping anyway is **skill scaffolding for
projectile tactics**. The bundled heuristic catcher (`agents/heuristic.py`)
scores every action by the resulting Chebyshev distance to the runner.
Shooting doesn't move the catcher, so the heuristic only shoots when the
bullet immediately one-shot-kills — which collapses behaviorally into the
"step onto the runner" case. A sparse RL catcher trained against the
heuristic runner will likely arrive at a similar policy: pure chase,
projectiles almost never used, because random shots usually miss and
sparse expectation marks them as wasted turns. The shaping signals below
exist specifically to teach the catcher that **a well-aimed bullet is
worth firing** even when it doesn't kill that turn — it controls space and
threatens the runner's chosen path.

`CatcherRewardShaper` lives alongside `RunnerRewardShaper` in
`rl_agent/reward_shaping.py`. Both subclass a shared `RewardShaper` base
that owns the `_cheb` helper and the public `shape(prev, curr, base)`
method (which short-circuits on `curr.terminated` and otherwise delegates
to the subclass's `_compute`). The environment picks the right shaper at
construction time based on `trainee_role`.

## Tunables (class attributes)

| Attribute                 | Value | What it controls                                                                                       |
|---------------------------|-------|--------------------------------------------------------------------------------------------------------|
| `CAPTURE_BLOCK_PENALTY`   | 0.20  | Penalty per special the runner captured this transition. Direct anti-progress signal.                  |
| `DISTANCE_CLOSURE_COEFF`  | 0.02  | Numerator of the inverse-distance bonus. Small on purpose — terminal kill is the dominant chase signal. |
| `BULLET_COVERAGE_COEFF`   | 0.10  | Numerator of the per-bullet coverage bonus.                                                            |
| `BULLET_COVERAGE_CAP`     | 3     | Only the `N` bullets closest to the runner contribute to the coverage bonus. Prevents spam inflation.  |

Distances are **Chebyshev** throughout, matching the engine's 8-directional
movement (one step in any direction equals one unit of Chebyshev distance).

## The reward function in one expression

For a catcher trainee, given `prev` (state before the trainee's action),
`curr` (state after the trainee's action *and* the opponent's response —
i.e. `self.state` inside `env.step()`), and the engine's `base` reward:

```
shaped = base
       − CAPTURE_BLOCK_PENALTY · newly_captured                          # anti-progress
       + DISTANCE_CLOSURE_COEFF / max(1, cheb(runner, catcher))          # distance closure
       + Σ over top-K bullets:  BULLET_COVERAGE_COEFF / max(1, min(d₁, d₂))   # bullet coverage
```

where `K = BULLET_COVERAGE_CAP` and `d₁`, `d₂` are Chebyshev distances
from the runner's current position to a bullet's next cell and the cell
after that. `newly_captured` is `|curr.captured_squares| −
|prev.captured_squares|` — `0` or `1` per transition.

## Component by component

### Capture-block penalty — `_capture_block_penalty(prev, curr)`

```
−CAPTURE_BLOCK_PENALTY · ( |curr.captured_squares| − |prev.captured_squares| )
```

The runner's progress is the catcher's anti-progress. This is the most
direct mid-episode signal that something the catcher could have prevented
just happened. The magnitude (`0.20`) is chosen to roughly mirror the
runner's `CAPTURE_BONUS = 0.24` — a special captured costs the catcher
about what it pays the runner. We keep it slightly smaller than the
runner's bonus so the terminal `±1` still dominates over an episode (a
catcher conceding all 7 specials still only accumulates `−1.4` from this
term across the whole episode, comparable but below the `+1` if it
somehow wins anyway, e.g. by timeout — terminal dominance preserved).

### Distance closure bonus — `_distance_closure_bonus(curr)`

```
+DISTANCE_CLOSURE_COEFF / max(1, cheb(curr.runner_pos, curr.catcher_pos))
```

A small inverse-distance bonus so the gradient pulls the catcher toward
the runner during early training. The terminal kill reward already
provides the strong "be close" signal; this is just denser feedback to
speed up early convergence. **The magnitude is deliberately tiny** — too
large and the catcher learns to chase greedily and never shoots,
recapitulating the exact heuristic failure mode this shaper exists to
escape. At cheb 1 the bonus is `+0.02`; at cheb 6 it's `≈ +0.0033`. Both
are far below the bullet-coverage and capture-block magnitudes, ensuring
distance closure doesn't out-vote projectile tactics.

The `max(1, …)` clamp is a safety guard — non-terminal states already
satisfy `cheb ≥ 1`, but the clamp makes the term robust if the engine
ever exposes a `0`-distance transitional state.

### Bullet coverage bonus — `_bullet_coverage_bonus(curr)`

```
for each (px, py), (dx, dy) in curr.projectiles:
    next_cell  = (px +   dx, py +   dy)
    next2_cell = (px + 2·dx, py + 2·dy)
    d₁ = cheb(curr.runner_pos, next_cell)
    d₂ = cheb(curr.runner_pos, next2_cell)
    score_b = BULLET_COVERAGE_COEFF / max(1, min(d₁, d₂))

bonus = sum of the top BULLET_COVERAGE_CAP scores
```

This is the **structural mirror** of the runner's projectile threat
penalty (`reward_shaping.py: _projectile_threat_penalty`). Both terms
sample each bullet's next two cells (one cell per half-turn of travel) and
take the smaller Chebyshev distance to the runner. From the catcher's
perspective, a bullet whose imminent path passes close to the runner is a
*credible threat* — even if it doesn't kill, it constrains the runner's
movement next turn.

Why sample both `next_cell` and `next2_cell` — and why `min`? Two ticks
of look-ahead matches the runner's threat penalty exactly, so the
catcher and runner are scoring the same situations symmetrically.
`min(d₁, d₂)` rewards the cell where the bullet is closest to the runner
within the lookahead window — the bullet doesn't have to be threatening
the runner *now*, just *soon*.

**The cap (`BULLET_COVERAGE_CAP = 3`) is essential.** Shoot is unlimited
in the engine. Without the cap, a catcher could learn to spam shots to
inflate the shaping bonus rather than to actually pressure the runner.
Capping at the three closest bullets keeps the per-step contribution
bounded at roughly `3 · 0.10 = 0.30` even in the pathological "every
bullet sitting on the runner" case. Three is enough to credit multi-shot
coverage of distinct runner-escape lanes but small enough to make spam
non-productive.

The cap is implemented by sorting per-bullet scores descending and
summing the prefix — bullets that are *closer* to the runner contribute
preferentially.

## Why these magnitudes (calibration)

A few worst-case sums to confirm the shaping never out-votes the terminal
`±1`:

- **Per-step ceiling.** Catcher adjacent to runner (closure `+0.02`),
  three bullets each one cell from the runner's next-tick path (coverage
  `+0.30`), no capture this transition: shaping `≈ +0.32`. Below the `+1`
  terminal kill reward — a kill is always more rewarding than maintaining
  pressure.
- **Per-step floor.** Catcher far from runner (closure `≈ +0.003` at
  cheb 6), no bullets in flight, runner captured a special
  (`−0.20`): shaping `≈ −0.20`. Above the `−1` terminal loss.
- **Per-episode cumulative.** Worst-case capture conceded: `7 · (−0.20)
  = −1.4`, but a 7-capture episode terminates in the runner's favor with
  base reward `−1`, total `≈ −2.4`. Still firmly in the "lose"
  direction — shaping doesn't flip a losing trajectory positive. Best
  case (catcher chases close throughout, maintains coverage, kills mid
  episode): `~20 · (0.02 + 0.30) = +6.4` is the rough upper bound of
  shaping, but this requires sustained adjacency AND 3 bullets close to
  the runner every single step, which is practically unreachable. A
  realistic active-kill episode accumulates maybe `+0.5` of shaping
  before terminating with `+1`, total `≈ +1.5` — still clearly a "win"
  signal, with the terminal share dominant.

Terminal dominance is the property we care about: at no point does
positive shaping reward a sequence of moves that loses the game more than
the engine rewards winning it.

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
component signs and magnitudes against a trained policy or random
actions.
