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

| Attribute                       | Value | What it controls                                                                                       |
|---------------------------------|-------|--------------------------------------------------------------------------------------------------------|
| `CAPTURE_BLOCK_PENALTY`         | 0.20  | Penalty per special the runner captured this transition. Direct anti-progress signal.                  |
| `DISTANCE_CLOSURE_COEFF`        | 0.02  | Numerator of the inverse-distance bonus. Small on purpose — terminal kill is the dominant chase signal. |
| `BULLET_COVERAGE_COEFF`         | 0.10  | Numerator of the per-bullet coverage bonus.                                                            |
| `BULLET_COVERAGE_CAP`           | 3     | Only the `N` bullets closest to the runner contribute to the coverage bonus. Prevents spam inflation.  |
| `SPECIAL_DEFENSE_COEFF`         | 0.10  | Flat bonus per bullet whose forward path threatens a special the runner is about to capture.           |
| `SPECIAL_DEFENSE_LOOKAHEAD`     | 2     | Cells along each bullet's path to scan. Covers cheb 2-3 from the catcher for a freshly fired bullet.   |
| `SPECIAL_DEFENSE_CAP`           | 2     | At most `N` qualifying items (bullet-path or catcher-ray) contribute to the defense bonus per step.    |
| `SPECIAL_DEFENSE_RAY_MIN_DIST`  | 4     | Lower Chebyshev bound (inclusive) for the catcher-ray scan that backs up the bullet-path check.        |
| `SPECIAL_DEFENSE_RAY_MAX_DIST`  | 5     | Upper Chebyshev bound (inclusive) for the catcher-ray scan.                                            |

The runner-threat side of the qualification is no longer a fixed
Chebyshev radius — it is a structural reach predicate covering both
step and sprint moves. See the "Special-defense bonus" section below
for the exact definition.

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
       + SPECIAL_DEFENSE_COEFF · qualifying                              # special defense (bullet-path + catcher-ray)
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

### Special-defense bonus — `_special_defense_bonus(curr)`

The bonus pools two categories of qualifying items into a single
shared cap (`SPECIAL_DEFENSE_CAP`). The bullet-path scan runs first;
if it doesn't saturate the cap, the catcher-ray scan tops it up.

```
qualifying = 0

# (A) Bullet-path scan — imminent threats already in flight.
for each bullet (px, py), (dx, dy) in curr.projectiles:
    for k in 1 .. SPECIAL_DEFENSE_LOOKAHEAD:
        cell = (px + k·dx, py + k·dy)
        if cell is an uncaptured special
           and runner can reach cell in one turn (step or legal sprint):
            qualifying += 1
            break       # count each bullet at most once
    if qualifying ≥ SPECIAL_DEFENSE_CAP:
        break

# (B) Catcher-ray scan — "good shot is available" on a slightly
# slower threat, independent of whether a bullet exists yet.
if qualifying < SPECIAL_DEFENSE_CAP:
    for (dx, dy) in DIRECTIONS_8:
        for k in SPECIAL_DEFENSE_RAY_MIN_DIST .. SPECIAL_DEFENSE_RAY_MAX_DIST:
            cell = (catcher_x + k·dx, catcher_y + k·dy)
            if cell out of bounds:
                continue
            if cell is an uncaptured special
               and runner can reach cell in exactly two turns:
                qualifying += 1
                break       # count each direction at most once
        if qualifying ≥ SPECIAL_DEFENSE_CAP:
            break

bonus = SPECIAL_DEFENSE_COEFF · qualifying
```

The "runner can reach in one turn" predicate is encoded directly
rather than as a Chebyshev radius:

```
runner_reaches_in_one(state, target):
    if cheb(state.runner_pos, target) ≤ 1:
        return True                              # 8-directional step
    if state.sprint_charges ≤ 0:
        return False
    dx, dy = target − state.runner_pos
    if not ((dx == 0 and |dy| == 3) or (dy == 0 and |dx| == 3)):
        return False                             # sprint is cardinal-3 only
    one_ahead = state.runner_pos + sign(dx, dy)
    if one_ahead == state.catcher_pos:           # sprint path blocked
        return False
    if target == state.catcher_pos:              # destination blocked
        return False
    return True
```

This term exists because the bullet-coverage bonus rewards proximity to
the runner directly — it pulls the catcher into "shoot wherever the
runner is moving" behavior. But the runner's actual *goal* is the special
squares, not the squares it currently occupies. A well-played catcher
shouldn't only chase the runner; it should pre-fire onto contested
specials when the runner is one step from capturing them.

The qualification check has three parts and all three must hold:

1. **A cell on the bullet's forward path** within `SPECIAL_DEFENSE_LOOKAHEAD`
   cells. Since bullets only travel in the 8 cardinal/diagonal
   directions, this is automatically equivalent to "the special is on
   one of the 8 rays from the bullet's current position" — the
   "shootable direction" requirement.
2. **That cell is an uncaptured special.** Captured specials don't need
   defending.
3. **The runner can reach that cell in one turn**, by either an
   8-directional step (`cheb(runner, cell) ≤ 1`) or a *legal* sprint.
   A sprint is legal when `sprint_charges > 0`, the move is a cardinal
   3-cell jump, the one-cell-ahead cell isn't occupied by the catcher,
   and the destination cell isn't occupied by the catcher. Sprint is
   included because a sprint-reachable special is just as imminently
   capturable as a step-reachable one — both are "one move away" by
   the engine's action set, and the catcher should treat them
   symmetrically when deciding where to fire. The predicate is
   encoded directly rather than as a Chebyshev radius because sprint
   is cardinal-only with charge and path constraints: a generic
   radius would over-count (it would include diagonals at distance 2
   and 3, which sprint cannot reach) and under-count (it would miss
   the cardinal-3 cells that sprint *can* reach unless the radius is
   widened, in which case it picks up everything in between). The
   predicate also refuses to mark a special as threatened when the
   runner physically cannot sprint to it next turn (catcher blocks
   the path, no charges remaining), so the bonus only fires for
   shots aimed at specials the runner could *actually* land on.

For a *freshly fired* bullet (current position `catcher + dir`), the
`LOOKAHEAD = 2` window scans cells at cheb 2 and 3 from the catcher — so
the bonus fires for shots aimed at specials 2 or 3 squares away in any
of the 8 directions, provided the runner is poised to capture them. For
*older* bullets in flight, the same lookahead scans the next two cells
of the bullet's trajectory regardless of where the catcher now stands.

This lookahead matches the sprint geometry cleanly: sprint-reachable
specials sit at cardinal distance 3 from the runner, and a bullet
aimed cardinally at such a special will land its second forward cell
on the special when fired from cheb-distance 3 — so the predicate's
sprint clause and the lookahead window cover the same shots without
extra tuning.

**Catcher-ray supplement.** The bullet-path scan only fires when the
catcher has *already* committed a bullet whose forward two cells land
on a runner-1-reachable special. That misses a class of situations the
shaper wants to credit: the catcher is standing 4-5 cells from an
uncaptured special on one of its 8 firing rays, and the runner is two
moves away from that special. The shot isn't fired yet, but the
*shot is available* — a SHOOT action in that direction this turn would
plant a bullet whose next cells track straight into the contested
square. Without this supplement the sparse expectation marks such
"setup" turns as wasted, exactly the failure mode this shaper exists
to prevent.

The supplement scans the catcher's 8 rays at cheb 4 and 5 and counts a
direction as qualifying when the cell on the ray is an uncaptured
special the runner can reach in *exactly* two turns (the
`_runner_reaches_in_exactly_two` predicate). The "exactly two" half is
load-bearing — 1-reach cells are already covered by the bullet-path
scan, and 3+ reach is too distant to credibly pressure right now.
"Exactly two" is computed by enumerating the runner's 12 legal first
moves (8 steps + 4 sprints, respecting bounds, catcher-blocking, and
sprint charges, with the +1 charge if the first move lands on a
special), and asking whether any of those intermediate positions
yields a 1-turn reach onto the target assuming the catcher stays put.
Catcher-side movement is approximated as stationary because the
shaping signal is about the catcher's *opportunity now*, not a
guaranteed kill.

The `[4, 5]` window is chosen to mesh with the lookahead window of the
bullet scan and the geometry of a 2-move runner reach. Runner-2-reach
cells live at cheb 2 from the runner (two steps), or cheb up to ~4 if
sprint is involved; with the runner and catcher separated by a
typical mid-board distance, the contested cells sit at cheb 4-5 from
the catcher. Below 4 those cells would already qualify under
"runner-1-reach" via the bullet path scan if a bullet were on the way;
above 5 the catcher's shot takes too long to arrive relative to a
2-move runner. Counting each direction at most once mirrors the
"count each bullet at most once" pattern from the bullet-path scan.

The two categories share the cap (`SPECIAL_DEFENSE_CAP = 2`), so the
per-step ceiling on this term remains `+0.20`. The bullet-path scan
runs first; the ray supplement only contributes if the bullet-path
scan didn't already saturate the cap. That ordering makes the cap
soak up the supplement's added qualifying frequency without inflating
the worst-case contribution — the calibration math in the next
section is unchanged.

Why a flat per-bullet bonus instead of an inverse-distance formula like
the coverage term? The defense condition is binary: either the bullet
threatens a runner-imminent special or it doesn't. There's no "almost
defending" a special — the bullet is either on a defensive line or it
isn't. A flat bonus reflects this cleanly without spuriously rewarding
near-misses.

The cap (`SPECIAL_DEFENSE_CAP = 2`) bounds the per-step contribution at
`2 · 0.10 = +0.20` even when multiple bullets happen to converge on
multiple contested specials simultaneously, keeping the term from
runaway-rewarding chaotic mid-game bullet clouds. The cap also
absorbs the modest increase in qualifying frequency introduced by the
sprint clause: more cells satisfy part 3 per turn — especially in the
early game when the runner has all 3 sprint charges and several
cardinal-3 lines into uncaptured specials — but the per-step
contribution is still bounded at `+0.20`, so the calibration math in
the next section is unchanged.

## Why these magnitudes (calibration)

A few worst-case sums to confirm the shaping never out-votes the terminal
`±1`:

- **Per-step ceiling.** Catcher adjacent to runner (closure `+0.02`),
  three bullets each one cell from the runner's next-tick path (coverage
  `+0.30`), two bullets simultaneously threatening contested specials
  (defense `+0.20`), no capture this transition: shaping `≈ +0.52`. Still
  below the `+1` terminal kill reward — a kill remains more rewarding
  than maintaining maximum pressure. In practice the coverage and
  defense terms rarely both saturate, since the bullets that maximize
  one tend not to maximize the other.
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
