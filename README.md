# Catchy Run - And The DRL Agents

A two-agent adversarial game on a 7×7 grid. The runner has to capture
enough special squares while staying out of the catcher's way — or the
catcher will close the distance and end them, by step or by bullet.

## Install & launch

```bash
pip install -e .
python -m catchy_run_game
```

### Double-click launch (macOS)

Run once from the project root:

```bash
bash build_app.sh
```

This produces `CatcherVsRunner.app` next to `pyproject.toml`. Double-click it from Finder to launch the game with no Terminal window. The bundle is relocatable — move it to your Desktop or `/Applications` and it still works.

The launcher uses your existing `python3`, so `pygame` and `numpy` must be installed in that interpreter. If you move the project to a different directory, rerun `bash build_app.sh` so the launcher picks up the new path.

## Mechanics

- **Board:** 7×7. Runner starts at `(3, 0)`, catcher at `(3, 6)`. Runner moves first.
- **Turn limit:** 40 half-turns. If no capture has happened by then, the runner wins iff they have captured at least **4** special squares (`SPECIAL_MAJORITY`); otherwise the catcher wins on timeout. Capturing all 7 special squares also ends the game immediately in the runner's favor.
- **Capture (catcher wins immediately):** the catcher steps onto the runner's cell *or* a projectile fired by the catcher lands on the runner's cell.
- **Movement:** 8 directions (N, NE, E, SE, S, SW, W, NW), one cell per turn. Diagonals are first-class.
- **Special squares:** 7 cells sampled at game start, all outside the spawn neighborhood. The runner permanently captures one by stepping onto it; each capture also grants **+1 sprint charge**.
- **Sprint** *(runner only)*: jump 3 cells in a straight **cardinal** direction (no diagonal sprints). The 1-cell-ahead and 3-cells-ahead cells must not be the catcher, and the destination must be in-bounds. **3 charges at game start**, replenished by capturing special squares.
- **Shoot** *(catcher only)*: fire a projectile in any of the 8 directions. Bullets travel one cell per half-turn, despawn when they leave the board, and end the game in the catcher's favor on contact with the runner. **Unlimited shots.**

There are no walls and no wait action — every turn, both agents must move (or, for the catcher, shoot).

## Action encoding (frozen — wrapper authors rely on this)

The discrete action space has size **16**. All directions are ordered clockwise from N: `N, NE, E, SE, S, SW, W, NW`.

| Index | Action | Semantics |
|---|---|---|
| 0–7  | `MOVE_*`    | 1-cell step in the indexed direction (both agents). |
| 8–15 | `SPECIAL_*` | Runner → sprint (3-cell cardinal jump; consumes a charge; diagonals illegal). Catcher → shoot (spawns a projectile in the indexed direction). |

Coordinates: `(0, 0)` is top-left; `+x` east, `+y` south.

## Engine contract

`catcher_vs_runner.engine` exposes:

- `GameState` — frozen dataclass holding the full state.
- `reset(seed: int | None = None) -> GameState` — explicit `int` reproduces the same special-square layout deterministically; `None` uses OS entropy.
- `step(state, action) -> (new_state, reward_runner, reward_catcher, terminated, truncated, info)` — pure, does not mutate input. Raises `ValueError` on illegal actions.
- `legal_action_mask(state) -> np.ndarray[bool]` of shape `(16,)`. All-`False` for terminal states.
- `encode_observation(state, perspective: str) -> np.ndarray[float32]` of shape `(9, 7, 7)`. `perspective` is `"runner"` or `"catcher"` and swaps own/opponent channels so a single network can play either role.
- `clone(state) -> GameState`.
- `with_overrides(state, **fields) -> GameState` — return a copy with selected fields replaced (for tests and scripted scenarios only; do not use inside `step`).

The engine has no GUI, file, or global state. Importing `engine` is side-effect free and depends only on `numpy` + the standard library.

`truncated` is always `False`: the turn limit is encoded as `terminated=True` with the winner set by the special-square majority rule, since the timeout is a real terminal condition rather than wrapper-side truncation.

### Observation channels

Indexed `[channel, y, x]`. Shape `(9, 7, 7)`.

| Channel | Content |
|---|---|
| 0 | Own position (one-hot) |
| 1 | Opponent position (one-hot) |
| 2 | Chebyshev distance between agents, normalized broadcast (`max(|dx|, |dy|) / (BOARD_SIZE - 1)`) |
| 3 | Own sprint charges remaining, normalized broadcast (`sprint_charges / SPRINT_CHARGES`) |
| 4 | Turn number, normalized broadcast (`turn / TURN_LIMIT`) |
| 5 | Projectile presence mask — `1.0` at each in-flight bullet's cell |
| 6 | Projectile direction `dx` at the bullet's cell, in `{-1, 0, +1}` |
| 7 | Projectile direction `dy` at the bullet's cell, in `{-1, 0, +1}` |
| 8 | Uncaptured special squares |

Channel 3 carries the runner's sprint counter. The catcher has no depletable resource (shoot is unlimited), so under the catcher perspective this channel is constant 0.

Rewards: `±1.0` to the winner / loser on termination, `0.0` on every other step.

## RL training

The `rl_agent/` package wraps the engine for training. The engine's own reward is sparse — `±1` on termination, `0` otherwise — so `rl_agent/reward_shaping.py` adds per-step shaping signals on top of it. A shared `RewardShaper` base class owns common helpers; `RunnerRewardShaper` and `CatcherRewardShaper` subclass it with role-specific signals, and `environment.py` picks the right one at construction time.

The **runner** shaper layers on:

- capture bonus per special collected,
- alive bonus,
- catcher-distance penalty (heavy when in danger and not retreating, light otherwise),
- projectile threat penalty,
- attraction toward the two nearest *safe* uncaptured specials,
- sprint-waste penalty,
- urgency penalty: `-URGENCY_COEFF · (SPECIAL_MAJORITY − captured) · (turn / TURN_LIMIT)`, which gives the runner a growing nudge toward the 4-capture win threshold as the clock runs down, and disengages once that threshold is reached,
- unsafe-capture penalty: flat `-UNSAFE_CAPTURE_PENALTY` when the runner captures a special with `cheb(runner, prev.catcher_pos) ≤ 1`, sized to neutralize the capture bonus so the existing distance penalty dominates the signal on unsafe grabs.

The **catcher** shaper is deliberately lighter — its job is to scaffold projectile tactics that a sparse catcher would otherwise skip (random shots usually miss, so sparse expectation marks them as wasted turns). Three components:

- capture-block penalty per special the runner just grabbed (direct anti-progress),
- small distance-closure bonus (`+COEFF / cheb(runner, catcher)`),
- bullet-coverage bonus: per in-flight bullet, score `+COEFF / max(1, min(d₁, d₂))` against the runner's current position, summed over the `BULLET_COVERAGE_CAP` closest bullets to prevent spam inflation.

All magnitudes are tuned so the engine's terminal `±1` still dominates the win/lose verdict. See `rl_agent/environment_explanations/reward_shaping.md` and `catcher_reward_shaping.md` for the full component-by-component derivations.

## Project layout

```
catchy_run/
  engine.py            Pure game logic.
  actions.py           Action index constants + ACTION_NAMES.
  agents/
    heuristic.py       Greedy baselines.
    rl_catcher.py
    rl_runner.py
  render/
    pygame_app.py      pygame renderer + click handlers.
  main.py              Entry point.
rl_agent/
  environment.py       Gymnasium-style wrapper around the engine.
  reward_shaping.py    RewardShaper base + RunnerRewardShaper / CatcherRewardShaper subclasses.
  model.py             Policy / value network construction. Role-aware training entry point.
  custom_cnn.py        CNN feature extractor for the 9-channel observation.
  opponents.py         Opponent providers used during training.
  evaluation.py        Evaluation harness for trained policies (runner and catcher metrics).
  trace_rewards.py     Per-component trace tool for either shaper.
  environment_explanations/
    reward_shaping.md          Deep doc for RunnerRewardShaper.
    catcher_reward_shaping.md  Deep doc for CatcherRewardShaper.
    opponent_structure.md      Deep doc for opponents.py.
tests/
  test_engine.py
```
