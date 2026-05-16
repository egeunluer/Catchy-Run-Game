# Catcher vs. Runner

A two-agent adversarial game on a 7×7 grid. The runner has to capture
enough special squares while staying out of the catcher's way — or the
catcher will close the distance and end them, by step or by bullet.

## Install & launch

```bash
pip install -e .
python -m catcher_vs_runner
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
- **Turn limit:** 40 half-turns. If no capture has happened by then, the runner wins iff they have captured at least **2** special squares; otherwise the catcher wins on timeout.
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

## Engine contract (for the Gymnasium / PettingZoo wrapper)

`catcher_vs_runner.engine` exposes:

- `GameState` — frozen dataclass holding the full state.
- `reset(seed: int | None = None) -> GameState` — explicit `int` reproduces the same special-square layout deterministically; `None` uses OS entropy.
- `step(state, action) -> (new_state, reward_runner, reward_catcher, terminated, truncated, info)` — pure, does not mutate input. Raises `ValueError` on illegal actions.
- `legal_action_mask(state) -> np.ndarray[bool]` of shape `(16,)`. All-`False` for terminal states.
- `encode_observation(state, perspective: str) -> np.ndarray[float32]` of shape `(7, 7, 7)`. `perspective` is `"runner"` or `"catcher"` and swaps own/opponent channels so a single network can play either role.
- `clone(state) -> GameState`.
- `with_overrides(state, **fields) -> GameState` — return a copy with selected fields replaced (for tests and scripted scenarios only; do not use inside `step`).

The engine has no GUI, file, or global state. Importing `engine` is side-effect free and depends only on `numpy` + the standard library.

`truncated` is always `False`: the turn limit is encoded as `terminated=True` with the winner set by the special-square majority rule, since the timeout is a real terminal condition rather than wrapper-side truncation.

### Observation channels

Indexed `[channel, y, x]`. Shape `(7, 7, 7)`.

| Channel | Content |
|---|---|
| 0 | Own position (one-hot) |
| 1 | Opponent position (one-hot) |
| 2 | Own charges remaining, normalized broadcast |
| 3 | Opponent charges remaining, normalized broadcast |
| 4 | Turn number, normalized broadcast (`turn / TURN_LIMIT`) |
| 5 | Projectile mask — `1.0` in any cell with at least one in-flight bullet |
| 6 | Uncaptured special squares |

The catcher's "charges" channel is always 0 — its shoot is unlimited. The slot is kept so the tensor stays symmetric across perspectives.

Rewards: `±1.0` to the winner / loser on termination, `0.0` on every other step.

## Heuristic baseline

```bash
python -m catcher_vs_runner.balance
```

Runs the bundled heuristic agents against each other and reports win rates with a 95% CI. Use it as a balance sanity-check whenever you change game parameters.

## Project layout

```
catcher_vs_runner/
  engine.py            Pure game logic.
  actions.py           Action index constants + ACTION_NAMES.
  balance.py           Heuristic-vs-heuristic harness.
  agents/
    heuristic.py       Greedy baselines.
  render/
    pygame_app.py      pygame renderer + click handlers.
  main.py              Entry point.
tests/
  test_engine.py
```
