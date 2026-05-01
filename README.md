# Catcher vs. Runner

A two-agent adversarial game on a 9×9 grid. Catcher tries to land on the runner; runner tries to survive 30 turns.

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

- **Board:** 9×9. Runner starts at `(0, 0)`, catcher at `(8, 8)`. Runner moves first.
- **Capture:** catcher wins by ending its turn on the runner's cell.
- **Survival:** runner wins if turn **65** ends without capture. (Original design value was 30; tuned to 65 because at 30 the corner-to-corner Manhattan distance of 16 is unreachable in 15 catcher half-moves under same-speed greedy play, which produces a ~90% runner win rate. 65 lands at ~53% under the heuristic baselines — see `python -m catcher_vs_runner.balance`.)
- **Movement:** 1 cell orthogonally per turn (default).
- **Walls:** orthogonal blockers. Runner can have **4** active on the board, catcher **3**. Place on adjacent cell (costs the turn). Remove your own (costs the turn). Cannot be destroyed by the opponent.
- **Sprint** *(runner only)*: move 2 cells in a straight orthogonal line, both intermediate and destination empty. **3 charges per match.**
- **Vault** *(catcher only)*: jump over an adjacent wall (any color), landing on the empty cell directly beyond. **3 charges per match.**

## Action encoding (frozen — wrapper authors rely on this)

The discrete action space has size **17**. Indices are stable across roles; semantics of indices 12–15 depend on the current agent.

| Index | Action | Notes |
|---|---|---|
| 0 | Move N | y - 1 |
| 1 | Move E | x + 1 |
| 2 | Move S | y + 1 |
| 3 | Move W | x - 1 |
| 4 | Place wall N | adjacent cell to the north |
| 5 | Place wall E | |
| 6 | Place wall S | |
| 7 | Place wall W | |
| 8 | Remove own wall N | only walls placed by the current agent |
| 9 | Remove own wall E | |
| 10 | Remove own wall S | |
| 11 | Remove own wall W | |
| 12 | Special N | Sprint N (runner) / Vault N (catcher) |
| 13 | Special E | |
| 14 | Special S | |
| 15 | Special W | |
| 16 | Wait | always legal |

Coordinates: `(0, 0)` is top-left; `+x` east, `+y` south.

## Engine contract (for the Gymnasium / PettingZoo wrapper)

`catcher_vs_runner.engine` exposes:

- `GameState` — frozen dataclass holding the full state.
- `reset(seed: int | None = None) -> GameState`
- `step(state, action) -> (new_state, reward_runner, reward_catcher, terminated, truncated, info)` — pure, does not mutate input.
- `legal_action_mask(state) -> np.ndarray[bool]` of shape `(17,)`. Illegal actions raise on `step`.
- `encode_observation(state, perspective: str) -> np.ndarray[float32]` of shape `(C, 9, 9)`. `perspective` is `"runner"` or `"catcher"`.
- `clone(state) -> GameState`.

The engine has no GUI, file, or global state. Importing `engine` is side-effect free and depends only on `numpy` + the standard library.

## Project layout

```
catcher_vs_runner/
  engine.py            Pure game logic.
  actions.py           Action index constants + ACTION_NAMES.
  agents/
    heuristic.py       Greedy baselines.
  render/
    pygame_app.py      pygame renderer + click handlers.
  main.py              Entry point.
tests/
  test_engine.py
```
