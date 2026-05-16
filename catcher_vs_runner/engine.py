"""Catcher vs. Runner — pure game engine.

Contract (load-bearing for the future Gymnasium / PettingZoo wrapper):

* Imports only `numpy` and the standard library. Importing this module has
  zero side effects.
* `GameState` is an immutable, hashable, picklable dataclass holding the
  full state, including in-flight projectiles fired by the catcher.
* `step(state, action)` is a pure function — it does not mutate `state`.
* The action space is a frozen size-16 discrete encoding (see `actions.py`).
* `legal_action_mask(state)` returns a `(16,)` boolean array. `step` raises
  `ValueError` on illegal actions, so a buggy wrapper fails loudly.
* `encode_observation(state, perspective)` returns an
  `(OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)` float32 tensor. `perspective` is
  `"runner"` or `"catcher"` and swaps own / opponent channels — the same
  observation shape works for both roles.
* `reset(seed=None)` returns a starting state. The seed controls the
  per-game special-square sampler (`seed=None` ⇒ OS entropy, fresh layout
  each call; an explicit int reproduces the same layout deterministically).

Roles:
* Runner's special action is sprint — a 3-cell cardinal jump that consumes
  one sprint charge. Charges are refilled by capturing special squares.
* Catcher's special action is shoot — spawns a projectile in any of 8
  directions. Projectiles advance one cell per half-turn, despawn off-board,
  and end the game in the catcher's favor on contact with the runner.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Literal, Optional

import numpy as np

from .actions import (
    ACTION_NAMES,
    ACTION_SPACE_SIZE,
    CARDINAL_DIRS,
    DIRECTIONS_8,
    MOVE_ACTIONS,
    SPECIAL_ACTIONS,
    direction_of,
)

# --- Tunable game parameters ---------------------------------------------

BOARD_SIZE: int = 7
TURN_LIMIT: int = 40
SPRINT_CHARGES: int = 2
RUNNER_START: tuple[int, int] = (3, 0)
CATCHER_START: tuple[int, int] = (3, BOARD_SIZE - 1)

# The runner must capture a majority of the per-game-sampled special
# squares (see _sample_special_squares) by stepping onto them. Capture is
# permanent. At turn TURN_LIMIT (no catch), runner wins iff it owns at
# least SPECIAL_MAJORITY of them.
SPECIAL_MAJORITY: int = 2

OBS_CHANNELS: int = 7

Agent = Literal["runner", "catcher"]
Position = tuple[int, int]
Projectile = tuple[Position, tuple[int, int]]  # (position, direction)


# --- State ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameState:
    """Full game state. Frozen — all transitions return a new instance."""

    runner_pos: Position
    catcher_pos: Position
    sprint_charges: int
    projectiles: frozenset[Projectile]
    special_squares: frozenset[Position]
    captured_squares: frozenset[Position]
    current_agent: Agent
    turn: int
    terminated: bool
    winner: Optional[Agent]

    def own_position(self) -> Position:
        return self.runner_pos if self.current_agent == "runner" else self.catcher_pos

    def opponent_position(self) -> Position:
        return self.catcher_pos if self.current_agent == "runner" else self.runner_pos


# --- Special-square sampler ---------------------------------------------


def _chebyshev(p1: Position, p2: Position) -> int:
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _excluded_spawn_cells() -> frozenset[Position]:
    """Spawn cells plus their in-bounds 8-neighbors. Off-limits to sampling."""
    cells: set[Position] = set()
    for start in (RUNNER_START, CATCHER_START):
        cells.add(start)
        for dx, dy in DIRECTIONS_8:
            nx, ny = start[0] + dx, start[1] + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                cells.add((nx, ny))
    return frozenset(cells)


def _sample_special_squares(rng: random.Random) -> frozenset[Position]:
    """Pick exactly 3 squares: one runner-favored, one balanced, one
    catcher-favored, all outside the spawn neighborhood. "Favored" is
    measured by Chebyshev distance from each player's start."""
    excluded = _excluded_spawn_cells()
    runner_fav: list[Position] = []
    balanced: list[Position] = []
    catcher_fav: list[Position] = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            cell = (x, y)
            if cell in excluded:
                continue
            dr = _chebyshev(cell, RUNNER_START)
            dc = _chebyshev(cell, CATCHER_START)
            if dr < dc:
                runner_fav.append(cell)
            elif dr > dc:
                catcher_fav.append(cell)
            else:
                balanced.append(cell)
    first_runner_fav = rng.choice(runner_fav)
    runner_fav.remove(first_runner_fav)
    first_catcher_fav = rng.choice(catcher_fav)
    catcher_fav.remove(first_catcher_fav)
    first_balanced = rng.choice(balanced)
    balanced.remove(first_balanced)
    second_balanced = rng.choice(balanced)
    balanced.remove(second_balanced)
    return frozenset({
        first_runner_fav,
        first_balanced,
        first_catcher_fav,
        rng.choice(runner_fav),
        second_balanced,
        rng.choice(catcher_fav),
        rng.choice(balanced),
    })


# --- Lifecycle -----------------------------------------------------------


def reset(seed: Optional[int] = None) -> GameState:
    rng = random.Random(seed)
    return GameState(
        runner_pos=RUNNER_START,
        catcher_pos=CATCHER_START,
        sprint_charges=SPRINT_CHARGES,
        projectiles=frozenset(),
        special_squares=_sample_special_squares(rng),
        captured_squares=frozenset(),
        current_agent="runner",
        turn=0,
        terminated=False,
        winner=None,
    )


def clone(state: GameState) -> GameState:
    # GameState is frozen and uses immutable containers, so the same
    # instance is safe to share. Helper exists for wrapper-author clarity.
    return state


# --- Legality ------------------------------------------------------------


def _in_bounds(pos: Position) -> bool:
    x, y = pos
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def _is_legal(state: GameState, action: int) -> bool:
    if state.terminated:
        return False
    if not 0 <= action < ACTION_SPACE_SIZE:
        return False

    pos = state.own_position()
    other = state.opponent_position()
    actor = state.current_agent
    dx, dy = direction_of(action)

    if action in MOVE_ACTIONS:
        target = (pos[0] + dx, pos[1] + dy)
        if not _in_bounds(target):
            return False
        # Runner refuses to volunteer capture; catcher landing on runner = win.
        if actor == "runner" and target == other:
            return False
        return True

    if action in SPECIAL_ACTIONS:
        if actor == "runner":  # Sprint — 3-cell cardinal jump only.
            if (dx, dy) not in CARDINAL_DIRS:
                return False
            if state.sprint_charges <= 0:
                return False
            mid = (pos[0] + dx, pos[1] + dy)
            dest = (pos[0] + 3 * dx, pos[1] + 3 * dy)
            if not _in_bounds(dest):
                return False
            # Cannot sprint through or onto the catcher.
            if mid == other or dest == other:
                return False
            return True
        # Shoot (catcher) — fires a projectile in any of the 8 directions.
        # Legal iff there is at least one in-bounds cell for the bullet to
        # travel into; otherwise the shot would immediately leave the board.
        first_cell = (pos[0] + dx, pos[1] + dy)
        return _in_bounds(first_cell)

    return False


def legal_action_mask(state: GameState) -> np.ndarray:
    """Boolean mask of shape (16,). All False if state is terminal."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    if state.terminated:
        return mask
    for a in range(ACTION_SPACE_SIZE):
        mask[a] = _is_legal(state, a)
    return mask


# --- Projectile helpers --------------------------------------------------


def _tick_projectiles(projectiles: frozenset[Projectile]) -> frozenset[Projectile]:
    """Advance every projectile by one cell. Off-board projectiles despawn.

    Runner-collision is handled by the caller after the tick using the
    returned projectile set, so this function is pure over projectiles.
    """
    new_projectiles: set[Projectile] = set()
    for pos, direction in projectiles:
        new_pos: Position = (pos[0] + direction[0], pos[1] + direction[1])
        if not _in_bounds(new_pos):
            continue
        new_projectiles.add((new_pos, direction))
    return frozenset(new_projectiles)


# --- Transition ----------------------------------------------------------


def _apply(state: GameState, action: int) -> tuple[GameState, float, float]:
    """Compute the post-action state. Caller must have verified legality."""
    actor = state.current_agent
    rx, ry = state.runner_pos
    cx, cy = state.catcher_pos
    sprint = state.sprint_charges
    projectiles = state.projectiles
    captured = state.captured_squares

    dx, dy = direction_of(action)
    if action in MOVE_ACTIONS:
        if actor == "runner":
            rx, ry = rx + dx, ry + dy
        else:
            cx, cy = cx + dx, cy + dy
    elif action in SPECIAL_ACTIONS:
        if actor == "runner":  # Sprint — cardinal jump.
            rx, ry = rx + 3 * dx, ry + 3 * dy
            sprint -= 1
        else:  # Shoot — spawn projectile at catcher's cell with direction.
            projectiles = projectiles | {((cx, cy), (dx, dy))}

    if actor == "runner" and (rx, ry) in state.special_squares and (rx, ry) not in captured:
        captured = captured | {(rx, ry)}
        if sprint < SPRINT_CHARGES:
            sprint += 1


    projectiles = _tick_projectiles(projectiles)

    new_turn = state.turn + 1
    next_agent: Agent = "catcher" if actor == "runner" else "runner"

    caught_by_move = (rx, ry) == (cx, cy)
    bullet_hit_runner = any(p == (rx, ry) for (p, _) in projectiles)
    caught = caught_by_move or bullet_hit_runner
    timed_out = (not caught) and new_turn >= TURN_LIMIT
    terminated = caught or timed_out

    winner: Optional[Agent]
    if caught:
        winner = "catcher"
        reward_runner, reward_catcher = -1.0, 1.0
    elif timed_out:
        if len(captured) >= SPECIAL_MAJORITY:
            winner = "runner"
            reward_runner, reward_catcher = 1.0, -1.0
        else:
            winner = "catcher"
            reward_runner, reward_catcher = -1.0, 1.0
    else:
        winner = None
        reward_runner, reward_catcher = 0.0, 0.0

    new_state = GameState(
        runner_pos=(rx, ry),
        catcher_pos=(cx, cy),
        sprint_charges=sprint,
        projectiles=projectiles,
        special_squares=state.special_squares,
        captured_squares=captured,
        current_agent=next_agent,
        turn=new_turn,
        terminated=terminated,
        winner=winner,
    )
    return new_state, reward_runner, reward_catcher


def step(state: GameState, action: int) -> tuple[GameState, float, float, bool, bool, dict]:
    """Apply `action` to `state`. Pure: `state` is not mutated.

    Returns: (new_state, reward_runner, reward_catcher, terminated,
    truncated, info). `truncated` is always False here — the turn limit
    is encoded as `terminated=True` with `winner` set by `SPECIAL_MAJORITY`,
    since for this game it is a real terminal condition, not a wrapper-side
    truncation.
    """
    if state.terminated:
        raise ValueError("Cannot step a terminal state — call reset() first.")
    if not 0 <= action < ACTION_SPACE_SIZE:
        raise ValueError(f"Action {action} out of range [0, {ACTION_SPACE_SIZE}).")
    if not _is_legal(state, action):
        raise ValueError(
            f"Illegal action {action} ({ACTION_NAMES[action]}) "
            f"for {state.current_agent} at turn {state.turn}."
        )

    new_state, r_runner, r_catcher = _apply(state, action)
    return new_state, r_runner, r_catcher, new_state.terminated, False, {}


# --- Observation ---------------------------------------------------------


def encode_observation(state: GameState, perspective: Agent) -> np.ndarray:
    """Return an `(OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)` float32 tensor.

    Channels:
        0: own position (one-hot)
        1: opponent position (one-hot)
        2: own charges remaining (normalized scalar broadcast)
        3: opponent charges remaining (normalized scalar broadcast)
        4: turn number (normalized scalar broadcast)
        5: projectile mask (1.0 in any cell containing >= 1 projectile)
        6: uncaptured special squares (1.0 at each remaining target cell)

    `perspective` swaps own / opponent positions so a single network can
    play either role. Coordinates: array is indexed [channel, y, x].

    Charge channels: only the runner has a depletable charge (sprint). The
    catcher's shoot is unlimited, so its charge channel is constant 0. The
    slot is preserved so the tensor shape is symmetric across perspectives.
    """
    if perspective not in ("runner", "catcher"):
        raise ValueError(f"perspective must be 'runner' or 'catcher', got {perspective!r}")

    obs = np.zeros((OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    if perspective == "runner":
        own_pos, opp_pos = state.runner_pos, state.catcher_pos
        own_charges_norm = state.sprint_charges / max(SPRINT_CHARGES, 1)
        opp_charges_norm = 0.0
    else:
        own_pos, opp_pos = state.catcher_pos, state.runner_pos
        own_charges_norm = 0.0
        opp_charges_norm = state.sprint_charges / max(SPRINT_CHARGES, 1)

    obs[0, own_pos[1], own_pos[0]] = 1.0
    obs[1, opp_pos[1], opp_pos[0]] = 1.0
    obs[2, :, :] = own_charges_norm
    obs[3, :, :] = opp_charges_norm
    obs[4, :, :] = state.turn / TURN_LIMIT
    for (px, py), _ in state.projectiles:
        obs[5, py, px] = 1.0
    for (sx, sy) in state.special_squares:
        if (sx, sy) not in state.captured_squares:
            obs[6, sy, sx] = 1.0

    return obs


# --- Convenience helpers -------------------------------------------------


def with_overrides(state: GameState, **overrides) -> GameState:
    """Return a copy of `state` with the given fields replaced.

    Intended for tests and scripted scenarios — not for use inside `step`.
    """
    return replace(state, **overrides)
