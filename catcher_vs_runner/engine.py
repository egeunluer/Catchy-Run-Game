"""Catcher vs. Runner — pure game engine.

Contract (load-bearing for the future Gymnasium / PettingZoo wrapper):

* Imports only `numpy` and the standard library. Importing this module has
  zero side effects.
* `GameState` is an immutable, hashable, picklable dataclass holding the
  full state, including in-flight projectiles fired by the catcher.
* `step(state, action)` is a pure function — it does not mutate `state`.
* The action space is a frozen size-33 discrete encoding (see `actions.py`).
* `legal_action_mask(state)` returns a `(33,)` boolean array. `step` raises
  `ValueError` on illegal actions, so a buggy wrapper fails loudly.
* `encode_observation(state, perspective)` returns an
  `(OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)` float32 tensor. `perspective` is
  `"runner"` or `"catcher"` and swaps own / opponent channels — the same
  observation shape works for both roles.
* `reset(seed=None)` accepts a seed (currently unused; reserved for future
  randomized variants) and returns a deterministic starting state.

Roles:
* Only the runner places and removes walls (in any of 8 surrounding cells,
  capped at `RUNNER_WALL_CAP`).
* The catcher cannot place walls; its special action is shoot, which spawns
  a projectile in any of 8 directions. Projectiles advance one cell per
  half-turn, are destroyed by walls (taking the wall with them), and end
  the game in the catcher's favor on contact with the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

import numpy as np

from .actions import (
    ACTION_NAMES,
    ACTION_SPACE_SIZE,
    ACTION_WAIT,
    CARDINAL_DIRS,
    MOVE_ACTIONS,
    PLACE_WALL_ACTIONS,
    REMOVE_WALL_ACTIONS,
    SPECIAL_ACTIONS,
    direction_of,
)

# --- Tunable game parameters ---------------------------------------------

BOARD_SIZE: int = 6
TURN_LIMIT: int = 40
RUNNER_WALL_CAP: int = 2
WALL_LIFETIME: int = 4  # half-turns a wall stays on the board after placement
SPRINT_CHARGES: int = 3
RUNNER_START: tuple[int, int] = (0, 0)
CATCHER_START: tuple[int, int] = (BOARD_SIZE - 1, BOARD_SIZE - 1)

OBS_CHANNELS: int = 7

Agent = Literal["runner", "catcher"]
Position = tuple[int, int]
Projectile = tuple[Position, tuple[int, int]]  # (position, direction)
# A wall is a (cell, expiry_turn) pair. Expiry is the latest turn-index at
# which the wall is still on the board; once `state.turn > expiry_turn` it
# is pruned automatically inside `_apply`.
Wall = tuple[Position, int]


# --- State ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameState:
    """Full game state. Frozen — all transitions return a new instance.

    Walls are owned by the runner only; there is no `catcher_walls` field
    because the catcher cannot place walls. Each wall carries an expiry turn
    so it can disappear automatically `WALL_LIFETIME` half-turns after it
    was placed.
    """

    runner_pos: Position
    catcher_pos: Position
    walls: frozenset[Wall]
    sprint_charges: int
    projectiles: frozenset[Projectile]
    current_agent: Agent
    turn: int
    terminated: bool
    winner: Optional[Agent]

    def own_position(self) -> Position:
        return self.runner_pos if self.current_agent == "runner" else self.catcher_pos

    def opponent_position(self) -> Position:
        return self.catcher_pos if self.current_agent == "runner" else self.runner_pos

    def wall_positions(self) -> frozenset[Position]:
        return frozenset(p for (p, _) in self.walls)


# --- Lifecycle -----------------------------------------------------------


def reset(seed: Optional[int] = None) -> GameState:  # noqa: ARG001 — seed reserved
    return GameState(
        runner_pos=RUNNER_START,
        catcher_pos=CATCHER_START,
        walls=frozenset(),
        sprint_charges=SPRINT_CHARGES,
        projectiles=frozenset(),
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

    if action == ACTION_WAIT:
        return True

    pos = state.own_position()
    other = state.opponent_position()
    actor = state.current_agent
    dx, dy = direction_of(action)
    wall_positions = state.wall_positions()

    if action in MOVE_ACTIONS:
        target = (pos[0] + dx, pos[1] + dy)
        if not _in_bounds(target):
            return False
        if target in wall_positions:
            return False
        # Runner refuses to volunteer capture; catcher landing on runner = win.
        if actor == "runner" and target == other:
            return False
        return True

    if action in PLACE_WALL_ACTIONS:
        # Only the runner can place walls.
        if actor != "runner":
            return False
        if len(state.walls) >= RUNNER_WALL_CAP:
            return False
        target = (pos[0] + dx, pos[1] + dy)
        if not _in_bounds(target):
            return False
        if target in wall_positions:
            return False
        if target == other:
            return False
        return True

    if action in REMOVE_WALL_ACTIONS:
        # Only the runner can remove walls.
        if actor != "runner":
            return False
        target = (pos[0] + dx, pos[1] + dy)
        return target in wall_positions

    if action in SPECIAL_ACTIONS:
        if actor == "runner":  # Sprint — 2-cell cardinal jump only.
            if (dx, dy) not in CARDINAL_DIRS:
                return False
            if state.sprint_charges <= 0:
                return False
            mid = (pos[0] + dx, pos[1] + dy)
            dest = (pos[0] + 3 * dx, pos[1] + 3 * dy)
            if not _in_bounds(dest):
                return False
            if mid in wall_positions or dest in wall_positions:
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
    """Boolean mask of shape (33,). All False if state is terminal."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    if state.terminated:
        return mask
    for a in range(ACTION_SPACE_SIZE):
        mask[a] = _is_legal(state, a)
    return mask


# --- Projectile helpers --------------------------------------------------


def _tick_projectiles(
    projectiles: frozenset[Projectile],
    walls: frozenset[Wall],
) -> tuple[frozenset[Projectile], frozenset[Wall]]:
    """Advance every projectile by one cell.

    Resolution rules:
    * Off-board next cell → projectile despawns.
    * Wall on next cell → both the wall and the projectile are destroyed.
    * Otherwise → projectile continues at the new cell.

    Runner-collision is handled by the caller after the tick using the
    returned projectile set, so this function is pure over walls/projectiles.
    """
    new_projectiles: set[Projectile] = set()
    walls_by_pos: dict[Position, int] = {p: exp for (p, exp) in walls}

    for pos, direction in projectiles:
        nx, ny = pos[0] + direction[0], pos[1] + direction[1]
        new_pos: Position = (nx, ny)
        if not _in_bounds(new_pos):
            continue
        if new_pos in walls_by_pos:
            del walls_by_pos[new_pos]
            continue
        new_projectiles.add((new_pos, direction))

    return frozenset(new_projectiles), frozenset(walls_by_pos.items())


# --- Transition ----------------------------------------------------------


def _apply(state: GameState, action: int) -> tuple[GameState, float, float]:
    """Compute the post-action state. Caller must have verified legality."""
    actor = state.current_agent
    rx, ry = state.runner_pos
    cx, cy = state.catcher_pos
    walls = state.walls
    sprint = state.sprint_charges
    projectiles = state.projectiles

    if action != ACTION_WAIT:
        dx, dy = direction_of(action)
        if action in MOVE_ACTIONS:
            if actor == "runner":
                rx, ry = rx + dx, ry + dy
            else:
                cx, cy = cx + dx, cy + dy
        elif action in PLACE_WALL_ACTIONS:
            target = (rx + dx, ry + dy)
            walls = walls | {(target, state.turn + WALL_LIFETIME)}
        elif action in REMOVE_WALL_ACTIONS:
            target = (rx + dx, ry + dy)
            walls = frozenset(w for w in walls if w[0] != target)
        elif action in SPECIAL_ACTIONS:
            if actor == "runner":  # Sprint — cardinal jump.
                rx, ry = rx + 3 * dx, ry + 3 * dy
                sprint -= 1
            else:  # Shoot — spawn projectile at catcher's cell with direction.
                projectiles = projectiles | {((cx + dx, cy + dy), (dx, dy))}

    # Tick all projectiles once at the end of every half-turn.
    if actor == "runner":
        projectiles, walls = _tick_projectiles(projectiles, walls)

    new_turn = state.turn + 1
    # Prune walls whose lifetime has elapsed. A wall placed at turn N has
    # `expiry = N + WALL_LIFETIME` and stays on the board through turns
    # N+1 .. N+WALL_LIFETIME. It is removed once `new_turn > expiry`.
    walls = frozenset((p, exp) for (p, exp) in walls if exp >= new_turn)
    next_agent: Agent = "catcher" if actor == "runner" else "runner"

    captured_by_move = (rx, ry) == (cx, cy)
    bullet_hit_runner = any(p == (rx, ry) for (p, _) in projectiles)
    captured = captured_by_move or bullet_hit_runner
    timed_out = (not captured) and new_turn >= TURN_LIMIT
    terminated = captured or timed_out

    if captured:
        winner: Optional[Agent] = "catcher"
        reward_runner, reward_catcher = -1.0, 1.0
    elif timed_out:
        winner = "runner"
        reward_runner, reward_catcher = 1.0, -1.0
    else:
        winner = None
        reward_runner, reward_catcher = 0.0, 0.0

    new_state = GameState(
        runner_pos=(rx, ry),
        catcher_pos=(cx, cy),
        walls=walls,
        sprint_charges= sprint,
        projectiles=projectiles,
        current_agent=next_agent,
        turn=new_turn,
        terminated=terminated,
        winner=winner,
    )
    return new_state, reward_runner, reward_catcher


def step(
    state: GameState, action: int
) -> tuple[GameState, float, float, bool, bool, dict]:
    """Apply `action` to `state`. Pure: `state` is not mutated.

    Returns: (new_state, reward_runner, reward_catcher, terminated,
    truncated, info). `truncated` is always False here — the turn limit
    is encoded as `terminated=True` with `winner="runner"`, since for this
    game it is a real terminal condition (runner wins), not a wrapper-side
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
        2: walls (only the runner places walls, so this is shared)
        3: own charges remaining (normalized scalar broadcast)
        4: opponent charges remaining (normalized scalar broadcast)
        5: turn number (normalized scalar broadcast)
        6: projectile mask (1.0 in any cell containing >= 1 projectile)

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
    for (x, y), _expiry in state.walls:
        obs[2, y, x] = 1.0
    obs[3, :, :] = own_charges_norm
    obs[4, :, :] = opp_charges_norm
    obs[5, :, :] = state.turn / TURN_LIMIT
    for (px, py), _ in state.projectiles:
        obs[6, py, px] = 1.0

    return obs


# --- Convenience helpers -------------------------------------------------


def with_overrides(state: GameState, **overrides) -> GameState:
    """Return a copy of `state` with the given fields replaced.

    Intended for tests and scripted scenarios — not for use inside `step`.
    """
    return replace(state, **overrides)
