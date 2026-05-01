"""Catcher vs. Runner — pure game engine.

Contract (load-bearing for the future Gymnasium / PettingZoo wrapper):

* Imports only `numpy` and the standard library. Importing this module has
  zero side effects.
* `GameState` is an immutable, hashable, picklable dataclass holding the
  full state.
* `step(state, action)` is a pure function — it does not mutate `state`.
* The action space is a frozen size-17 discrete encoding (see `actions.py`).
* `legal_action_mask(state)` returns a `(17,)` boolean array. `step` raises
  `ValueError` on illegal actions, so a buggy wrapper fails loudly.
* `encode_observation(state, perspective)` returns a `(C, 9, 9) float32`
  tensor. `perspective` is `"runner"` or `"catcher"` and swaps own / opponent
  channels — the same observation shape works for both roles.
* `reset(seed=None)` accepts a seed (currently unused; reserved for future
  randomized variants) and returns a deterministic starting state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

import numpy as np

from .actions import (
    ACTION_NAMES,
    ACTION_SPACE_SIZE,
    ACTION_WAIT,
    MOVE_ACTIONS,
    PLACE_WALL_ACTIONS,
    REMOVE_WALL_ACTIONS,
    SPECIAL_ACTIONS,
    direction_of,
)

# --- Tunable game parameters ---------------------------------------------

BOARD_SIZE: int = 9
# Half-moves total. Tuned from the original design value of 30: at 30, greedy
# heuristics produce a ~90% runner win rate because the catcher cannot close
# 16 cells in 15 half-moves under same-speed greedy play. 65 lands at ~53%
# runner-wins under heuristic-vs-heuristic, inside the target 45-55% band.
TURN_LIMIT: int = 65
RUNNER_WALL_CAP: int = 4
CATCHER_WALL_CAP: int = 3
SPRINT_CHARGES: int = 3
VAULT_CHARGES: int = 3
RUNNER_START: tuple[int, int] = (0, 0)
CATCHER_START: tuple[int, int] = (BOARD_SIZE - 1, BOARD_SIZE - 1)

OBS_CHANNELS: int = 7

Agent = Literal["runner", "catcher"]
Position = tuple[int, int]


# --- State ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameState:
    """Full game state. Frozen — all transitions return a new instance."""

    runner_pos: Position
    catcher_pos: Position
    runner_walls: frozenset[Position]
    catcher_walls: frozenset[Position]
    sprint_charges: int
    vault_charges: int
    current_agent: Agent
    turn: int
    terminated: bool
    winner: Optional[Agent]

    def all_walls(self) -> frozenset[Position]:
        return self.runner_walls | self.catcher_walls

    def own_walls(self) -> frozenset[Position]:
        return self.runner_walls if self.current_agent == "runner" else self.catcher_walls

    def own_position(self) -> Position:
        return self.runner_pos if self.current_agent == "runner" else self.catcher_pos

    def opponent_position(self) -> Position:
        return self.catcher_pos if self.current_agent == "runner" else self.runner_pos

    def own_wall_cap(self) -> int:
        return RUNNER_WALL_CAP if self.current_agent == "runner" else CATCHER_WALL_CAP


# --- Lifecycle -----------------------------------------------------------


def reset(seed: Optional[int] = None) -> GameState:  # noqa: ARG001 — seed reserved
    return GameState(
        runner_pos=RUNNER_START,
        catcher_pos=CATCHER_START,
        runner_walls=frozenset(),
        catcher_walls=frozenset(),
        sprint_charges=SPRINT_CHARGES,
        vault_charges=VAULT_CHARGES,
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
    walls = state.all_walls()
    actor = state.current_agent
    dx, dy = direction_of(action)

    if action in MOVE_ACTIONS:
        target = (pos[0] + dx, pos[1] + dy)
        if not _in_bounds(target):
            return False
        if target in walls:
            return False
        # Runner refuses to volunteer capture; catcher landing on runner = win.
        if actor == "runner" and target == other:
            return False
        return True

    if action in PLACE_WALL_ACTIONS:
        if len(state.own_walls()) >= state.own_wall_cap():
            return False
        target = (pos[0] + dx, pos[1] + dy)
        if not _in_bounds(target):
            return False
        if target in walls:
            return False
        if target == other:
            return False
        return True

    if action in REMOVE_WALL_ACTIONS:
        target = (pos[0] + dx, pos[1] + dy)
        return target in state.own_walls()

    if action in SPECIAL_ACTIONS:
        mid = (pos[0] + dx, pos[1] + dy)
        dest = (pos[0] + 2 * dx, pos[1] + 2 * dy)
        if not _in_bounds(dest):
            return False
        if actor == "runner":  # Sprint
            if state.sprint_charges <= 0:
                return False
            if mid in walls or dest in walls:
                return False
            # Cannot sprint through or onto the catcher.
            if mid == other or dest == other:
                return False
            return True
        # Vault (catcher)
        if state.vault_charges <= 0:
            return False
        if mid not in walls:
            return False  # need a wall to vault over
        if dest in walls:
            return False  # cannot land on a wall
        # Landing on the runner is fine — that is the capture.
        return True

    return False


def legal_action_mask(state: GameState) -> np.ndarray:
    """Boolean mask of shape (17,). All False if state is terminal."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    if state.terminated:
        return mask
    for a in range(ACTION_SPACE_SIZE):
        mask[a] = _is_legal(state, a)
    return mask


# --- Transition ----------------------------------------------------------


def _apply(state: GameState, action: int) -> tuple[GameState, float, float]:
    """Compute the post-action state. Caller must have verified legality."""
    actor = state.current_agent
    rx, ry = state.runner_pos
    cx, cy = state.catcher_pos
    runner_walls = state.runner_walls
    catcher_walls = state.catcher_walls
    sprint = state.sprint_charges
    vault = state.vault_charges

    if action != ACTION_WAIT:
        dx, dy = direction_of(action)
        if action in MOVE_ACTIONS:
            if actor == "runner":
                rx, ry = rx + dx, ry + dy
            else:
                cx, cy = cx + dx, cy + dy
        elif action in PLACE_WALL_ACTIONS:
            target = (
                (rx + dx, ry + dy) if actor == "runner" else (cx + dx, cy + dy)
            )
            if actor == "runner":
                runner_walls = runner_walls | {target}
            else:
                catcher_walls = catcher_walls | {target}
        elif action in REMOVE_WALL_ACTIONS:
            target = (
                (rx + dx, ry + dy) if actor == "runner" else (cx + dx, cy + dy)
            )
            if actor == "runner":
                runner_walls = runner_walls - {target}
            else:
                catcher_walls = catcher_walls - {target}
        elif action in SPECIAL_ACTIONS:
            if actor == "runner":  # Sprint
                rx, ry = rx + 2 * dx, ry + 2 * dy
                sprint -= 1
            else:  # Vault
                cx, cy = cx + 2 * dx, cy + 2 * dy
                vault -= 1

    new_turn = state.turn + 1
    next_agent: Agent = "catcher" if actor == "runner" else "runner"

    captured = (rx, ry) == (cx, cy)
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
        runner_walls=runner_walls,
        catcher_walls=catcher_walls,
        sprint_charges=sprint,
        vault_charges=vault,
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
    truncated, info). `truncated` is always False here — the 30-turn limit
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
    """Return a (7, 9, 9) float32 tensor. Channels:

    0: own position (one-hot)
    1: opponent position (one-hot)
    2: own walls
    3: opponent walls
    4: own charges remaining (normalized scalar broadcast)
    5: opponent charges remaining (normalized scalar broadcast)
    6: turn number (normalized scalar broadcast)

    `perspective` swaps own / opponent so a single network can play either
    role. Coordinates: array is indexed [channel, y, x].
    """
    if perspective not in ("runner", "catcher"):
        raise ValueError(f"perspective must be 'runner' or 'catcher', got {perspective!r}")

    obs = np.zeros((OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    if perspective == "runner":
        own_pos, opp_pos = state.runner_pos, state.catcher_pos
        own_walls, opp_walls = state.runner_walls, state.catcher_walls
        own_charges, own_max = state.sprint_charges, SPRINT_CHARGES
        opp_charges, opp_max = state.vault_charges, VAULT_CHARGES
    else:
        own_pos, opp_pos = state.catcher_pos, state.runner_pos
        own_walls, opp_walls = state.catcher_walls, state.runner_walls
        own_charges, own_max = state.vault_charges, VAULT_CHARGES
        opp_charges, opp_max = state.sprint_charges, SPRINT_CHARGES

    obs[0, own_pos[1], own_pos[0]] = 1.0
    obs[1, opp_pos[1], opp_pos[0]] = 1.0
    for x, y in own_walls:
        obs[2, y, x] = 1.0
    for x, y in opp_walls:
        obs[3, y, x] = 1.0
    obs[4, :, :] = own_charges / max(own_max, 1)
    obs[5, :, :] = opp_charges / max(opp_max, 1)
    obs[6, :, :] = state.turn / TURN_LIMIT

    return obs


# --- Convenience helpers -------------------------------------------------


def with_overrides(state: GameState, **overrides) -> GameState:
    """Return a copy of `state` with the given fields replaced.

    Intended for tests and scripted scenarios — not for use inside `step`.
    """
    return replace(state, **overrides)
