"""Frozen action encoding for Catcher vs. Runner.

The discrete action space has size 33. All direction-indexed action groups
share the same 8-direction layout (clockwise from N): N, NE, E, SE, S, SW,
W, NW.

* Indices 0-7   — `MOVE_*` (both agents).
* Indices 8-15  — `PLACE_WALL_*` (runner only; catcher illegal).
* Indices 16-23 — `REMOVE_WALL_*` (runner only; catcher illegal).
* Indices 24-31 — `SPECIAL_*`. Role-dependent:
    * Runner = sprint (2-cell jump, cardinal-only — diagonal sprint illegal).
    * Catcher = shoot (fires a projectile in any of the 8 directions).
* Index 32 — `WAIT`.
"""

from __future__ import annotations

ACTION_SPACE_SIZE = 33

# 8-direction moves (indices 0-7).
ACTION_MOVE_N = 0
ACTION_MOVE_NE = 1
ACTION_MOVE_E = 2
ACTION_MOVE_SE = 3
ACTION_MOVE_S = 4
ACTION_MOVE_SW = 5
ACTION_MOVE_W = 6
ACTION_MOVE_NW = 7

# 8-direction place-wall (indices 8-15). Runner only.
ACTION_PLACE_WALL_N = 8
ACTION_PLACE_WALL_NE = 9
ACTION_PLACE_WALL_E = 10
ACTION_PLACE_WALL_SE = 11
ACTION_PLACE_WALL_S = 12
ACTION_PLACE_WALL_SW = 13
ACTION_PLACE_WALL_W = 14
ACTION_PLACE_WALL_NW = 15

# 8-direction remove-wall (indices 16-23). Runner only.
ACTION_REMOVE_WALL_N = 16
ACTION_REMOVE_WALL_NE = 17
ACTION_REMOVE_WALL_E = 18
ACTION_REMOVE_WALL_SE = 19
ACTION_REMOVE_WALL_S = 20
ACTION_REMOVE_WALL_SW = 21
ACTION_REMOVE_WALL_W = 22
ACTION_REMOVE_WALL_NW = 23

# 8-direction special (indices 24-31): sprint for runner (cardinal only),
# shoot for catcher (all 8 directions).
ACTION_SPECIAL_N = 24
ACTION_SPECIAL_NE = 25
ACTION_SPECIAL_E = 26
ACTION_SPECIAL_SE = 27
ACTION_SPECIAL_S = 28
ACTION_SPECIAL_SW = 29
ACTION_SPECIAL_W = 30
ACTION_SPECIAL_NW = 31

ACTION_WAIT = 32

ACTION_NAMES: list[str] = [
    "MOVE_N", "MOVE_NE", "MOVE_E", "MOVE_SE",
    "MOVE_S", "MOVE_SW", "MOVE_W", "MOVE_NW",
    "PLACE_WALL_N", "PLACE_WALL_NE", "PLACE_WALL_E", "PLACE_WALL_SE",
    "PLACE_WALL_S", "PLACE_WALL_SW", "PLACE_WALL_W", "PLACE_WALL_NW",
    "REMOVE_WALL_N", "REMOVE_WALL_NE", "REMOVE_WALL_E", "REMOVE_WALL_SE",
    "REMOVE_WALL_S", "REMOVE_WALL_SW", "REMOVE_WALL_W", "REMOVE_WALL_NW",
    "SPECIAL_N", "SPECIAL_NE", "SPECIAL_E", "SPECIAL_SE",
    "SPECIAL_S", "SPECIAL_SW", "SPECIAL_W", "SPECIAL_NW",
    "WAIT",
]

# Direction vectors as (dx, dy). y grows downward (south).
DIR_N = (0, -1)
DIR_NE = (1, -1)
DIR_E = (1, 0)
DIR_SE = (1, 1)
DIR_S = (0, 1)
DIR_SW = (-1, 1)
DIR_W = (-1, 0)
DIR_NW = (-1, -1)

# 8-direction table; ordered clockwise from N.
DIRECTIONS_8: tuple[tuple[int, int], ...] = (
    DIR_N, DIR_NE, DIR_E, DIR_SE, DIR_S, DIR_SW, DIR_W, DIR_NW,
)
DIRECTION_NAMES_8: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# Cardinal-only set used by the engine to gate runner sprint to N/E/S/W.
CARDINAL_DIRS: frozenset[tuple[int, int]] = frozenset({DIR_N, DIR_E, DIR_S, DIR_W})

MOVE_ACTIONS = (
    ACTION_MOVE_N, ACTION_MOVE_NE, ACTION_MOVE_E, ACTION_MOVE_SE,
    ACTION_MOVE_S, ACTION_MOVE_SW, ACTION_MOVE_W, ACTION_MOVE_NW,
)
PLACE_WALL_ACTIONS = (
    ACTION_PLACE_WALL_N, ACTION_PLACE_WALL_NE, ACTION_PLACE_WALL_E, ACTION_PLACE_WALL_SE,
    ACTION_PLACE_WALL_S, ACTION_PLACE_WALL_SW, ACTION_PLACE_WALL_W, ACTION_PLACE_WALL_NW,
)
REMOVE_WALL_ACTIONS = (
    ACTION_REMOVE_WALL_N, ACTION_REMOVE_WALL_NE, ACTION_REMOVE_WALL_E, ACTION_REMOVE_WALL_SE,
    ACTION_REMOVE_WALL_S, ACTION_REMOVE_WALL_SW, ACTION_REMOVE_WALL_W, ACTION_REMOVE_WALL_NW,
)
SPECIAL_ACTIONS = (
    ACTION_SPECIAL_N, ACTION_SPECIAL_NE, ACTION_SPECIAL_E, ACTION_SPECIAL_SE,
    ACTION_SPECIAL_S, ACTION_SPECIAL_SW, ACTION_SPECIAL_W, ACTION_SPECIAL_NW,
)


def direction_of(action: int) -> tuple[int, int]:
    """Return the (dx, dy) direction vector for any direction-indexed action.

    All groups (move, place, remove, special) share the same 8-direction
    layout, so the lookup is a single modulo. Raises ValueError for
    `ACTION_WAIT` or out-of-range actions.
    """
    if 0 <= action < 32:
        return DIRECTIONS_8[action % 8]
    raise ValueError(f"action {action} has no direction")
