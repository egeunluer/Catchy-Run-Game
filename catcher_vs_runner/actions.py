"""Frozen action encoding for Catcher vs. Runner.

The discrete action space has size 17. Indices 0-15 are grouped into four
direction-indexed action types of 4 each (N, E, S, W). Index 16 is wait.
Indices 12-15 are role-dependent: sprint for the runner, vault for the catcher.
"""

from __future__ import annotations

ACTION_SPACE_SIZE = 17

ACTION_MOVE_N = 0
ACTION_MOVE_E = 1
ACTION_MOVE_S = 2
ACTION_MOVE_W = 3

ACTION_PLACE_WALL_N = 4
ACTION_PLACE_WALL_E = 5
ACTION_PLACE_WALL_S = 6
ACTION_PLACE_WALL_W = 7

ACTION_REMOVE_WALL_N = 8
ACTION_REMOVE_WALL_E = 9
ACTION_REMOVE_WALL_S = 10
ACTION_REMOVE_WALL_W = 11

ACTION_SPECIAL_N = 12
ACTION_SPECIAL_E = 13
ACTION_SPECIAL_S = 14
ACTION_SPECIAL_W = 15

ACTION_WAIT = 16

ACTION_NAMES: list[str] = [
    "MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W",
    "PLACE_WALL_N", "PLACE_WALL_E", "PLACE_WALL_S", "PLACE_WALL_W",
    "REMOVE_WALL_N", "REMOVE_WALL_E", "REMOVE_WALL_S", "REMOVE_WALL_W",
    "SPECIAL_N", "SPECIAL_E", "SPECIAL_S", "SPECIAL_W",
    "WAIT",
]

# Direction vectors as (dx, dy). y grows downward (south).
DIR_N = (0, -1)
DIR_E = (1, 0)
DIR_S = (0, 1)
DIR_W = (-1, 0)

DIRECTIONS: tuple[tuple[int, int], ...] = (DIR_N, DIR_E, DIR_S, DIR_W)
DIRECTION_NAMES: tuple[str, ...] = ("N", "E", "S", "W")

MOVE_ACTIONS = (ACTION_MOVE_N, ACTION_MOVE_E, ACTION_MOVE_S, ACTION_MOVE_W)
PLACE_WALL_ACTIONS = (
    ACTION_PLACE_WALL_N, ACTION_PLACE_WALL_E,
    ACTION_PLACE_WALL_S, ACTION_PLACE_WALL_W,
)
REMOVE_WALL_ACTIONS = (
    ACTION_REMOVE_WALL_N, ACTION_REMOVE_WALL_E,
    ACTION_REMOVE_WALL_S, ACTION_REMOVE_WALL_W,
)
SPECIAL_ACTIONS = (
    ACTION_SPECIAL_N, ACTION_SPECIAL_E,
    ACTION_SPECIAL_S, ACTION_SPECIAL_W,
)


def direction_of(action: int) -> tuple[int, int]:
    """Return the (dx, dy) direction vector for any direction-indexed action.

    Raises ValueError for ACTION_WAIT.
    """
    if action == ACTION_WAIT:
        raise ValueError("ACTION_WAIT has no direction")
    if not 0 <= action < 16:
        raise ValueError(f"action {action} out of range")
    return DIRECTIONS[action % 4]
