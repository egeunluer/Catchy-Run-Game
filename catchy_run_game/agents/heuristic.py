"""Greedy baseline policies for both roles.

These are used by:
* the GUI to offer a "play vs. AI" mode, and
* the balance harness, to estimate win-rate parity between the two kits.

The policy is intentionally shallow: enumerate legal actions, score each by
Chebyshev (king-move) distance to the opponent after applying the action,
with terminal states overriding the score. The runner additionally subtracts
a danger term equal to its proximity to the nearest projectile, so it
prefers staying away from bullets.

This is too weak to find subtle imbalances, but it stress-tests the engine
and gives a baseline non-trivial opponent.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from ..actions import ACTION_SPACE_SIZE, DIRECTIONS_8, CARDINAL_DIRS, SPECIAL_ACTIONS
from ..engine import (
    Agent,
    BOARD_SIZE,
    GameState,
    legal_action_mask,
    step,
)


def _chebyshev(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _manhattan(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _score(state: GameState, action: int, role: Agent) -> float:
    """Score `action` from `role`'s perspective. Higher is better."""
    new_state, _, _, terminated, _, _ = step(state, action)
    if terminated:
        if new_state.winner == role:
            return math.inf
        return -math.inf
    distance = _chebyshev(new_state.runner_pos, new_state.catcher_pos)
    base = distance if role == "runner" else -distance
    if role == "runner":
        uncaptured = new_state.special_squares - new_state.captured_squares
        if uncaptured:
            nearest_sq = min(_chebyshev(new_state.runner_pos, sq) for sq in uncaptured)
            base += 1.5 * (BOARD_SIZE - nearest_sq)
        if new_state.projectiles:
            nearest = min(
                _chebyshev(new_state.runner_pos, pos) for (pos, _) in new_state.projectiles
            )
            base += 0.5 * nearest
    return base


def _choose(state: GameState, role: Agent, rng: random.Random) -> int:
    mask = legal_action_mask(state)
    legal = [a for a in range(ACTION_SPACE_SIZE) if mask[a]]
    if not legal:
        raise RuntimeError(f"No legal actions for {role} at turn {state.turn}.")
    scores = [_score(state, a, role) for a in legal]
    best = max(scores)
    best_actions = [a for a, s in zip(legal, scores) if s == best]
    if role == "catcher" and len(best_actions) > 1:
        candidates = [(a, step(state, a)[0].catcher_pos) for a in best_actions]
        if all(_chebyshev(pos, state.runner_pos) == 1 for _, pos in candidates):
            min_man = min(_manhattan(pos, state.runner_pos) for _, pos in candidates)
            best_actions = [a for a, pos in candidates if _manhattan(pos, state.runner_pos) == min_man]
    return rng.choice(best_actions)


def runner_policy(state: GameState, rng: Optional[random.Random] = None) -> int:
    if state.current_agent != "runner":
        raise ValueError("runner_policy called on a non-runner turn.")
    return _choose(state, "runner", rng or random.Random())


def catcher_policy(state: GameState, rng: Optional[random.Random] = None) -> int:
    if state.current_agent != "catcher":
        raise ValueError("catcher_policy called on a non-catcher turn.")
    return _choose(state, "catcher", rng or random.Random())


def policy_for(role: Agent):
    """Return the heuristic policy callable for the given role."""
    if role == "runner":
        return runner_policy
    if role == "catcher":
        return catcher_policy
    raise ValueError(f"Unknown role: {role!r}")


# --- Shooting catcher --------------------------------------------------------
# Fires when a special square on the shoot ray is imminently reachable by the
# runner; otherwise falls back to catcher_policy. The reach predicates use
# Chebyshev distance to match the engine's 8-directional movement.

_DEFENSE_NEAR_MAX = 3   # inclusive max ray distance for "runner reaches in 1"
_DEFENSE_FAR_MIN  = 4   # inclusive min ray distance for "runner reaches in 2"
_DEFENSE_FAR_MAX  = 5   # inclusive max ray distance for "runner reaches in 2"


def _runner_reaches_in_one(state: GameState, target: tuple[int, int]) -> bool:
    """True iff the runner can land on `target` in a single move or sprint."""
    runner  = state.runner_pos
    catcher = state.catcher_pos
    if _chebyshev(runner, target) <= 1:
        return True
    if state.sprint_charges <= 0:
        return False
    dx = target[0] - runner[0]
    dy = target[1] - runner[1]
    if not ((dx == 0 and abs(dy) == 3) or (dy == 0 and abs(dx) == 3)):
        return False
    step_dx = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_dy = 0 if dy == 0 else (1 if dy > 0 else -1)
    one_ahead = (runner[0] + step_dx, runner[1] + step_dy)
    if one_ahead == catcher or target == catcher:
        return False
    return True


def _runner_reaches_in_exactly_two(state: GameState, target: tuple[int, int]) -> bool:
    """True iff the runner needs exactly two turns to land on `target`."""
    if _runner_reaches_in_one(state, target):
        return False
    runner  = state.runner_pos
    catcher = state.catcher_pos
    specials_open = state.special_squares - state.captured_squares

    candidates: list[tuple[tuple[int, int], int]] = []
    for sdx, sdy in DIRECTIONS_8:
        mid = (runner[0] + sdx, runner[1] + sdy)
        if not (0 <= mid[0] < BOARD_SIZE and 0 <= mid[1] < BOARD_SIZE):
            continue
        charges_after = state.sprint_charges + (1 if mid in specials_open else 0)
        candidates.append((mid, charges_after))
    if state.sprint_charges > 0:
        for sdx, sdy in CARDINAL_DIRS:
            mid = (runner[0] + 3 * sdx, runner[1] + 3 * sdy)
            one_ahead = (runner[0] + sdx, runner[1] + sdy)
            if not (0 <= mid[0] < BOARD_SIZE and 0 <= mid[1] < BOARD_SIZE):
                continue
            if one_ahead == catcher or mid == catcher:
                continue
            charges_after = state.sprint_charges - 1 + (1 if mid in specials_open else 0)
            candidates.append((mid, charges_after))

    for mid, charges_after in candidates:
        if _chebyshev(mid, target) <= 1:
            return True
        if charges_after <= 0:
            continue
        dxt = target[0] - mid[0]
        dyt = target[1] - mid[1]
        if not ((dxt == 0 and abs(dyt) == 3) or (dyt == 0 and abs(dxt) == 3)):
            continue
        sx = 0 if dxt == 0 else (1 if dxt > 0 else -1)
        sy = 0 if dyt == 0 else (1 if dyt > 0 else -1)
        one_ahead = (mid[0] + sx, mid[1] + sy)
        if one_ahead == catcher or target == catcher:
            continue
        return True
    return False


def _defended_special(state: GameState, direction: tuple[int, int]) -> tuple[int, int] | None:
    """Return the threatened special cell if shooting `direction` qualifies, else None."""
    dx, dy = direction
    cx, cy = state.catcher_pos
    remaining = state.special_squares - state.captured_squares
    if not remaining:
        return None
    for k in range(3, _DEFENSE_NEAR_MAX + 1):
        cell = (cx + k * dx, cy + k * dy)
        if not (0 <= cell[0] < BOARD_SIZE and 0 <= cell[1] < BOARD_SIZE):
            break
        if cell in remaining and _runner_reaches_in_one(state, cell):
            return cell
    for k in range(_DEFENSE_FAR_MIN, _DEFENSE_FAR_MAX + 1):
        cell = (cx + k * dx, cy + k * dy)
        if not (0 <= cell[0] < BOARD_SIZE and 0 <= cell[1] < BOARD_SIZE):
            break
        if cell in remaining and _runner_reaches_in_exactly_two(state, cell):
            return cell
    return None


def _pick_defensive_shot(state: GameState, rng: random.Random) -> int | None:
    """Return the DIRECTIONS_8 index of the best defensive shot, or None."""
    candidates = []
    for idx, direction in enumerate(DIRECTIONS_8):
        cell = _defended_special(state, direction)
        if cell is not None:
            candidates.append((_chebyshev(state.runner_pos, cell), idx))
    if not candidates:
        return None
    best_dist = min(d for d, _ in candidates)
    best_idxs = [idx for d, idx in candidates if d == best_dist]
    return rng.choice(best_idxs)


def shooting_catcher_policy(state: GameState, rng: Optional[random.Random] = None) -> int:
    """Catcher that shoots to defend threatened specials; otherwise chases."""
    if state.current_agent != "catcher":
        raise ValueError("shooting_catcher_policy called on a non-catcher turn.")
    rng = rng or random.Random()
    direction_idx = _pick_defensive_shot(state, rng)
    if direction_idx is not None:
        return SPECIAL_ACTIONS[direction_idx]
    return catcher_policy(state, rng)
