import random

from catchy_run_game.agents import heuristic
from catchy_run_game import engine
from catchy_run_game.actions import (
    DIRECTIONS_8,
    CARDINAL_DIRS,
    SPECIAL_ACTIONS,
)
from catchy_run_game.engine import BOARD_SIZE


def heuristic_opponent(state):
    if state.current_agent == "runner":
        return heuristic.runner_policy(state)
    else:
        return heuristic.catcher_policy(state)


# --- Defensive shooting catcher --------------------------------------------
#
# A catcher that fires precisely when the CatcherRewardShaper's special-defense
# bonus would pay out, and otherwise moves like the greedy heuristic catcher.
# Its purpose is to give the runner a training opponent whose bullets are
# *meaningful* — defending an uncaptured special the runner is poised to reach —
# so the runner learns to dodge purposeful shots rather than random spam.
#
# The reach predicates and the near/far ray scan below are a deliberate mirror
# of CatcherRewardShaper._special_defense_bonus / _runner_reaches_in_one /
# _runner_reaches_in_exactly_two in rl_agent/reward_shaping.py. They are
# duplicated here (rather than imported) on purpose; if that shoot condition is
# ever retuned, update both places so the opponent keeps shooting exactly under
# the conditions the reward rewards.

# Mirrors CatcherRewardShaper.SPECIAL_DEFENSE_NEAR_MAX / _FAR_MIN / _FAR_MAX.
_DEFENSE_NEAR_MAX = 3
_DEFENSE_FAR_MIN = 4
_DEFENSE_FAR_MAX = 5


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _runner_reaches_in_one(state, target):
    """True iff the runner can land on `target` in a single move or sprint.

    Mirror of CatcherRewardShaper._runner_reaches_in_one.
    """
    runner = state.runner_pos
    catcher = state.catcher_pos
    if _cheb(runner, target) <= 1:
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


def _runner_reaches_in_exactly_two(state, target):
    """True iff the runner needs exactly two turns to land on `target`.

    Mirror of CatcherRewardShaper._runner_reaches_in_exactly_two.
    """
    if _runner_reaches_in_one(state, target):
        return False
    runner = state.runner_pos
    catcher = state.catcher_pos
    specials_open = state.special_squares - state.captured_squares

    candidates = []
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
        if _cheb(mid, target) <= 1:
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


def _defended_special(state, direction):
    """If shooting `direction` would satisfy the special-defense condition,
    return the threatened special cell; otherwise None.

    Mirror of the positive branch of CatcherRewardShaper._special_defense_bonus:
    scan the bullet's ray from the catcher; a near cell (k=1..3) qualifies if its
    special is reachable in one runner turn, a far cell (k=4..5) if reachable in
    exactly two. The scan breaks when the ray leaves the board.
    """
    dx, dy = direction
    cx, cy = state.catcher_pos
    remaining = state.special_squares - state.captured_squares
    if not remaining:
        return None

    for k in range(1, _DEFENSE_NEAR_MAX + 1):
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


def _pick_defensive_shot(state, rng):
    """Return the index into DIRECTIONS_8 of the best defensive shot, or None.

    Shooting takes priority whenever any direction qualifies. Among qualifying
    directions, prefer the one whose defended special is nearest the runner
    (most urgent), tie-broken by clockwise order, with rng resolving exact ties.
    """
    candidates = []  # (runner_distance, clockwise_index)
    for idx, direction in enumerate(DIRECTIONS_8):
        cell = _defended_special(state, direction)
        if cell is not None:
            candidates.append((_cheb(state.runner_pos, cell), idx))
    if not candidates:
        return None
    best_key = min((dist, idx) for dist, idx in candidates)
    best_dist = best_key[0]
    best_idxs = [idx for dist, idx in candidates if dist == best_dist]
    return rng.choice(best_idxs)


def defensive_shooter_opponent(state, rng=None):
    """Heuristic catcher that shoots under the reward's special-defense
    conditions and otherwise chases like heuristic.catcher_policy."""
    rng = rng or random.Random()
    if state.current_agent == "runner":
        return heuristic.runner_policy(state, rng)
    direction_idx = _pick_defensive_shot(state, rng)
    if direction_idx is not None:
        return SPECIAL_ACTIONS[direction_idx]
    return heuristic.catcher_policy(state, rng)


def make_rl_opponent(model_path: str, role: engine.Agent, deterministic: bool = False):
    """Wrap a trained checkpoint as a (state) -> int opponent policy.

    The model is loaded once and captured in the closure, so every opponent in a
    pool keeps its own instance. `deterministic` defaults to False for training
    opponents on purpose: a stochastic opponent shows varied behavior across
    episodes, so the trainee learns a general response (e.g. respecting bullets
    from anywhere) instead of a narrow counter to one frozen trajectory.
    """
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(model_path)

    def policy(state):
        obs = engine.encode_observation(state, role)
        mask = engine.legal_action_mask(state)
        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        return int(action)

    return policy