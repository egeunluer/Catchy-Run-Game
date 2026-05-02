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

from ..actions import ACTION_SPACE_SIZE
from ..engine import Agent, GameState, legal_action_mask, step


def _chebyshev(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _score(state: GameState, action: int, role: Agent) -> float:
    """Score `action` from `role`'s perspective. Higher is better."""
    new_state, _, _, terminated, _, _ = step(state, action)
    if terminated:
        if new_state.winner == role:
            return math.inf
        return -math.inf
    distance = _chebyshev(new_state.runner_pos, new_state.catcher_pos)
    base = distance if role == "runner" else -distance
    if role == "runner" and new_state.projectiles:
        nearest = min(
            _chebyshev(new_state.runner_pos, pos) for (pos, _) in new_state.projectiles
        )
        # Prefer cells farther from any bullet. Weight kept small so the
        # runner still primarily flees the catcher.
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
