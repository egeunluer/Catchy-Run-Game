"""Heuristic-vs-heuristic balance harness.

Runs N games between the greedy baselines and prints the runner win rate
with a normal-approximation 95% confidence interval. Used to verify that
the kit / dials produce approximate parity (target band 45-55%) before
handing off to DRL training.

Run:

    python -m catchy_run_game.balance --games 200 --seed 0
"""

from __future__ import annotations

import argparse
import math
import random

from . import engine
from .agents.heuristic import catcher_policy, runner_policy


def play_one(rng: random.Random) -> str:
    state = engine.reset(seed=rng.randrange(2**31))
    while not state.terminated:
        if state.current_agent == "runner":
            action = runner_policy(state, rng)
        else:
            action = catcher_policy(state, rng)
        state, *_ = engine.step(state, action)
    assert state.winner is not None
    return state.winner


def run(games: int, seed: int) -> tuple[int, int, int]:
    """Return (runner_wins, catcher_wins, total)."""
    rng = random.Random(seed)
    runner_wins = 0
    catcher_wins = 0
    for _ in range(games):
        winner = play_one(rng)
        if winner == "runner":
            runner_wins += 1
        else:
            catcher_wins += 1
    return runner_wins, catcher_wins, games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runner_wins, catcher_wins, n = run(args.games, args.seed)
    p = runner_wins / n
    # Normal-approximation 95% CI for a binomial proportion.
    half_width = 1.96 * math.sqrt(p * (1 - p) / n)
    lo = max(0.0, p - half_width)
    hi = min(1.0, p + half_width)

    print(f"Games:           {n}")
    print(f"Runner wins:     {runner_wins} ({p * 100:.1f}%)")
    print(f"Catcher wins:    {catcher_wins} ({(1 - p) * 100:.1f}%)")
    print(f"Runner win rate: {p * 100:.1f}% (95% CI: {lo * 100:.1f}%-{hi * 100:.1f}%)")

    target_lo, target_hi = 0.45, 0.55
    if target_lo <= p <= target_hi:
        print("RESULT: within target band [45%, 55%].")
    else:
        side = "runner" if p > target_hi else "catcher"
        print(f"RESULT: outside target band — {side} is favored. Tune dials.")


if __name__ == "__main__":
    main()
