"""Tests for the defensive shooting catcher opponent.

The opponent must fire exactly when the CatcherRewardShaper's special-defense
bonus would pay out, and otherwise fall back to a chasing move. These tests
pin the trigger to a controlled board and cross-check it against the reward
shaper so the duplicated scan in opponents.py cannot silently drift.
"""

from __future__ import annotations

import random

from catchy_run_game import engine
from catchy_run_game.actions import ACTION_SPECIAL_S, DIRECTIONS_8
from rl_agent.opponents import (
    defensive_shooter_opponent,
    _pick_defensive_shot,
)
from rl_agent.reward_shaping import CatcherRewardShaper


def _catcher_state(**overrides):
    """A catcher-turn state with a single, controllable special square."""
    base = engine.reset(seed=0)
    fields = dict(
        catcher_pos=(3, 0),
        runner_pos=(3, 3),
        special_squares=frozenset({(3, 2)}),
        captured_squares=frozenset(),
        projectiles=frozenset(),
        current_agent="catcher",
        sprint_charges=3,
    )
    fields.update(overrides)
    return engine.with_overrides(base, **fields)


def test_shoots_south_to_defend_reachable_special():
    # Special at (3,2) sits on the catcher's south ray (k=2); runner at (3,3)
    # is one step away, so the special is reachable in one turn -> qualifies.
    state = _catcher_state()
    idx = _pick_defensive_shot(state, random.Random(0))
    assert idx is not None
    assert DIRECTIONS_8[idx] == (0, 1)  # South

    action = defensive_shooter_opponent(state, random.Random(0))
    assert action == ACTION_SPECIAL_S


def test_shot_matches_reward_special_defense_bonus():
    # The chosen shot, when actually fired, must score the special-defense bonus.
    state = _catcher_state()
    action = defensive_shooter_opponent(state, random.Random(0))
    curr, *_ = engine.step(state, action)

    shaper = CatcherRewardShaper()
    bonus = shaper._special_defense_bonus(state, curr)
    assert bonus == CatcherRewardShaper.SPECIAL_DEFENSE_COEFF


def test_no_shot_when_special_unreachable_falls_back_to_move():
    # Runner parked far from the special: not reachable in one or two turns,
    # so no direction qualifies and the opponent chases with a move action.
    state = _catcher_state(runner_pos=(0, 6), sprint_charges=0)
    assert _pick_defensive_shot(state, random.Random(0)) is None

    action = defensive_shooter_opponent(state, random.Random(0))
    assert 0 <= action < 8  # a MOVE_* action, not a shot


def test_no_shot_when_special_already_captured():
    state = _catcher_state(captured_squares=frozenset({(3, 2)}))
    assert _pick_defensive_shot(state, random.Random(0)) is None
