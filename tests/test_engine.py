"""Tests for catchy_run_game.engine.

Covers the engine contract: legality, mask consistency, win conditions,
purity / determinism, and the observation shape & perspective behavior.
The action space has been narrowed to 16 (8-direction move + 8-direction
special); wall placement and the wait action are no longer part of the
game and have no test coverage.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from catcher_vs_runner import engine
from catcher_vs_runner.actions import (
    ACTION_MOVE_E,
    ACTION_MOVE_N,
    ACTION_MOVE_NE,
    ACTION_MOVE_NW,
    ACTION_MOVE_S,
    ACTION_MOVE_SE,
    ACTION_MOVE_SW,
    ACTION_MOVE_W,
    ACTION_SPACE_SIZE,
    ACTION_SPECIAL_E,
    ACTION_SPECIAL_N,
    ACTION_SPECIAL_NE,
    ACTION_SPECIAL_NW,
    ACTION_SPECIAL_S,
    ACTION_SPECIAL_SE,
    ACTION_SPECIAL_SW,
    ACTION_SPECIAL_W,
    MOVE_ACTIONS,
    SPECIAL_ACTIONS,
)
from catcher_vs_runner.engine import (
    BOARD_SIZE,
    CATCHER_START,
    OBS_CHANNELS,
    RUNNER_START,
    SPECIAL_MAJORITY,
    SPRINT_CHARGES,
    TURN_LIMIT,
    GameState,
    encode_observation,
    legal_action_mask,
    reset,
    step,
    with_overrides,
)


# --- Test helpers --------------------------------------------------------


def _pace_step(state: GameState) -> GameState:
    """Advance one half-turn with a safe pacing move.

    Runner oscillates between (0, 0) and (1, 0); catcher oscillates between
    (6, 5) and (6, 6). The two paths never approach each other, so this is
    safe to drive a state to timeout. Caller must set up the corner positions.
    """
    if state.current_agent == "runner":
        rx, _ = state.runner_pos
        action = ACTION_MOVE_E if rx == 0 else ACTION_MOVE_W
    else:
        _, cy = state.catcher_pos
        action = ACTION_MOVE_N if cy == 6 else ACTION_MOVE_S
    new_state, *_ = step(state, action)
    return new_state


# --- Reset / starting state ---------------------------------------------


def test_reset_starting_state():
    state = reset()
    assert state.runner_pos == RUNNER_START
    assert state.catcher_pos == CATCHER_START
    assert state.sprint_charges == SPRINT_CHARGES
    assert state.projectiles == frozenset()
    assert state.current_agent == "runner"
    assert state.turn == 0
    assert not state.terminated
    assert state.winner is None


def test_reset_deterministic_for_same_seed():
    assert reset(seed=42) == reset(seed=42)
    assert reset(seed=1) != reset(seed=2)


def test_action_space_layout():
    assert ACTION_SPACE_SIZE == 16
    assert len(MOVE_ACTIONS) == 8
    assert len(SPECIAL_ACTIONS) == 8


# --- Movement -----------------------------------------------------------


def test_runner_can_move_orthogonally():
    state = reset()
    new_state, *_ = step(state, ACTION_MOVE_E)
    assert new_state.runner_pos == (4, 0)
    assert new_state.current_agent == "catcher"
    assert new_state.turn == 1


def test_runner_can_move_diagonally():
    state = reset()
    new_state, *_ = step(state, ACTION_MOVE_SE)
    assert new_state.runner_pos == (4, 1)


def test_catcher_can_move_diagonally():
    state = with_overrides(reset(), current_agent="catcher")
    new_state, *_ = step(state, ACTION_MOVE_NW)
    assert new_state.catcher_pos == (2, BOARD_SIZE - 2)


def test_off_board_move_illegal():
    state = with_overrides(reset(), runner_pos=(0, 0))
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_N)
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_W)
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_NW)
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_NE)


def test_runner_cannot_move_onto_catcher():
    state = with_overrides(
        reset(), runner_pos=(3, 3), catcher_pos=(4, 3), current_agent="runner"
    )
    mask = legal_action_mask(state)
    assert not mask[ACTION_MOVE_E]
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_E)


def test_catcher_capture_by_movement():
    state = with_overrides(
        reset(), runner_pos=(4, 3), catcher_pos=(3, 3), current_agent="catcher"
    )
    new_state, r_runner, r_catcher, terminated, truncated, _ = step(
        state, ACTION_MOVE_E
    )
    assert terminated and not truncated
    assert new_state.winner == "catcher"
    assert r_runner == -1.0 and r_catcher == 1.0
    assert new_state.runner_pos == new_state.catcher_pos


def test_catcher_capture_by_diagonal_move():
    state = with_overrides(
        reset(), runner_pos=(4, 4), catcher_pos=(3, 3), current_agent="catcher"
    )
    new_state, *_, terminated, _, _ = step(state, ACTION_MOVE_SE)
    assert terminated
    assert new_state.winner == "catcher"


# --- Sprint -------------------------------------------------------------


def test_runner_sprint_three_cells_cardinal():
    state = with_overrides(reset(), special_squares=frozenset(), captured_squares=frozenset())
    new_state, *_ = step(state, ACTION_SPECIAL_E)
    assert new_state.runner_pos == (6, 0)
    assert new_state.sprint_charges == SPRINT_CHARGES - 1


def test_runner_cannot_sprint_diagonally():
    state = with_overrides(reset(), runner_pos=(3, 3))
    mask = legal_action_mask(state)
    for action in (ACTION_SPECIAL_NE, ACTION_SPECIAL_SE, ACTION_SPECIAL_SW, ACTION_SPECIAL_NW):
        assert not mask[action]
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_NE)


def test_sprint_off_board():
    state = with_overrides(reset(), runner_pos=(4, 0))
    mask = legal_action_mask(state)
    assert not mask[ACTION_SPECIAL_E]


def test_sprint_requires_charge():
    state = with_overrides(reset(), sprint_charges=0)
    mask = legal_action_mask(state)
    for a in (ACTION_SPECIAL_N, ACTION_SPECIAL_E, ACTION_SPECIAL_S, ACTION_SPECIAL_W):
        assert not mask[a]


def test_sprint_blocked_when_catcher_at_mid_or_dest():
    # Catcher at the mid cell (1 ahead).
    state = with_overrides(
        reset(),
        runner_pos=(3, 3),
        catcher_pos=(4, 3),
    )
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_E)

    # Catcher at the dest cell (3 ahead).
    state2 = with_overrides(state, catcher_pos=(6, 3))
    with pytest.raises(ValueError):
        step(state2, ACTION_SPECIAL_E)


# --- Shoot / projectiles ------------------------------------------------


def test_shoot_creates_projectile_cardinal():
    state = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(5, 5),
        current_agent="catcher",
    )
    new_state, *_ = step(state, ACTION_SPECIAL_N)
    assert new_state.catcher_pos == (5, 5)
    positions = {p for (p, _) in new_state.projectiles}
    assert positions == {(5, 4)}


def test_shoot_creates_projectile_diagonal():
    state = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(5, 5),
        current_agent="catcher",
    )
    new_state, *_ = step(state, ACTION_SPECIAL_NW)
    positions_and_dirs = set(new_state.projectiles)
    assert positions_and_dirs == {((4, 4), (-1, -1))}


def test_diagonal_projectile_continues_diagonally():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(5, 5),
        special_squares=frozenset(),
        captured_squares=frozenset(),
        current_agent="catcher",
    )
    state, *_ = step(state, ACTION_SPECIAL_NW)  # bullet at (4, 4)
    state, *_ = step(state, ACTION_MOVE_E)      # runner→(1,5), bullet (3, 3)
    assert {p for (p, _) in state.projectiles} == {(3, 3)}
    state, *_ = step(state, ACTION_MOVE_S)      # catcher→(5,6), bullet (2, 2)
    assert {p for (p, _) in state.projectiles} == {(2, 2)}


def test_shoot_off_board_only_direction_is_illegal():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(BOARD_SIZE - 1, 0),  # NE corner of the board
        current_agent="catcher",
    )
    mask = legal_action_mask(state)
    assert not mask[ACTION_SPECIAL_N]
    assert not mask[ACTION_SPECIAL_E]
    assert not mask[ACTION_SPECIAL_NE]
    assert mask[ACTION_SPECIAL_S]
    assert mask[ACTION_SPECIAL_W]
    assert mask[ACTION_SPECIAL_SW]


def test_shoot_unlimited():
    state = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(3, 6),
        special_squares=frozenset(),
        captured_squares=frozenset(),
        current_agent="catcher",
    )
    state, *_ = step(state, ACTION_SPECIAL_N)    # catcher fires
    state, *_ = step(state, ACTION_MOVE_E)       # runner→(1,0)
    state, *_ = step(state, ACTION_SPECIAL_NW)   # catcher fires again
    state, *_ = step(state, ACTION_MOVE_W)       # runner→(0,0)
    state, *_ = step(state, ACTION_SPECIAL_W)    # catcher fires a third time
    assert len(state.projectiles) >= 1
    assert not state.terminated


def test_projectile_advances_on_runner_turn():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(5, 5),
        special_squares=frozenset(),
        captured_squares=frozenset(),
        current_agent="catcher",
    )
    state, *_ = step(state, ACTION_SPECIAL_N)
    assert {p for (p, _) in state.projectiles} == {(5, 4)}
    state, *_ = step(state, ACTION_MOVE_E)
    assert {p for (p, _) in state.projectiles} == {(5, 3)}


def test_projectile_offboard_removal():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(5, 0),
        special_squares=frozenset(),
        captured_squares=frozenset(),
        current_agent="catcher",
    )
    state, *_ = step(state, ACTION_SPECIAL_W)   # bullet (4, 0)
    state, *_ = step(state, ACTION_MOVE_E)      # bullet (3, 0)
    state, *_ = step(state, ACTION_MOVE_S)      # bullet (2, 0)
    state, *_ = step(state, ACTION_MOVE_W)      # bullet (1, 0)
    state, *_ = step(state, ACTION_MOVE_N)      # bullet (0, 0)
    state, *_ = step(state, ACTION_MOVE_E)      # bullet → off-board
    assert state.projectiles == frozenset()
    assert not state.terminated


def test_diagonal_projectile_offboard_removal():
    state = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(3, 3),
        special_squares=frozenset(),
        captured_squares=frozenset(),
        current_agent="catcher",
    )
    state, *_ = step(state, ACTION_SPECIAL_SE)  # bullet (4, 4)
    state, *_ = step(state, ACTION_MOVE_E)      # bullet (5, 5)
    state, *_ = step(state, ACTION_MOVE_W)      # bullet (6, 6)
    state, *_ = step(state, ACTION_MOVE_E)      # bullet → off-board
    assert state.projectiles == frozenset()
    assert not state.terminated


def test_projectile_hits_runner_catcher_wins():
    state = with_overrides(
        reset(),
        runner_pos=(4, 3),
        catcher_pos=(5, 5),
        special_squares=frozenset(),
        captured_squares=frozenset(),
        current_agent="catcher",
    )
    # Catcher fires N: bullet spawns at (5,5), ticks to (5,4). Not terminated.
    state, *_, terminated, _, _ = step(state, ACTION_SPECIAL_N)
    assert not terminated
    # Runner moves E onto (5,3) just as the bullet ticks from (5,4) to (5,3).
    new_state, r_runner, r_catcher, terminated, _, _ = step(state, ACTION_MOVE_E)
    assert terminated
    assert new_state.winner == "catcher"
    assert r_runner == -1.0 and r_catcher == 1.0


def test_diagonal_projectile_hits_runner():
    state = with_overrides(
        reset(),
        runner_pos=(2, 1),
        catcher_pos=(5, 5),
        special_squares=frozenset(),
        captured_squares=frozenset(),
        current_agent="catcher",
    )
    # Catcher fires NW: bullet at (4, 4) after tick.
    state, *_ = step(state, ACTION_SPECIAL_NW)
    assert {p for (p, _) in state.projectiles} == {(4, 4)}
    # Runner moves S to (2, 2); bullet ticks to (3, 3). Not yet hit.
    state, *_ = step(state, ACTION_MOVE_S)
    assert not state.terminated
    # Catcher moves W (safe); bullet ticks to (2, 2). Runner caught.
    new_state, *_, terminated, _, _ = step(state, ACTION_MOVE_W)
    assert terminated
    assert new_state.winner == "catcher"


def test_projectile_hits_runner_immediately_on_fire():
    state = with_overrides(
        reset(),
        runner_pos=(5, 4),
        catcher_pos=(5, 5),
        current_agent="catcher",
    )
    new_state, *_, terminated, _, _ = step(state, ACTION_SPECIAL_N)
    assert terminated
    assert new_state.winner == "catcher"


# --- Turn flow ----------------------------------------------------------


def test_turn_alternation():
    state = reset()
    moves = [ACTION_MOVE_E, ACTION_MOVE_W, ACTION_MOVE_W, ACTION_MOVE_E]
    for action, expected in zip(moves, ["catcher", "runner", "catcher", "runner"]):
        state, *_ = step(state, action)
        assert state.current_agent == expected


# --- Win conditions -----------------------------------------------------


def test_catcher_wins_on_timeout_without_majority():
    state = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(6, 6),
        special_squares=frozenset(),
        captured_squares=frozenset(),
    )
    while not state.terminated:
        state = _pace_step(state)
    assert state.turn == TURN_LIMIT
    assert state.winner == "catcher"


def test_runner_wins_on_timeout_with_majority():
    captured = frozenset({(3, 3), (4, 4), (3, 4), (4, 3)})
    assert len(captured) >= SPECIAL_MAJORITY
    state = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(6, 6),
        special_squares=captured,
        captured_squares=captured,
    )
    while not state.terminated:
        state = _pace_step(state)
    assert state.turn == TURN_LIMIT
    assert state.winner == "runner"


def test_cannot_step_terminal_state():
    state = with_overrides(reset(), terminated=True, winner="runner")
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_E)


def test_terminal_mask_all_false():
    state = with_overrides(reset(), terminated=True, winner="runner")
    mask = legal_action_mask(state)
    assert not mask.any()


# --- Mask correctness vs step ------------------------------------------


def _representative_states() -> list[GameState]:
    s0 = reset()
    s1 = with_overrides(reset(), runner_pos=(4, 4), catcher_pos=(5, 4))
    s2 = with_overrides(reset(), runner_pos=(2, 2), sprint_charges=1)
    s3 = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(3, 3),
        current_agent="catcher",
    )
    s4 = with_overrides(reset(), sprint_charges=0)
    s5 = with_overrides(
        reset(),
        current_agent="catcher",
        catcher_pos=(5, 0),
    )
    s6 = with_overrides(
        reset(),
        current_agent="catcher",
        projectiles=frozenset({((3, 3), (0, -1))}),
    )
    return [s0, s1, s2, s3, s4, s5, s6]


@pytest.mark.parametrize("state", _representative_states())
def test_mask_matches_step_outcome(state: GameState):
    mask = legal_action_mask(state)
    for a in range(ACTION_SPACE_SIZE):
        if mask[a]:
            step(state, a)
        else:
            with pytest.raises(ValueError):
                step(state, a)


# --- Purity & determinism ----------------------------------------------


def test_step_does_not_mutate_input():
    state = reset()
    snapshot = with_overrides(state)
    step(state, ACTION_MOVE_E)
    assert state == snapshot


def test_step_is_deterministic():
    state = reset()
    a, *_ = step(state, ACTION_MOVE_E)
    b, *_ = step(state, ACTION_MOVE_E)
    assert a == b


def test_state_is_picklable():
    state = reset()
    state, *_ = step(state, ACTION_MOVE_E)
    state, *_ = step(state, ACTION_MOVE_W)
    restored = pickle.loads(pickle.dumps(state))
    assert restored == state


# --- Observation contract ----------------------------------------------


def test_observation_shape_and_dtype():
    state = reset()
    obs = encode_observation(state, "runner")
    assert obs.shape == (OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert obs.dtype == np.float32
    assert OBS_CHANNELS == 9


def test_observation_invalid_perspective_rejected():
    state = reset()
    with pytest.raises(ValueError):
        encode_observation(state, "wrong")  # type: ignore[arg-type]


def test_observation_perspective_swaps_own_and_opponent():
    state = with_overrides(
        reset(),
        runner_pos=(2, 3),
        catcher_pos=(5, 1),
        sprint_charges=2,
    )
    runner_obs = encode_observation(state, "runner")
    catcher_obs = encode_observation(state, "catcher")

    np.testing.assert_array_equal(runner_obs[0], catcher_obs[1])
    np.testing.assert_array_equal(runner_obs[1], catcher_obs[0])
    # Chebyshev distance is symmetric across perspectives.
    np.testing.assert_array_equal(runner_obs[2], catcher_obs[2])
    assert runner_obs[2, 0, 0] == pytest.approx(3 / (BOARD_SIZE - 1))
    # Own-charges channel: runner has sprint, catcher has none.
    assert runner_obs[3, 0, 0] == pytest.approx(2 / SPRINT_CHARGES)
    assert catcher_obs[3, 0, 0] == pytest.approx(0.0)
    np.testing.assert_array_equal(runner_obs[4], catcher_obs[4])  # turn shared


def test_observation_position_channels_are_one_hot():
    state = reset()
    obs = encode_observation(state, "runner")
    assert obs[0].sum() == 1.0
    assert obs[1].sum() == 1.0
    rx, ry = state.runner_pos
    assert obs[0, ry, rx] == 1.0
    cx, cy = state.catcher_pos
    assert obs[1, cy, cx] == 1.0


def test_observation_projectile_channel():
    state = with_overrides(
        reset(),
        projectiles=frozenset({((3, 4), (0, -1)), ((1, 2), (1, 1))}),
    )
    obs = encode_observation(state, "runner")
    # Presence mask at each bullet's cell.
    assert obs[5, 4, 3] == 1.0
    assert obs[5, 2, 1] == 1.0
    assert obs[5].sum() == 2.0
    # Signed direction channels at the same cells.
    assert obs[6, 4, 3] == 0.0
    assert obs[7, 4, 3] == -1.0
    assert obs[6, 2, 1] == 1.0
    assert obs[7, 2, 1] == 1.0
    # Direction channels are zero everywhere else.
    assert (obs[6] != 0).sum() == 1
    assert (obs[7] != 0).sum() == 2
