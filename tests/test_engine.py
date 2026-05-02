"""Tests for catcher_vs_runner.engine.

Covers the engine contract: legality, mask consistency, win conditions,
purity / determinism, and the observation shape & perspective behavior.
Includes coverage for 8-direction movement, runner-only 8-direction wall
placement, the catcher's 8-direction shoot mechanic, and the absence of
both the old vault action and any catcher wall placement.
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
    ACTION_PLACE_WALL_E,
    ACTION_PLACE_WALL_N,
    ACTION_PLACE_WALL_NE,
    ACTION_PLACE_WALL_S,
    ACTION_PLACE_WALL_SE,
    ACTION_PLACE_WALL_W,
    ACTION_REMOVE_WALL_E,
    ACTION_REMOVE_WALL_NE,
    ACTION_SPACE_SIZE,
    ACTION_SPECIAL_E,
    ACTION_SPECIAL_N,
    ACTION_SPECIAL_NE,
    ACTION_SPECIAL_NW,
    ACTION_SPECIAL_S,
    ACTION_SPECIAL_SE,
    ACTION_SPECIAL_SW,
    ACTION_SPECIAL_W,
    ACTION_WAIT,
    MOVE_ACTIONS,
    PLACE_WALL_ACTIONS,
    REMOVE_WALL_ACTIONS,
    SPECIAL_ACTIONS,
)
from catcher_vs_runner.engine import (
    BOARD_SIZE,
    CATCHER_START,
    OBS_CHANNELS,
    RUNNER_START,
    RUNNER_WALL_CAP,
    SPRINT_CHARGES,
    TURN_LIMIT,
    WALL_LIFETIME,
    GameState,
    encode_observation,
    legal_action_mask,
    reset,
    step,
    with_overrides,
)


def _walls(*positions: tuple[int, int], expiry: int = 999) -> frozenset:
    """Build a wall frozenset from positions, all with the same expiry.

    Tests that don't care about the lifetime use the default `expiry=999`
    (effectively permanent within a test). Tests that specifically exercise
    expiration pass an explicit expiry tied to `state.turn`.
    """
    return frozenset((p, expiry) for p in positions)


# --- Reset / starting state ---------------------------------------------


def test_reset_starting_state():
    state = reset()
    assert state.runner_pos == RUNNER_START
    assert state.catcher_pos == CATCHER_START
    assert state.walls == frozenset()
    assert state.sprint_charges == SPRINT_CHARGES
    assert state.projectiles == frozenset()
    assert state.current_agent == "runner"
    assert state.turn == 0
    assert not state.terminated
    assert state.winner is None


def test_reset_deterministic_for_same_seed():
    assert reset(seed=42) == reset(seed=42)
    assert reset(seed=1) == reset(seed=2)


def test_action_space_layout():
    assert ACTION_SPACE_SIZE == 33
    assert len(MOVE_ACTIONS) == 8
    assert len(PLACE_WALL_ACTIONS) == 8
    assert len(REMOVE_WALL_ACTIONS) == 8
    assert len(SPECIAL_ACTIONS) == 8


# --- Movement -----------------------------------------------------------


def test_runner_can_move_orthogonally():
    state = reset()
    new_state, *_ = step(state, ACTION_MOVE_E)
    assert new_state.runner_pos == (1, 0)
    assert new_state.current_agent == "catcher"
    assert new_state.turn == 1


def test_runner_can_move_diagonally():
    state = reset()
    new_state, *_ = step(state, ACTION_MOVE_SE)
    assert new_state.runner_pos == (1, 1)


def test_catcher_can_move_diagonally():
    state = with_overrides(reset(), current_agent="catcher")
    new_state, *_ = step(state, ACTION_MOVE_NW)
    assert new_state.catcher_pos == (BOARD_SIZE - 2, BOARD_SIZE - 2)


def test_off_board_move_illegal():
    state = reset()  # runner at (0, 0); N, W, NW, NE all off-board
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_N)
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_W)
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_NW)
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_NE)


def test_diagonal_move_blocked_by_wall_at_destination():
    state = with_overrides(reset(), walls=_walls((1, 1)))
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_SE)


def test_move_into_wall_illegal():
    state = with_overrides(reset(), walls=_walls((1, 0)))
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_E)


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


def test_catcher_cannot_jump_over_wall():
    state = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(3, 3),
        walls=_walls((4, 3)),
        current_agent="catcher",
    )
    mask = legal_action_mask(state)
    assert not mask[ACTION_MOVE_E]


# --- Walls (runner only) ------------------------------------------------


def test_runner_places_wall_orthogonal():
    state = reset()
    state, *_ = step(state, ACTION_PLACE_WALL_E)
    assert (1, 0) in state.walls


def test_runner_places_wall_diagonal():
    state = reset()
    state, *_ = step(state, ACTION_PLACE_WALL_SE)
    assert (1, 1) in state.walls


def test_runner_can_place_wall_in_all_eight_directions():
    state = with_overrides(reset(), runner_pos=(3, 3))
    mask = legal_action_mask(state)
    for action in PLACE_WALL_ACTIONS:
        assert mask[action], f"{action} should be legal at (3, 3) with no walls"


def test_runner_remove_diagonal_wall():
    # Runner at (1, 1), wall to its NE at (2, 0).
    state = with_overrides(reset(), runner_pos=(1, 1), walls=_walls((2, 0)))
    state, *_ = step(state, ACTION_REMOVE_WALL_NE)
    assert (2, 0) not in state.walls


def test_catcher_cannot_place_walls():
    state = with_overrides(reset(), current_agent="catcher")
    mask = legal_action_mask(state)
    for action in PLACE_WALL_ACTIONS:
        assert not mask[action], f"catcher should not be able to place wall {action}"
    with pytest.raises(ValueError):
        step(state, ACTION_PLACE_WALL_N)


def test_catcher_cannot_remove_walls():
    state = with_overrides(
        reset(),
        catcher_pos=(3, 3),
        walls=_walls((3, 2), (4, 3)),
        current_agent="catcher",
    )
    mask = legal_action_mask(state)
    for action in REMOVE_WALL_ACTIONS:
        assert not mask[action]


def test_runner_wall_cap_enforced():
    # Pick `RUNNER_WALL_CAP` positions on row 0 starting at (2, 0).
    positions = tuple((2 + i, 0) for i in range(RUNNER_WALL_CAP))
    walls = _walls(*positions)
    assert len(walls) == RUNNER_WALL_CAP
    state = with_overrides(reset(), walls=walls)
    mask = legal_action_mask(state)
    for action in PLACE_WALL_ACTIONS:
        assert not mask[action]
    with pytest.raises(ValueError):
        step(state, ACTION_PLACE_WALL_E)


def test_cannot_place_wall_on_opponent():
    state = with_overrides(
        reset(), runner_pos=(3, 3), catcher_pos=(4, 3), current_agent="runner"
    )
    with pytest.raises(ValueError):
        step(state, ACTION_PLACE_WALL_E)


def test_cannot_place_wall_off_board():
    state = reset()
    with pytest.raises(ValueError):
        step(state, ACTION_PLACE_WALL_N)


def test_cannot_place_wall_on_existing_wall():
    state = with_overrides(reset(), walls=_walls((1, 0)))
    with pytest.raises(ValueError):
        step(state, ACTION_PLACE_WALL_E)


def test_cannot_remove_nonexistent_wall():
    state = reset()
    with pytest.raises(ValueError):
        step(state, ACTION_REMOVE_WALL_E)


# --- Sprint -------------------------------------------------------------


def test_runner_sprint_two_cells_cardinal():
    state = reset()
    new_state, *_ = step(state, ACTION_SPECIAL_E)
    assert new_state.runner_pos == (2, 0)
    assert new_state.sprint_charges == SPRINT_CHARGES - 1


def test_runner_cannot_sprint_diagonally():
    state = with_overrides(reset(), runner_pos=(3, 3))
    mask = legal_action_mask(state)
    for action in (ACTION_SPECIAL_NE, ACTION_SPECIAL_SE, ACTION_SPECIAL_SW, ACTION_SPECIAL_NW):
        assert not mask[action]
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_NE)


def test_sprint_blocked_by_intermediate_wall():
    state = with_overrides(reset(), walls=_walls((1, 0)))
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_E)


def test_sprint_blocked_by_destination_wall():
    state = with_overrides(reset(), walls=_walls((2, 0)))
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_E)


def test_sprint_off_board():
    state = with_overrides(reset(), runner_pos=(4, 0))
    mask = legal_action_mask(state)
    assert not mask[ACTION_SPECIAL_E]


def test_sprint_requires_charge():
    state = with_overrides(reset(), sprint_charges=0)
    mask = legal_action_mask(state)
    for a in (ACTION_SPECIAL_N, ACTION_SPECIAL_E, ACTION_SPECIAL_S, ACTION_SPECIAL_W):
        assert not mask[a]


def test_sprint_cannot_pass_through_or_land_on_catcher():
    state = with_overrides(
        reset(),
        runner_pos=(3, 3),
        catcher_pos=(4, 3),
    )
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_E)

    state2 = with_overrides(state, catcher_pos=(5, 3))
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
    # Fire NW: bullet spawns at (5, 5), ticks to (4, 4).
    new_state, *_ = step(state, ACTION_SPECIAL_NW)
    positions_and_dirs = set(new_state.projectiles)
    assert positions_and_dirs == {((4, 4), (-1, -1))}


def test_diagonal_projectile_continues_diagonally():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(5, 5),
        current_agent="catcher",
    )
    # Fire NW: bullet at (4, 4) after fire.
    state, *_ = step(state, ACTION_SPECIAL_NW)
    state, *_ = step(state, ACTION_WAIT)  # runner waits → bullet ticks to (3, 3)
    positions = {p for (p, _) in state.projectiles}
    assert positions == {(3, 3)}
    state, *_ = step(state, ACTION_WAIT)  # catcher waits → bullet ticks to (2, 2)
    positions = {p for (p, _) in state.projectiles}
    assert positions == {(2, 2)}


def test_shoot_off_board_only_direction_is_illegal():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(5, 0),  # north and east off-board; NE off-board too
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
    state = with_overrides(reset(), current_agent="catcher")
    state, *_ = step(state, ACTION_SPECIAL_N)
    state, *_ = step(state, ACTION_WAIT)
    state, *_ = step(state, ACTION_SPECIAL_NW)
    state, *_ = step(state, ACTION_WAIT)
    state, *_ = step(state, ACTION_SPECIAL_W)
    assert len(state.projectiles) >= 1


def test_projectile_advances_on_runner_turn():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(5, 5),
        current_agent="catcher",
    )
    state, *_ = step(state, ACTION_SPECIAL_N)
    positions_after_fire = {p for (p, _) in state.projectiles}
    assert positions_after_fire == {(5, 4)}
    state, *_ = step(state, ACTION_WAIT)
    positions_after_runner = {p for (p, _) in state.projectiles}
    assert positions_after_runner == {(5, 3)}


def test_projectile_offboard_removal():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(5, 0),
        current_agent="catcher",
    )
    state, *_ = step(state, ACTION_SPECIAL_W)
    for _ in range(2 * BOARD_SIZE):
        if not state.projectiles:
            break
        state, *_ = step(state, ACTION_WAIT)
    assert state.projectiles == frozenset()
    assert not state.terminated


def test_diagonal_projectile_offboard_removal():
    # Catcher at (3, 3) firing SE: bullet steps (4,4) → (5,5) → off-board.
    state = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(3, 3),
        current_agent="catcher",
    )
    state, *_ = step(state, ACTION_SPECIAL_SE)
    for _ in range(2 * BOARD_SIZE):
        if not state.projectiles:
            break
        state, *_ = step(state, ACTION_WAIT)
    assert state.projectiles == frozenset()
    assert not state.terminated


def test_projectile_and_wall_mutual_destruction():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(5, 5),
        walls=_walls((5, 3)),
        current_agent="catcher",
    )
    state, *_ = step(state, ACTION_SPECIAL_N)  # bullet at (5, 4)
    assert state.projectiles
    assert (5, 3) in state.walls
    state, *_ = step(state, ACTION_WAIT)  # bullet ticks into (5, 3)
    assert state.projectiles == frozenset()
    assert (5, 3) not in state.walls


def test_diagonal_projectile_destroys_wall():
    state = with_overrides(
        reset(),
        runner_pos=(0, 5),
        catcher_pos=(5, 5),
        walls=_walls((3, 3)),
        current_agent="catcher",
    )
    # Fire NW: spawn at (5,5), tick to (4,4).
    state, *_ = step(state, ACTION_SPECIAL_NW)
    assert {p for (p, _) in state.projectiles} == {(4, 4)}
    assert (3, 3) in state.walls
    state, *_ = step(state, ACTION_WAIT)  # bullet ticks to (3, 3) → destroys wall
    assert state.projectiles == frozenset()
    assert (3, 3) not in state.walls


def test_projectile_hits_runner_catcher_wins():
    state = with_overrides(
        reset(),
        runner_pos=(5, 3),
        catcher_pos=(5, 5),
        current_agent="catcher",
    )
    state, *_, terminated, _, _ = step(state, ACTION_SPECIAL_N)
    assert not terminated
    new_state, r_runner, r_catcher, terminated, _, _ = step(state, ACTION_WAIT)
    assert terminated
    assert new_state.winner == "catcher"
    assert r_runner == -1.0 and r_catcher == 1.0


def test_diagonal_projectile_hits_runner():
    state = with_overrides(
        reset(),
        runner_pos=(2, 2),
        catcher_pos=(5, 5),
        current_agent="catcher",
    )
    # Fire NW: bullet ticks (5,5) → (4,4) → (3,3) → (2,2).
    state, *_ = step(state, ACTION_SPECIAL_NW)
    assert not state.terminated
    state, *_ = step(state, ACTION_WAIT)  # → (3, 3)
    assert not state.terminated
    new_state, *_, terminated, _, _ = step(state, ACTION_WAIT)  # → (2, 2) hits runner
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


# --- Wait & turn flow ---------------------------------------------------


def test_wait_always_legal_and_advances_turn():
    state = reset()
    mask = legal_action_mask(state)
    assert mask[ACTION_WAIT]
    new_state, r_runner, r_catcher, _, _, _ = step(state, ACTION_WAIT)
    assert new_state.runner_pos == state.runner_pos
    assert new_state.current_agent == "catcher"
    assert new_state.turn == 1
    assert r_runner == 0.0 and r_catcher == 0.0


def test_turn_alternation():
    state = reset()
    for expected in ["catcher", "runner", "catcher", "runner"]:
        state, *_ = step(state, ACTION_WAIT)
        assert state.current_agent == expected


# --- Win conditions -----------------------------------------------------


def test_runner_wins_on_timeout():
    state = reset()
    for _ in range(TURN_LIMIT - 1):
        state, *_ = step(state, ACTION_WAIT)
        assert not state.terminated
    new_state, r_runner, r_catcher, terminated, _, _ = step(state, ACTION_WAIT)
    assert terminated
    assert new_state.winner == "runner"
    assert r_runner == 1.0 and r_catcher == -1.0
    assert new_state.turn == TURN_LIMIT


def test_cannot_step_terminal_state():
    state = with_overrides(reset(), terminated=True, winner="runner")
    with pytest.raises(ValueError):
        step(state, ACTION_WAIT)


def test_terminal_mask_all_false():
    state = with_overrides(reset(), terminated=True, winner="runner")
    mask = legal_action_mask(state)
    assert not mask.any()


# --- Mask correctness vs step ------------------------------------------


def _representative_states() -> list[GameState]:
    s0 = reset()
    s1 = with_overrides(reset(), runner_pos=(4, 4), catcher_pos=(5, 4))
    s2 = with_overrides(
        reset(),
        runner_pos=(2, 2),
        walls=_walls((2, 1), (3, 2), (1, 2)),
        sprint_charges=1,
    )
    s3 = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(3, 3),
        walls=_walls((4, 3), (3, 4)),
        current_agent="catcher",
    )
    s4 = with_overrides(
        reset(),
        walls=_walls((2, 0), (3, 0), (4, 0), (5, 0)),
    )
    s5 = with_overrides(reset(), sprint_charges=0)
    s6 = with_overrides(
        reset(),
        current_agent="catcher",
        catcher_pos=(5, 0),
    )
    s7 = with_overrides(
        reset(),
        current_agent="catcher",
        projectiles=frozenset({((3, 3), (0, -1))}),
    )
    return [s0, s1, s2, s3, s4, s5, s6, s7]


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
    assert OBS_CHANNELS == 7


def test_observation_invalid_perspective_rejected():
    state = reset()
    with pytest.raises(ValueError):
        encode_observation(state, "wrong")  # type: ignore[arg-type]


def test_observation_perspective_swaps_own_and_opponent():
    state = with_overrides(
        reset(),
        runner_pos=(2, 3),
        catcher_pos=(5, 1),
        walls=_walls((2, 4), (4, 1)),
        sprint_charges=2,
    )
    runner_obs = encode_observation(state, "runner")
    catcher_obs = encode_observation(state, "catcher")

    np.testing.assert_array_equal(runner_obs[0], catcher_obs[1])
    np.testing.assert_array_equal(runner_obs[1], catcher_obs[0])
    np.testing.assert_array_equal(runner_obs[2], catcher_obs[2])  # walls shared
    # Charge channel: runner has sprint, catcher has none.
    assert runner_obs[3, 0, 0] == pytest.approx(2 / SPRINT_CHARGES)
    assert catcher_obs[3, 0, 0] == pytest.approx(0.0)
    np.testing.assert_array_equal(runner_obs[5], catcher_obs[5])  # turn shared


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
    assert obs[6, 4, 3] == 1.0
    assert obs[6, 2, 1] == 1.0
    assert obs[6].sum() == 2.0
