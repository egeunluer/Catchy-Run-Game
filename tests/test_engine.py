"""Tests for catcher_vs_runner.engine.

Covers the engine contract: legality, mask consistency, win conditions,
purity / determinism, and the observation shape & perspective behavior.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from catcher_vs_runner import engine
from catcher_vs_runner.actions import (
    ACTION_MOVE_E,
    ACTION_MOVE_N,
    ACTION_MOVE_S,
    ACTION_MOVE_W,
    ACTION_PLACE_WALL_E,
    ACTION_PLACE_WALL_N,
    ACTION_PLACE_WALL_S,
    ACTION_PLACE_WALL_W,
    ACTION_REMOVE_WALL_E,
    ACTION_REMOVE_WALL_N,
    ACTION_REMOVE_WALL_W,
    ACTION_SPACE_SIZE,
    ACTION_SPECIAL_E,
    ACTION_SPECIAL_N,
    ACTION_SPECIAL_S,
    ACTION_SPECIAL_W,
    ACTION_WAIT,
)
from catcher_vs_runner.engine import (
    BOARD_SIZE,
    CATCHER_START,
    CATCHER_WALL_CAP,
    OBS_CHANNELS,
    RUNNER_START,
    RUNNER_WALL_CAP,
    SPRINT_CHARGES,
    TURN_LIMIT,
    VAULT_CHARGES,
    GameState,
    encode_observation,
    legal_action_mask,
    reset,
    step,
    with_overrides,
)


# --- Reset / starting state ---------------------------------------------


def test_reset_starting_state():
    state = reset()
    assert state.runner_pos == RUNNER_START
    assert state.catcher_pos == CATCHER_START
    assert state.runner_walls == frozenset()
    assert state.catcher_walls == frozenset()
    assert state.sprint_charges == SPRINT_CHARGES
    assert state.vault_charges == VAULT_CHARGES
    assert state.current_agent == "runner"
    assert state.turn == 0
    assert not state.terminated
    assert state.winner is None


def test_reset_deterministic_for_same_seed():
    assert reset(seed=42) == reset(seed=42)
    # Currently the seed has no effect; this test pins the contract so a
    # future change can decide intentionally whether seeds diverge.
    assert reset(seed=1) == reset(seed=2)


# --- Movement -----------------------------------------------------------


def test_runner_can_move_orthogonally():
    state = reset()
    new_state, *_ = step(state, ACTION_MOVE_E)
    assert new_state.runner_pos == (1, 0)
    assert new_state.current_agent == "catcher"
    assert new_state.turn == 1


def test_off_board_move_illegal():
    state = reset()  # runner at (0, 0); N and W are off-board
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_N)
    with pytest.raises(ValueError):
        step(state, ACTION_MOVE_W)


def test_move_into_wall_illegal():
    state = with_overrides(reset(), runner_walls=frozenset({(1, 0)}))
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


# --- Walls --------------------------------------------------------------


def test_place_and_remove_own_wall():
    state = reset()
    state, *_ = step(state, ACTION_PLACE_WALL_E)  # runner places at (1, 0)
    assert (1, 0) in state.runner_walls
    assert state.current_agent == "catcher"

    # Catcher does something so it becomes runner's turn again.
    state, *_ = step(state, ACTION_MOVE_W)
    assert state.current_agent == "runner"

    state, *_ = step(state, ACTION_REMOVE_WALL_E)
    assert (1, 0) not in state.runner_walls


def test_runner_wall_cap_enforced():
    walls = frozenset({(2, 0), (3, 0), (4, 0), (5, 0)})  # cap = 4
    assert len(walls) == RUNNER_WALL_CAP
    state = with_overrides(reset(), runner_walls=walls)
    mask = legal_action_mask(state)
    # All four place-wall directions illegal because cap reached.
    assert not mask[ACTION_PLACE_WALL_E]
    assert not mask[ACTION_PLACE_WALL_S]
    with pytest.raises(ValueError):
        step(state, ACTION_PLACE_WALL_E)


def test_catcher_wall_cap_smaller():
    walls = frozenset({(7, 8), (8, 7), (6, 8)})  # cap = 3
    assert len(walls) == CATCHER_WALL_CAP
    state = with_overrides(reset(), catcher_walls=walls, current_agent="catcher")
    mask = legal_action_mask(state)
    assert not mask[ACTION_PLACE_WALL_N]
    assert not mask[ACTION_PLACE_WALL_W]


def test_cannot_place_wall_on_opponent():
    state = with_overrides(
        reset(), runner_pos=(3, 3), catcher_pos=(4, 3), current_agent="runner"
    )
    with pytest.raises(ValueError):
        step(state, ACTION_PLACE_WALL_E)


def test_cannot_place_wall_off_board():
    state = reset()  # runner at (0, 0)
    with pytest.raises(ValueError):
        step(state, ACTION_PLACE_WALL_N)


def test_cannot_place_wall_on_existing_wall():
    state = with_overrides(reset(), catcher_walls=frozenset({(1, 0)}))
    with pytest.raises(ValueError):
        step(state, ACTION_PLACE_WALL_E)


def test_cannot_remove_opponents_wall():
    state = with_overrides(reset(), catcher_walls=frozenset({(1, 0)}))
    # Runner is at (0, 0). The wall at (1, 0) is the catcher's.
    with pytest.raises(ValueError):
        step(state, ACTION_REMOVE_WALL_E)


def test_cannot_remove_nonexistent_wall():
    state = reset()
    with pytest.raises(ValueError):
        step(state, ACTION_REMOVE_WALL_E)


# --- Sprint -------------------------------------------------------------


def test_runner_sprint_two_cells():
    state = reset()
    new_state, *_ = step(state, ACTION_SPECIAL_E)
    assert new_state.runner_pos == (2, 0)
    assert new_state.sprint_charges == SPRINT_CHARGES - 1


def test_sprint_blocked_by_intermediate_wall():
    state = with_overrides(reset(), catcher_walls=frozenset({(1, 0)}))
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_E)


def test_sprint_blocked_by_destination_wall():
    state = with_overrides(reset(), catcher_walls=frozenset({(2, 0)}))
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_E)


def test_sprint_off_board():
    state = with_overrides(reset(), runner_pos=(7, 0))  # sprint E goes to (9, 0)
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
        catcher_pos=(4, 3),  # immediately east — sprint E would pass through
    )
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_E)

    state2 = with_overrides(state, catcher_pos=(5, 3))  # land on
    with pytest.raises(ValueError):
        step(state2, ACTION_SPECIAL_E)


# --- Vault --------------------------------------------------------------


def test_catcher_vaults_over_wall():
    state = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(3, 3),
        catcher_walls=frozenset({(4, 3)}),
        current_agent="catcher",
    )
    new_state, *_ = step(state, ACTION_SPECIAL_E)
    assert new_state.catcher_pos == (5, 3)
    assert new_state.vault_charges == VAULT_CHARGES - 1
    assert (4, 3) in new_state.catcher_walls  # the wall remains


def test_vault_requires_adjacent_wall():
    state = with_overrides(reset(), current_agent="catcher")  # no walls
    mask = legal_action_mask(state)
    for a in (ACTION_SPECIAL_N, ACTION_SPECIAL_E, ACTION_SPECIAL_S, ACTION_SPECIAL_W):
        assert not mask[a]


def test_vault_destination_cannot_be_wall():
    state = with_overrides(
        reset(),
        catcher_pos=(3, 3),
        catcher_walls=frozenset({(4, 3), (5, 3)}),  # wall at dest
        current_agent="catcher",
    )
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_E)


def test_vault_onto_runner_captures():
    state = with_overrides(
        reset(),
        runner_pos=(5, 3),
        catcher_pos=(3, 3),
        runner_walls=frozenset({(4, 3)}),  # runner placed it; catcher still vaults
        current_agent="catcher",
    )
    new_state, r_runner, r_catcher, terminated, _, _ = step(state, ACTION_SPECIAL_E)
    assert terminated
    assert new_state.winner == "catcher"
    assert r_catcher == 1.0 and r_runner == -1.0


def test_vault_requires_charge():
    state = with_overrides(
        reset(),
        catcher_pos=(3, 3),
        catcher_walls=frozenset({(4, 3)}),
        vault_charges=0,
        current_agent="catcher",
    )
    mask = legal_action_mask(state)
    assert not mask[ACTION_SPECIAL_E]


def test_vault_off_board():
    state = with_overrides(
        reset(),
        catcher_pos=(BOARD_SIZE - 2, 3),
        catcher_walls=frozenset({(BOARD_SIZE - 1, 3)}),
        current_agent="catcher",
    )
    with pytest.raises(ValueError):
        step(state, ACTION_SPECIAL_E)


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
    # 30 waits brings turn from 0 to 30. Mask any failure modes by waiting.
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
        runner_walls=frozenset({(2, 1), (3, 2), (1, 2)}),
        sprint_charges=1,
    )
    s3 = with_overrides(
        reset(),
        runner_pos=(0, 0),
        catcher_pos=(3, 3),
        catcher_walls=frozenset({(4, 3), (3, 4)}),
        current_agent="catcher",
    )
    s4 = with_overrides(
        reset(),
        runner_walls=frozenset({(2, 0), (3, 0), (4, 0), (5, 0)}),  # cap reached
    )
    s5 = with_overrides(reset(), sprint_charges=0)
    s6 = with_overrides(reset(), current_agent="catcher", vault_charges=0)
    return [s0, s1, s2, s3, s4, s5, s6]


@pytest.mark.parametrize("state", _representative_states())
def test_mask_matches_step_outcome(state: GameState):
    mask = legal_action_mask(state)
    for a in range(ACTION_SPACE_SIZE):
        if mask[a]:
            # Step must succeed.
            step(state, a)
        else:
            with pytest.raises(ValueError):
                step(state, a)


# --- Purity & determinism ----------------------------------------------


def test_step_does_not_mutate_input():
    state = reset()
    snapshot = with_overrides(state)  # copy via dataclass replace; same fields
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


def test_observation_invalid_perspective_rejected():
    state = reset()
    with pytest.raises(ValueError):
        encode_observation(state, "wrong")  # type: ignore[arg-type]


def test_observation_perspective_swaps_own_and_opponent():
    state = with_overrides(
        reset(),
        runner_pos=(2, 3),
        catcher_pos=(7, 1),
        runner_walls=frozenset({(2, 4)}),
        catcher_walls=frozenset({(6, 1)}),
        sprint_charges=2,
        vault_charges=1,
    )
    runner_obs = encode_observation(state, "runner")
    catcher_obs = encode_observation(state, "catcher")

    # Position channels are swapped between perspectives.
    np.testing.assert_array_equal(runner_obs[0], catcher_obs[1])
    np.testing.assert_array_equal(runner_obs[1], catcher_obs[0])
    # Wall channels swapped too.
    np.testing.assert_array_equal(runner_obs[2], catcher_obs[3])
    np.testing.assert_array_equal(runner_obs[3], catcher_obs[2])
    # Charge channels swapped.
    assert runner_obs[4, 0, 0] == pytest.approx(2 / SPRINT_CHARGES)
    assert catcher_obs[4, 0, 0] == pytest.approx(1 / VAULT_CHARGES)
    # Turn channel is shared.
    np.testing.assert_array_equal(runner_obs[6], catcher_obs[6])


def test_observation_position_channels_are_one_hot():
    state = reset()
    obs = encode_observation(state, "runner")
    assert obs[0].sum() == 1.0
    assert obs[1].sum() == 1.0
    rx, ry = state.runner_pos
    assert obs[0, ry, rx] == 1.0
    cx, cy = state.catcher_pos
    assert obs[1, cy, cx] == 1.0
