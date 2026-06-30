import random

from catchy_run_game.agents import heuristic
from catchy_run_game.agents.heuristic import shooting_catcher_policy as defensive_shooter_opponent
from catchy_run_game import engine


def heuristic_opponent(state):
    if state.current_agent == "runner":
        return heuristic.runner_policy(state)
    else:
        return heuristic.catcher_policy(state)


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