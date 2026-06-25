"""Adapter exposing the trained RL runner model as a (state, rng) -> int policy.

Mirrors the interface of `agents.heuristic.runner_policy` so the pygame UI
can swap policies without further special-casing. The model is loaded lazily
on the first call and cached at module level.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from .. import engine

DEFAULT_MODEL_PATH = "trained_model_checkpoints/runner_models/catchy_run_runner_stage1_v0.zip"

_model = None


def _load_model(path: Path):
    global _model
    if _model is not None:
        return _model
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise RuntimeError(
            "sb3_contrib is required for the RL runner mode. "
            "Install it with `pip install sb3-contrib`."
        ) from e
    if not Path(path).exists():
        raise RuntimeError(f"RL runner checkpoint not found at {path}")
    _model = MaskablePPO.load(str(path))
    return _model


def rl_runner_policy(
    state: engine.GameState, rng: Optional[random.Random] = None
) -> int:
    if state.current_agent != "runner":
        raise ValueError("rl_runner_policy called when current_agent is not runner")
    model = _load_model(DEFAULT_MODEL_PATH)
    obs = engine.encode_observation(state, "runner")
    mask = engine.legal_action_mask(state)
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return int(action)
