"""Trace reward-shaping components over random episodes.

Run from project root:
    python3 -m catchy_run.rl_agent.trace_rewards

Plays N episodes with a trained policy for the chosen trainee against the
env's default opponent, intercepts every call to RewardShaper.shape, and prints
per-component statistics. Use as a sanity check after edits to reward_shaping.py:
every component should fire at the expected sign and magnitude, no component
should silently stay zero, no component should dominate the sum.
"""
import numpy as np
from sb3_contrib import MaskablePPO
from catchy_run.rl_agent.environment import CatchyRunEnv
from catchy_run.catchy_run_game.engine import Agent


RUNNER_COMPONENTS = ["base", "alive", "capture", "catcher", "projectile",
                     "attraction", "sprint_waste", "urgency", "unsafe_capture"]

CATCHER_COMPONENTS = ["base", "special_defense", "special_blocking", "time_advantage", "chase"]


def _runner_step_dict(shaper, prev, curr, base):
    return {
        "turn": curr.turn,
        "captured": len(curr.captured_squares),
        "base": base,
        "alive": shaper.ALIVE_BONUS,
        "capture": shaper._capture_bonus(prev, curr),
        "catcher": shaper._catcher_distance_rewarding(curr, prev),
        "projectile": shaper._projectile_threat_penalty(curr),
        "attraction": shaper._special_attraction(curr),
        "sprint_waste": shaper._sprint_waste_penalty(prev, curr),
        "urgency": shaper._urgency_penalty(curr),
        "unsafe_capture": shaper._unsafe_capture_penalty(prev, curr),
    }


def _catcher_step_dict(shaper, prev, curr, base):
    return {
        "turn": curr.turn,
        "captured": len(curr.captured_squares),
        "base": base,
        "special_defense": shaper._special_defense_bonus(prev, curr),
        "special_blocking": shaper._special_blocking_attraction(curr),
        "time_advantage": shaper._time_advantage_bonus(curr),
        "chase": shaper._chase_bonus(prev, curr),
    }


def _components_and_builder(trainee_role: Agent):
    if trainee_role == "runner":
        return RUNNER_COMPONENTS, _runner_step_dict
    return CATCHER_COMPONENTS, _catcher_step_dict


def run(num_episodes: int = 50, seed: int = 42, trainee_role: Agent = "runner"):
    env = CatchyRunEnv(trainee_role=trainee_role)
    shaper = env.reward_shaper
    rng = np.random.default_rng(seed)
    _, step_builder = _components_and_builder(trainee_role)

    per_step = []
    per_episode = []

    original_shape = shaper.shape

    def wrapped_shape(prev, curr, base):
        # Mirror shape()'s short-circuit: only log when shaping actually runs.
        if not curr.terminated:
            per_step.append(step_builder(shaper, prev, curr, base))
        return original_shape(prev, curr, base)

    shaper.shape = wrapped_shape

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        start = len(per_step)
        total = 0.0
        terminated = False
        while not terminated:
            mask = info["action_mask"]
            action, _ = model.predict(obs, action_masks=mask, deterministic=False)
            #action = int(rng.choice(legal))
            action = int(action)
            obs, reward, terminated, _, info = env.step(action)
            total += reward
        per_episode.append({
            "total_reward": total,
            "captured": len(env.state.captured_squares),
            "turn": env.state.turn,
            "winner": env.state.winner,
            "steps_logged": len(per_step) - start,
        })

    return per_step, per_episode


def summarize(per_step, per_episode, trainee_role: Agent = "runner"):
    components, _ = _components_and_builder(trainee_role)
    print(f"Steps shaped: {len(per_step)}  Episodes: {len(per_episode)}\n")
    header = f"{'component':<16} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'%nonzero':>10}"
    print(header)
    print("-" * len(header))
    for c in components:
        vals = np.array([s[c] for s in per_step], dtype=np.float64)
        nonzero_pct = 100.0 * (vals != 0).mean()
        print(f"{c:<16} {vals.mean():>+10.4f} {vals.std():>10.4f} "
              f"{vals.min():>+10.4f} {vals.max():>+10.4f} {nonzero_pct:>9.1f}%")

    ep_totals = np.array([e["total_reward"] for e in per_episode])
    ep_caps = np.array([e["captured"] for e in per_episode])
    winners = [e["winner"] for e in per_episode]
    print()
    print(f"Per-episode reward: mean={ep_totals.mean():+.3f}  std={ep_totals.std():.3f}  "
          f"min={ep_totals.min():+.3f}  max={ep_totals.max():+.3f}")
    print(f"Captures/episode:   mean={ep_caps.mean():.2f}  max={int(ep_caps.max())}")
    print(f"Winners: runner={winners.count('runner')}  catcher={winners.count('catcher')}  "
          f"none={winners.count(None)}")


def trace_single_episode(seed: int = 0, trainee_role: Agent = "runner"):
    """Print every shaped step of one episode. Useful for first-pass eyeballing."""
    env = CatchyRunEnv(trainee_role=trainee_role)
    shaper = env.reward_shaper
    rng = np.random.default_rng(seed)
    original_shape = shaper.shape

    def wrapped_shape(prev, curr, base):
        result = original_shape(prev, curr, base)
        if curr.terminated:
            return result
        if trainee_role == "runner":
            parts = [
                f"turn={curr.turn:2d}",
                f"cap={len(curr.captured_squares)}",
                f"alive=+{shaper.ALIVE_BONUS:.3f}",
                f"capture={shaper._capture_bonus(prev, curr):+.3f}",
                f"catcher={shaper._catcher_distance_rewarding(curr, prev):+.3f}",
                f"proj={shaper._projectile_threat_penalty(curr):+.3f}",
                f"attr={shaper._special_attraction(curr):+.3f}",
                f"sprint_waste={shaper._sprint_waste_penalty(prev, curr):+.3f}",
                f"urgency={shaper._urgency_penalty(curr):+.3f}",
                f"unsafe_cap={shaper._unsafe_capture_penalty(prev, curr):+.3f}",
                f"=> {result:+.3f}",
            ]
        else:
            parts = [
                f"turn={curr.turn:2d}",
                f"cap={len(curr.captured_squares)}",
                f"special_defense={shaper._special_defense_bonus(prev, curr):+.3f}",
                f"special_blocking={shaper._special_blocking_attraction(curr):+.3f}",
                f"time_advantage={shaper._time_advantage_bonus(curr):+.3f}",
                f"chase={shaper._chase_bonus(prev, curr):+.3f}",
                f"=> {result:+.3f}",
            ]
        print("  ".join(parts))
        return result

    shaper.shape = wrapped_shape

    obs, info = env.reset(seed=seed)
    terminated = False
    while not terminated:
        mask = info["action_mask"]
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        action = int(action)
        obs, reward, terminated, _, info = env.step(action)
    print(f"\nTerminal: turn={env.state.turn}  winner={env.state.winner}  "
          f"captured={len(env.state.captured_squares)}")


if __name__ == "__main__":
    # Runner trace example:
    model_path = "catchy_run_runner_stage0_v1_4_1"
    trainee_role: Agent = "runner"

    # Catcher trace example (swap in once a catcher checkpoint exists):
    # model_path = "catchy_run_catcher_stage0_v0"
    # trainee_role = "catcher"

    model = MaskablePPO.load(model_path)
    print(f"=== Single-episode trace (seed=0, role={trainee_role}) ===")
    trace_single_episode(seed=0, trainee_role=trainee_role)
    print(f"\n=== 50-episode summary (seed=42, role={trainee_role}) ===")
    per_step, per_episode = run(num_episodes=50, trainee_role=trainee_role)
    summarize(per_step, per_episode, trainee_role=trainee_role)
