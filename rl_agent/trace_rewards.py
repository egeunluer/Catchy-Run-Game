"""Trace reward-shaping components over random episodes.

Run from project root:
    python3 -m grid.rl_agent.trace_rewards

Plays N episodes with random legal actions for the runner trainee against the
env's default opponent, intercepts every call to RewardShaper.shape, and prints
per-component statistics. Use as a sanity check after edits to reward_shaping.py:
every component should fire at the expected sign and magnitude, no component
should silently stay zero, no component should dominate the sum.
"""
import numpy as np
from sb3_contrib import MaskablePPO
from grid.rl_agent.environment import CatchyRunEnv


COMPONENTS = ["base", "alive", "capture", "catcher", "projectile",
              "attraction", "sprint_waste", "urgency"]


def run(num_episodes: int = 50, seed: int = 42):
    env = CatchyRunEnv(trainee_role="runner")
    shaper = env.reward_shaper
    rng = np.random.default_rng(seed)

    per_step = []
    per_episode = []

    original_shape = shaper.shape

    def wrapped_shape(prev, curr, base):
        # Mirror shape()'s short-circuit: only log when shaping actually runs.
        if shaper.trainee_role == "runner" and not curr.terminated:
            per_step.append({
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
            })
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


def summarize(per_step, per_episode):
    print(f"Steps shaped: {len(per_step)}  Episodes: {len(per_episode)}\n")
    header = f"{'component':<14} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'%nonzero':>10}"
    print(header)
    print("-" * len(header))
    for c in COMPONENTS:
        vals = np.array([s[c] for s in per_step], dtype=np.float64)
        nonzero_pct = 100.0 * (vals != 0).mean()
        print(f"{c:<14} {vals.mean():>+10.4f} {vals.std():>10.4f} "
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


def trace_single_episode(seed: int = 0):
    """Print every shaped step of one episode. Useful for first-pass eyeballing."""
    env = CatchyRunEnv(trainee_role="runner")
    shaper = env.reward_shaper
    rng = np.random.default_rng(seed)
    original_shape = shaper.shape

    def wrapped_shape(prev, curr, base):
        result = original_shape(prev, curr, base)
        if shaper.trainee_role == "runner" and not curr.terminated:
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
    model_path = "catchy_run_runner_stage0_v1_1"
    model = MaskablePPO.load(model_path)
    print("=== Single-episode trace (seed=0) ===")
    trace_single_episode(seed=0)
    print("\n=== 50-episode summary (seed=42) ===")
    per_step, per_episode = run(num_episodes=50)
    summarize(per_step, per_episode)
