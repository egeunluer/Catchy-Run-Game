import numpy as np
from sb3_contrib import MaskablePPO

from grid.rl_agent.environment import CatchyRunEnv
from grid.rl_agent.opponents import heuristic_opponent

def evaluate(model_path: str, n_episodes: int = 400, base_seed: int = 0):
    env = CatchyRunEnv(opponent_policy=heuristic_opponent)
    model = MaskablePPO.load(model_path)

    results = {
        "runner": {"wins": 0, "losses": 0, "lengths": []},
        "catcher": {"wins": 0, "losses": 0, "lengths": []}
    }

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        role = info["trainee_role"]
        terminated = False
        total_reward = 0.0
        steps = 0
        while not terminated:
            mask = info["action_mask"]
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, _, info = env.step(int(action))
            total_reward += reward
            steps += 1

        if total_reward > 0:
            results[role]["wins"] +=1
        else:
            results[role]["losses"] += 1
        results[role]["lengths"].append(steps)

    print(f"\nEvaluated {model_path} over {n_episodes} episodes\n")
    print(f"{'role':<10}{'wins':>6}{'losses':>8}{'win rate':>12}{'avg len':>10}")
    print("-" * 46)
    for role in ("runner", "catcher"):
        r = results[role]
        total = r["wins"] + r["losses"]
        rate = r["wins"] / total if total else 0.0
        avg_len = np.mean(r["lengths"]) if r["lengths"] else 0.0
        print(f"{role:<10}{r['wins']:>6}{r['losses']:>8}{rate:>11.1%}{avg_len:>10.1f}")

    overall_wins = results["runner"]["wins"] + results["catcher"]["wins"]
    print("-" * 46)
    print(f"{'overall':<10}{overall_wins:>6}{n_episodes - overall_wins:>8}{overall_wins / n_episodes:>11.1%}")

if __name__ == "__main__":
      evaluate("checkpoints/catchy_run_200000_steps", n_episodes=400)

