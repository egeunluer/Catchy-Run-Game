import numpy as np
from sb3_contrib import MaskablePPO

from grid.rl_agent.environment import CatchyRunEnv
from grid.rl_agent.opponents import heuristic_opponent
from grid.catcher_vs_runner.engine import Agent

def evaluate(trainee_role: Agent, model_path: str, n_episodes: int = 400, base_seed: int = 0, opponent_policy = None):
    env = CatchyRunEnv(trainee_role=trainee_role, opponent_policy=opponent_policy)
    model = MaskablePPO.load(model_path)

    wins, losses, lengths = 0, 0, []
    captures_per_episode = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        terminated = False
        total_reward = 0.0
        steps = 0
        while not terminated:
            mask = info["action_mask"]
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, _, info = env.step(int(action))
            total_reward += reward
            steps += 1

        captures_per_episode.append(len(env.unwrapped.state.captured_squares))
        if total_reward > 0:
            wins +=1
        else:
            losses += 1
        lengths.append(steps)

    print(f"\nEvaluated {model_path} over {n_episodes} episodes\n")
    print("-" * 46)
    print(f"{'overall':<10}{wins:>6} Wins{n_episodes - wins:>8} Losses{wins / n_episodes:>11.1%}")
    mean_captures = np.mean(captures_per_episode) if captures_per_episode else 0.0
    mean_lengths = np.mean(lengths) if lengths else 0.0
    print(f"Mean Captures: {mean_captures:.2f} / 7")
    print(f"Mean Lengths: {mean_lengths:.2f}")

if __name__ == "__main__":
      evaluate(trainee_role= "runner",model_path="catchy_run_runner_stage0_v3", n_episodes=400,)

