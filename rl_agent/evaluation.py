import numpy as np
from sb3_contrib import MaskablePPO

from rl_agent.environment import CatchyRunEnv
from rl_agent.opponents import heuristic_opponent, defensive_shooter_opponent

def evaluate(model_path: str, n_episodes: int = 400, base_seed: int = 0, opponent_policy=defensive_shooter_opponent):
    env = CatchyRunEnv(trainee_role="runner", opponent_policy=opponent_policy)
    model = MaskablePPO.load(model_path)

    wins, losses = 0, 0
    lengths = []
    captures_per_episode = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        terminated = False
        steps = 0
        while not terminated:
            mask = info["action_mask"]
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, _, info = env.step(int(action))
            steps += 1

        end_state = env.unwrapped.state
        captures_per_episode.append(len(end_state.captured_squares))
        lengths.append(steps)
        if end_state.winner == "runner":
            wins += 1
        else:
            losses += 1

    print(f"\nEvaluated {model_path} over {n_episodes} episodes\n")
    print("-" * 46)
    print(f"{'overall':<10}{wins:>6} Wins{n_episodes - wins:>8} Losses{wins / n_episodes:>11.1%}")
    mean_captures = np.mean(captures_per_episode) if captures_per_episode else 0.0
    mean_lengths = np.mean(lengths) if lengths else 0.0
    print(f"Mean Captures: {mean_captures:.2f} / 7")
    print(f"Mean Lengths: {mean_lengths:.2f}")

if __name__ == "__main__":
    evaluate(model_path="trained_model_checkpoints/runner_models/catchy_run_runner_stage1_v2.zip", n_episodes=400)
    # python3 -m rl_agent.evaluation
