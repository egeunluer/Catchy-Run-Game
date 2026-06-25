import numpy as np
from sb3_contrib import MaskablePPO

from rl_agent.environment import CatchyRunEnv
from rl_agent.opponents import heuristic_opponent, defensive_shooter_opponent
from catchy_run_game.engine import Agent, TURN_LIMIT

def evaluate(trainee_role: Agent, model_path: str, n_episodes: int = 400, base_seed: int = 0, opponent_policy = defensive_shooter_opponent):
    if trainee_role == "runner":
        env = CatchyRunEnv(trainee_role=trainee_role, opponent_policy=opponent_policy)
    else:
        from catchy_run_game.agents.rl_runner import rl_runner_policy
        env = CatchyRunEnv(trainee_role=trainee_role, opponent_policy=lambda state: rl_runner_policy(state))
    model = MaskablePPO.load(model_path)

    wins, losses = 0, 0
    lengths = []
    captures_per_episode = []
    active_kills = 0
    kill_turns = []

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
        if end_state.winner == trainee_role:
            wins += 1
            if trainee_role == "catcher" and end_state.turn < TURN_LIMIT:
                active_kills += 1
                kill_turns.append(end_state.turn)
        else:
            losses += 1

    print(f"\nEvaluated {model_path} over {n_episodes} episodes\n")
    print("-" * 46)
    print(f"{'overall':<10}{wins:>6} Wins{n_episodes - wins:>8} Losses{wins / n_episodes:>11.1%}")
    mean_captures = np.mean(captures_per_episode) if captures_per_episode else 0.0
    mean_lengths = np.mean(lengths) if lengths else 0.0

    if trainee_role == "runner":
        print(f"Mean Captures: {mean_captures:.2f} / 7")
        print(f"Mean Lengths: {mean_lengths:.2f}")
    else:
        timeout_wins = wins - active_kills
        mean_kill_turn = float(np.mean(kill_turns)) if kill_turns else 0.0
        print(f"Active kills:   {active_kills:>6}  ({active_kills / n_episodes:>6.1%})")
        print(f"Timeout wins:   {timeout_wins:>6}  ({timeout_wins / n_episodes:>6.1%})")
        print(f"Mean kill turn (active kills only): {mean_kill_turn:.2f} / {TURN_LIMIT}")
        print(f"Mean runner captures conceded: {mean_captures:.2f} / 7")
        print(f"Mean Lengths: {mean_lengths:.2f}")

if __name__ == "__main__":
      evaluate(trainee_role= "runner",model_path="trained_model_checkpoints/runner_models/catchy_run_runner_stage1_v2.zip", n_episodes=400, )
      # Catcher evaluation example:
      # evaluate(trainee_role="catcher", model_path="catchy_run_catcher_stage1_v0", n_episodes=400)
      #python3 -m rl_agent.evaluation
