from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback

from grid.rl_agent.environment import CatchyRunEnv
from grid.rl_agent.opponents import heuristic_opponent
from grid.rl_agent.custom_cnn import policy_kwargs
from grid.catcher_vs_runner.engine import Agent

def mask_fn(env: CatchyRunEnv):
    """ActionMasker will call this on every step to extract the current action mask."""
    return env.unwrapped._info()["action_mask"]

def make_env(trainee_role: Agent):
    env = CatchyRunEnv(trainee_role= trainee_role)
    env = ActionMasker(env, mask_fn)
    return env

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="catchy_run_runner"
)

def train(trainee_role: Agent,
          total_timesteps: int = 200_000,
          load_from: str | None = None,
          save_to: str = "catchy_run_runner_stage0_v0",
          tb_log_name: str = "runner_stage0_v0",
          ent_coef: float = 0.01
          ):
    env = make_env(trainee_role)
    if load_from is not None:
        model = MaskablePPO.load(load_from, env=env)
        model.ent_coef = ent_coef
    else:
        model = MaskablePPO(
            policy="CnnPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./tb_logs/",
            device="auto"
        )
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, reset_num_timesteps=(load_from is None), callback=checkpoint_callback)
    model.save(save_to)
    return model


if __name__ == "__main__":
    #Change the signature of this train method to continue from a trained model
    train(load_from= "catchy_run_runner_stage0_v0",total_timesteps= 300000,trainee_role="runner", save_to="catchy_run_runner_stage0_v1", tb_log_name="runner_stage0_v1")
