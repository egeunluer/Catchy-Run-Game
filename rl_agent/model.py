from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.callbacks import CheckpointCallback

from catchy_run.rl_agent.environment import CatchyRunEnv
from catchy_run.rl_agent.opponents import heuristic_opponent
from catchy_run.rl_agent.custom_cnn import policy_kwargs
from catchy_run.catchy_run_game.engine import Agent

def mask_fn(env: CatchyRunEnv):
    """ActionMasker will call this on every step to extract the current action mask."""
    return env.unwrapped._info()["action_mask"]

def make_env(trainee_role: Agent):
    if trainee_role == "runner":
        env = CatchyRunEnv(trainee_role=trainee_role)
        env.set_opponent_pool([env._default_opponent, heuristic_opponent], weights=[0.1, 0.9])
    else:
        from catchy_run.catchy_run_game.agents.rl_runner import rl_runner_policy
        env = CatchyRunEnv(trainee_role=trainee_role, opponent_policy=lambda state: rl_runner_policy(state))
    env = ActionMasker(env, mask_fn)
    return env

def train(trainee_role: Agent,
          total_timesteps: int = 200_000,
          load_from: str | None = None,
          save_to: str | None = None,
          tb_log_name: str | None = None,
          ent_coef: float = 0.01,
          learning_rate : float = 3e-4
          ):
    save_to = save_to or f"catchy_run_{trainee_role}_stage0_v0"
    tb_log_name = tb_log_name or f"{trainee_role}_stage0_v0"
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/",
        name_prefix=f"catchy_run_{trainee_role}"
    )
    env = make_env(trainee_role)
    if load_from is not None:
        model = MaskablePPO.load(load_from, env=env)
        model.ent_coef = ent_coef
        model.learning_rate = learning_rate
        model._setup_lr_schedule()
        model.n_steps = 4096
        model.batch_size = 256
        model.rollout_buffer = MaskableRolloutBuffer(
            model.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
        )
    else:
        model = MaskablePPO(
            policy="CnnPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            verbose=1,
            tensorboard_log="./tb_logs/",
            device="auto"
        )
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, reset_num_timesteps=(load_from is None), callback=checkpoint_callback)
    model.save(save_to)
    return model


if __name__ == "__main__":
    #Change the signature of this train method to continue from a trained model
    train(load_from= None,total_timesteps= 300000,trainee_role="catcher", save_to="catchy_run/trained_model_checkpoints/catcher_models/catchy_run_catcher_stage1_v0_0", tb_log_name="catcher_stage1_v0_0", ent_coef=0.01, learning_rate=1e-4)
    # Catcher Stage 1 example (fresh train against the heuristic runner):
    # train(trainee_role="catcher", total_timesteps=200_000, save_to="catchy_run_catcher_stage1_v0", tb_log_name="catcher_stage1_v0")
