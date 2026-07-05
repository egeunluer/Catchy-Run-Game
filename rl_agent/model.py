from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_agent.environment import CatchyRunEnv
from rl_agent.opponents import heuristic_opponent, defensive_shooter_opponent
from rl_agent.custom_cnn import policy_kwargs
from catchy_run_game.engine import Agent

def mask_fn(env: CatchyRunEnv):
    return env.unwrapped._info()["action_mask"]

def make_env(trainee_role: Agent = "runner"):
    env = CatchyRunEnv(trainee_role=trainee_role)
    env.set_opponent_pool(
        [heuristic_opponent, defensive_shooter_opponent],
        weights=[0.0, 1.0],
    )
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
    # Continue runner training from latest checkpoint:
    train(load_from="trained_model_checkpoints/runner_models/catchy_run_runner_stage2_v4",
           total_timesteps=800_000, trainee_role="runner",
           save_to="trained_model_checkpoints/runner_models/catchy_run_runner_stage2_v5",
           tb_log_name="runner_stage2_v5", ent_coef=0.005, learning_rate=2e-5)

    # Fresh runner training from scratch (Stage 0):
    # train(trainee_role="runner", total_timesteps=500_000,
    #      save_to="trained_model_checkpoints/runner_models/catchy_run_runner_stage0_v0",
    #      tb_log_name="runner_stage0_v0")
    # python3 -m rl_agent.model
