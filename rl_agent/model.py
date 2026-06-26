from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_agent.environment import CatchyRunEnv
from rl_agent.opponents import heuristic_opponent, make_rl_opponent, defensive_shooter_opponent
from rl_agent.custom_cnn import policy_kwargs
from catchy_run_game.engine import Agent

# Bullet-spamming catcher used to teach the runner to respect/dodge bullets.
SPAM_CATCHER_CKPT = "trained_model_checkpoints/catcher_models/catchy_run_catcher_stage1_v0_1.zip"

def mask_fn(env: CatchyRunEnv):
    """ActionMasker will call this on every step to extract the current action mask."""
    return env.unwrapped._info()["action_mask"]

def make_env(trainee_role: Agent):
    if trainee_role == "runner":
        env = CatchyRunEnv(trainee_role=trainee_role)
        # Stochastic so the runner sees varied bullet patterns, not one fixed line.
        spam_catcher = make_rl_opponent(SPAM_CATCHER_CKPT, role="catcher", deterministic=False)
        env.set_opponent_pool(
            # random shooter (bullets from anywhere) | heuristic chaser (keeps evasion
            # sharp) | defensive shooter (chaser that fires exactly when a special is
            # threatened — teaches purposeful dodging) | the spammer (the actual bullet
            # problem to solve). Tune weights.
            [defensive_shooter_opponent, spam_catcher],
            weights=[0.8, 0.2],
        )
    else:
        from catchy_run_game.agents.rl_runner import rl_runner_policy
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
    train(load_from= None,total_timesteps= 500000,trainee_role="catcher", save_to="trained_model_checkpoints/catcher_models/catchy_run_catcher_stage1_v0", tb_log_name="catcher_stage1_v0", ent_coef=0.05, learning_rate=2e-4)
    # Catcher Stage 1 example (fresh train against the heuristic runner):
    # train(trainee_role="catcher", total_timesteps=200_000, save_to="catchy_run_catcher_stage1_v0", tb_log_name="catcher_stage1_v0")

    # Teach the runner to dodge: continue the current runner against the pool that now
    # includes the bullet-spamming catcher (SPAM_CATCHER_CKPT). Swap load_from to your runner ckpt.
    """train(load_from="trained_model_checkpoints/runner_models/catchy_run_runner_stage1_v1",
           total_timesteps=400_000, trainee_role="runner",
           save_to="trained_model_checkpoints/runner_models/catchy_run_runner_stage1_v2",
           tb_log_name="runner_stage1_v2", ent_coef=0.02, learning_rate=5e-5)"""
    #python3 -m rl_agent.model
