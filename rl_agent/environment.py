from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from catchy_run_game import engine as engine
from rl_agent.reward_shaping import RunnerRewardShaper, CatcherRewardShaper

class CatchyRunEnv(gym.Env):

    metadata = {"render_modes": [], "name": "catchy_run_v0"}


    def _default_opponent(self, state):
        legal = np.flatnonzero(engine.legal_action_mask(state))
        if state.current_agent == "catcher":
            moves = legal[legal < 8]
            shoots = legal[legal >= 8]
            if len(shoots) and len(moves):
                pool = moves if self.np_random.random() < 0.80 else shoots
                return int(self.np_random.choice(pool))
        return int(self.np_random.choice(legal))

    def __init__(self, trainee_role : engine.Agent, opponent_policy = None):
        super().__init__()
        self._opponent_pool = None
        self._opponent_weights = None
        self._opponent_policy = opponent_policy or self._default_opponent
        self._current_opponent = self._opponent_policy
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(9,7,7), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(16)
        self.state: Optional[engine.GameState] = None
        self.trainee_role: engine.Agent = trainee_role
        if trainee_role == "runner":
            self.reward_shaper = RunnerRewardShaper()
        else:
            self.reward_shaper = CatcherRewardShaper()

    #Extract the observation from the engine + add the perspective channel at the start
    def _obs(self) -> np.ndarray:
        return engine.encode_observation(self.state, self.trainee_role)

    #Extract the action mask and the trainee role
    def _info(self):
        mask = engine.legal_action_mask(self.state)
        return {"action_mask" : mask, "trainee_role": self.trainee_role}

    def _shape_reward(self, prev, curr, base):
        return self.reward_shaper.shape(prev, curr, base)

        #Opponent configuration
    def set_opponent(self, opponent_policy):
        self._opponent_policy = opponent_policy
        self._opponent_pool = None

    def set_opponent_pool(self, opponents: list, weights: list | None = None):
        self._opponent_pool = list(opponents)
        self._opponent_weights = weights
        self._opponent_policy = None

    def _sample_opponent(self):
        if self._opponent_pool is not None:
            if self._opponent_weights is not None:
                idx = self.np_random.choice(len(self._opponent_pool), p= self._opponent_weights)
            else:
                idx = self.np_random.integers(len(self._opponent_pool))
            return self._opponent_pool[idx]
        return self._opponent_policy

    """Opponent simulation so that we train the trainee. Take a step in the engine, but not in the environment"""
    def _play_opponent_turn(self) -> tuple[float, bool]:
        if self.state.terminated:
            return 0.0, True
        opp_action = self._current_opponent(self.state)
        mask = engine.legal_action_mask(self.state)
        if not mask[opp_action]:
            opp_action = int(self.np_random.choice(np.flatnonzero(mask)))
        new_state, r_runner, r_catcher, terminated, _, _ = engine.step(self.state, opp_action)
        self.state = new_state
        trainee_reward = r_runner if self.trainee_role == "runner" else r_catcher
        return trainee_reward, terminated

    def reset(self, *, seed=None, options = None):
        super().reset(seed=seed)
        self.state = engine.reset(seed=seed)
        self._current_opponent = self._sample_opponent()
        if self.state.current_agent != self.trainee_role:
            self._play_opponent_turn()
        return self._obs(), self._info()

    def step(self, action):
        mask = engine.legal_action_mask(self.state)
        if not mask[action]:
            action = int(self.np_random.choice(np.flatnonzero(mask)))
        prev_state = self.state
        new_state, r_runner, r_catcher, terminated, _, _ = engine.step(self.state, action)
        self.state = new_state
        reward = r_runner if self.trainee_role == "runner" else r_catcher
        if not terminated:
            opp_reward, terminated = self._play_opponent_turn()
            reward += opp_reward
        reward = self._shape_reward(prev_state, self.state, reward)
        return self._obs(), reward, terminated, False, self._info()

#check_env(CatchyRunEnv(trainee_role="runner"))


