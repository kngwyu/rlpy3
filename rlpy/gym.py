"""Provide OpenAI-gym interfaces of RLPy3 domains
"""
import gym
import numpy as np
from rlpy import domains
from rlpy import representations as reprs


class RlpyEnv(gym.Env):
    def __init__(self, domain, obs_fn, obs_space):
        self.domain = domain
        self.action_space = gym.spaces.Discrete(domain.actions_num)
        self.observation_space = obs_space
        self.obs_fn = obs_fn

    def step(self, action):
        reward, next_state, terminal, possible_actions = self.domain.step(action)
        obs = self.obs_fn(next_state)
        info = {"possible_actions": possible_actions}
        return obs, reward, terminal, info

    def reset(self):
        state, _, _ = self.domain.s0()
        return self.obs_fn(state)

    def seed(self, seed=None):
        self.domain.set_seed(seed)

    def render(self, mode="human"):
        self.domain.show_domain()


def gridworld_obs(domain, mode="onehot"):
    if mode == "onehot":
        rep = reprs.Tabular(domain)
        low = np.zeros(rep.features_num)
        high = np.zeros(rep.features_num)
        obs_space = gym.spaces.Box(low, high)

        def obs_fn(state):
            return rep.phi(state, False).astype(np.float32)
    elif mode == "raw":
        low = np.zeros(2)
        high = np.array(domain.map.shape, dtype=np.float32)
        obs_space = gym.spaces.Box(low, high)

        def obs_fn(state):
            return state.astype(np.float32)
    else:
        raise ValueError("obs_mode {} is not supported".format(mode))

    return obs_fn, obs_space


def gridworld(name="4x5", mode="onehot", **kwargs):
    mapfile = domains.GridWorld.default_map(name + ".txt")
    domain = domains.GridWorld(mapfile=mapfile, **kwargs)
    obs_fn, obs_space = gridworld_obs(domain, mode=mode)
    return RlpyEnv(domain, obs_fn, obs_space)


def deepsea(size=20, mode="onehot", **kwargs):
    domain = domains.DeepSea(size, **kwargs)
    obs_fn, obs_space = gridworld_obs(domain, mode=mode)
    return RlpyEnv(domain, obs_fn, obs_space)


gym.envs.register(
    id="GridWorld4x5-v0",
    entry_point="rlpy.gym:gridworld",
    max_episode_steps=100,
    reward_threshold=0.9,
)

gym.envs.register(
    id="GridWorld4x5-v1",
    entry_point="rlpy.gym:gridworld",
    max_episode_steps=100,
    kwargs=dict(mode="raw"),
    reward_threshold=0.9,
)


for size in [4, 8, 12, 16, 20, 24, 28, 32]:
    gym.envs.register(
        id="DeepSea{}-v0".format(size),
        entry_point="rlpy.gym:deepsea",
        max_episode_steps=100,
        kwargs=dict(size=size),
        reward_threshold=0.9,
    )
    gym.envs.register(
        id="DeepSea{}-v1".format(size),
        entry_point="rlpy.gym:deepsea",
        max_episode_steps=100,
        kwargs=dict(size=size, mode="raw"),
        reward_threshold=0.9,
    )
