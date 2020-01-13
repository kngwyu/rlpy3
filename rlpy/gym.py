"""Provide OpenAI-gym interfaces of RLPy3 domains
"""
import gym
import numpy as np
from rlpy import domains
from rlpy import representations as reprs


class RLPyEnv(gym.Env):
    def __init__(self, domain, obs_fn, obs_space):
        self.domain = domain
        self.action_space = gym.spaces.Discrete(domain.actions_num)
        self.observation_space = obs_space
        self.obs_fn = obs_fn

    def step(self, action):
        reward, next_state, terminal, possible_actions = self.domain.step(action)
        obs = self.obs_fn(self.domain, next_state)
        info = {"possible_actions": possible_actions}
        return obs, reward, terminal, info

    def reset(self):
        state, _, _ = self.domain.s0()
        return self.obs_fn(self.domain, state)

    def seed(self, seed=None):
        self.domain.set_seed(seed)

    def render(self, mode="human"):
        self.domain.show_domain()

    def get_obs(self, state):
        return self.obs_fn(self.domain, state)


def gridworld_obs(domain, mode="onehot"):
    if mode == "onehot":
        rep = reprs.Tabular(domain)
        low = np.zeros(rep.features_num)
        high = np.zeros(rep.features_num)
        obs_space = gym.spaces.Box(low, high)

        def obs_fn(_domain, state):
            return rep.phi(state, False).astype(np.float32)

    elif mode == "raw":
        low = np.zeros(2)
        high = np.array(domain.map.shape, dtype=np.float32)
        obs_space = gym.spaces.Box(low, high)

        def obs_fn(_domain, state):
            return state.astype(np.float32)

    elif mode == "image":
        obs_space = gym.spaces
        obs_space = gym.spaces
        shape = 1, *domain.map.shape
        low = np.zeros(shape, dtype=np.float32)
        high = np.ones(shape, dtype=np.float32) * domain.AGENT
        obs_space = gym.spaces.Box(low, high)

        def obs_fn(domain, _state):
            return domain.get_image(_state)

    else:
        raise ValueError("obs_mode {} is not supported".format(mode))

    return obs_fn, obs_space


def gridworld(mapfile, mode="onehot", **kwargs):
    random_goal = "RandomGoal" in mapfile.as_posix()
    domain = domains.GridWorld(mapfile=mapfile, random_goal=random_goal, **kwargs)
    obs_fn, obs_space = gridworld_obs(domain, mode=mode)
    return RLPyEnv(domain, obs_fn, obs_space)


def fr_gridworld(mapfile, mode="onehot", **kwargs):
    domain = domains.FixedRewardGridWorld(mapfile=mapfile, **kwargs)
    obs_fn, obs_space = gridworld_obs(domain, mode=mode)
    return RLPyEnv(domain, obs_fn, obs_space)


def br_gridworld(mapfile, mode="onehot", **kwargs):
    domain = domains.BernoulliGridWorld(mapfile=mapfile, **kwargs)
    obs_fn, obs_space = gridworld_obs(domain, mode=mode)
    return RLPyEnv(domain, obs_fn, obs_space)


def deepsea(size=20, mode="onehot", **kwargs):
    domain = domains.DeepSea(size, **kwargs)
    obs_fn, obs_space = gridworld_obs(domain, mode=mode)
    return RLPyEnv(domain, obs_fn, obs_space)


def pinball(noise, cfg):
    domain = domains.Pinball(noise=noise, config_file=cfg)

    lim = domain.statespace_limits
    obs_space = gym.spaces.Box(low=lim[:, 0], high=lim[:, 1])
    return RLPyEnv(domain, lambda _domain, state: state, obs_space)


def _to_camel(snake_str):
    return "".join(s.title() for s in snake_str.split("_"))


def register_gridworld(mapfile, max_steps=100, threshold=0.9):
    name = mapfile.stem
    gym.envs.register(
        id="RLPyGridWorld{}-v0".format(name),
        entry_point="rlpy.gym:gridworld",
        max_episode_steps=max_steps,
        kwargs=dict(mapfile=mapfile),
        reward_threshold=threshold,
    )
    gym.envs.register(
        id="RLPyGridWorld{}-v1".format(name),
        entry_point="rlpy.gym:gridworld",
        max_episode_steps=max_steps,
        kwargs=dict(mapfile=mapfile, mode="raw"),
        reward_threshold=threshold,
    )
    gym.envs.register(
        id="RLPyGridWorld{}-v2".format(name),
        entry_point="rlpy.gym:gridworld",
        max_episode_steps=max_steps,
        kwargs=dict(mapfile=mapfile, mode="image"),
        reward_threshold=threshold,
    )


for mapfile in domains.GridWorld.DEFAULT_MAP_DIR.glob("*.txt"):
    register_gridworld(mapfile)

for mapfile in domains.FixedRewardGridWorld.DEFAULT_MAP_DIR.glob("*.txt"):
    register_gridworld(mapfile, max_steps=20, threshold=80)

for mapfile in domains.BernoulliGridWorld.DEFAULT_MAP_DIR.glob("*.txt"):
    register_gridworld(mapfile, max_steps=20)

for size in range(4, 40, 4):
    gym.envs.register(
        id=f"RLPyDeepSea{size}-v0",
        entry_point="rlpy.gym:deepsea",
        max_episode_steps=size,
        kwargs=dict(size=size),
        reward_threshold=0.9,
    )
    gym.envs.register(
        id=f"RLPyDeepSea{size}-v1",
        entry_point="rlpy.gym:deepsea",
        max_episode_steps=size,
        kwargs=dict(size=size, mode="raw"),
        reward_threshold=0.9,
    )
    gym.envs.register(
        id=f"RLPyDeepSea{size}-v2",
        entry_point="rlpy.gym:deepsea",
        max_episode_steps=size,
        kwargs=dict(size=size, mode="image"),
        reward_threshold=0.9,
    )


for cfgfile in domains.Pinball.DEFAULT_CONFIG_DIR.glob("*.json"):
    name = _to_camel(cfgfile.stem[len("pinball_") :])
    gym.envs.register(
        id=f"RLPyPinball{name}-v0",
        entry_point="rlpy.gym:pinball",
        max_episode_steps=1000,
        kwargs=dict(noise=0.0, cfg=cfgfile),
        reward_threshold=9000,
    )
    gym.envs.register(
        id=f"RLPyPinball{name}-v1",
        entry_point="rlpy.gym:pinball",
        max_episode_steps=1000,
        kwargs=dict(noise=0.1, cfg=cfgfile),
        reward_threshold=8000,
    )
