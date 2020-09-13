"""Provide OpenAI-gym interfaces of RLPy3 domains
"""
import gym
import numpy as np
from rlpy import domains
from rlpy import representations as reprs


class RLPyEnv(gym.Env):
    def __init__(self, domain, obs_fn, obs_space):
        self.domain = domain
        self.action_space = gym.spaces.Discrete(domain.num_actions)
        self.raw_observation_space = _get_box_space(domain)
        self.observation_space = obs_space
        self.obs_fn = obs_fn
        self._init_state_reserved = None

    def step(self, action):
        reward, next_state, terminal, possible_actions = self.domain.step(action)
        obs = self.obs_fn(self.domain, next_state)
        info = {"possible_actions": possible_actions, "raw_obs": next_state}
        return obs, reward, terminal, info

    def reset(self):
        state, _, _ = self.domain.s0()
        self._init_state_reserved = state
        return self.obs_fn(self.domain, state)

    def initial_raw_obs(self):
        return self._init_state_reserved

    def seed(self, seed=None):
        self.domain.set_seed(seed)

    def render(self, mode="human"):
        self.domain.show_domain()

    def get_obs(self, state):
        return self.obs_fn(self.domain, state)

    def close(self):
        self.domain.close_visualizations()


def gridworld_obs(domain, mode="onehot"):
    if mode == "onehot":
        rep = reprs.Tabular(domain)
        low = np.zeros(rep.features_num)
        high = np.zeros(rep.features_num)
        obs_space = gym.spaces.Box(low, high)

        def obs_fn(_domain, state):
            return rep.phi(state, False).astype(np.float32)

    elif mode == "raw":
        obs_space = _get_box_space(domain)

        def obs_fn(_domain, state):
            return state.astype(np.float32)

    elif mode == "image":
        shape = 1, *domain.map.shape
        low = np.zeros(shape, dtype=np.float32)
        high = np.ones(shape, dtype=np.float32) * domain.AGENT
        obs_space = gym.spaces.Box(low, high)

        def obs_fn(domain, _state):
            return domain.get_image(_state)

    elif mode == "binary-image":
        shape = domain.MAP_CATEGORY, *domain.map.shape
        low = np.zeros(shape, dtype=np.float32)
        high = np.ones(shape, dtype=np.float32)
        obs_space = gym.spaces.Box(low, high)

        def obs_fn(domain, _state):
            return domain.get_binary_image(_state)

    else:
        raise ValueError("obs_mode {} is not supported".format(mode))

    return obs_fn, obs_space


def gridworld(mapfile, mode="onehot", cls=domains.GridWorld, **kwargs):
    random_goal = "RandomGoal" in mapfile.as_posix()
    domain = cls(mapfile=mapfile, random_goal=random_goal, **kwargs)
    obs_fn, obs_space = gridworld_obs(domain, mode=mode)
    return RLPyEnv(domain, obs_fn, obs_space)


def deepsea(size=20, mode="onehot", **kwargs):
    domain = domains.DeepSea(size, **kwargs)
    obs_fn, obs_space = gridworld_obs(domain, mode=mode)
    return RLPyEnv(domain, obs_fn, obs_space)


def lifegame(mapfile, rule, mode="image", **kwargs):
    domain = domains.LifeGameSurvival(mapfile, rule)
    obs_fn, obs_space = gridworld_obs(domain, mode=mode)
    return RLPyEnv(domain, obs_fn, obs_space)


def pinball(noise, cfg):
    domain = domains.Pinball(noise=noise, config_file=cfg)
    obs_space = _get_box_space(domain)
    return RLPyEnv(domain, lambda _domain, state: state, obs_space)


def _to_camel(snake_str):
    return "".join(s.title() for s in snake_str.split("_"))


def _get_box_space(domain):
    lim = domain.raw_statespace_limits
    return gym.spaces.Box(low=lim[:, 0], high=lim[:, 1], dtype=lim.dtype)


def register_gridworld(mapfile, cls=domains.GridWorld, max_steps=100, threshold=0.9):
    name = cls.__name__ + mapfile.stem
    gym.envs.register(
        id=f"RLPy{name}-v0",
        entry_point="rlpy.gym:gridworld",
        max_episode_steps=max_steps,
        kwargs=dict(mapfile=mapfile, cls=cls),
        reward_threshold=threshold,
    )
    gym.envs.register(
        id=f"RLPy{name}-v1",
        entry_point="rlpy.gym:gridworld",
        max_episode_steps=max_steps,
        kwargs=dict(mapfile=mapfile, cls=cls, mode="raw"),
        reward_threshold=threshold,
    )
    gym.envs.register(
        id=f"RLPy{name}-v2",
        entry_point="rlpy.gym:gridworld",
        max_episode_steps=max_steps,
        kwargs=dict(mapfile=mapfile, cls=cls, mode="image"),
        reward_threshold=threshold,
    )
    gym.envs.register(
        id=f"RLPy{name}-v3",
        entry_point="rlpy.gym:gridworld",
        max_episode_steps=max_steps,
        kwargs=dict(mapfile=mapfile, cls=cls, mode="binary-image"),
        reward_threshold=threshold,
    )


for mapfile in domains.GridWorld.DEFAULT_MAP_DIR.glob("*.txt"):
    register_gridworld(mapfile)

for mapfile in domains.FixedRewardGridWorld.DEFAULT_MAP_DIR.glob("*.txt"):
    register_gridworld(mapfile, cls=domains.FixedRewardGridWorld, threshold=80)

for mapfile in domains.BernoulliGridWorld.DEFAULT_MAP_DIR.glob("*.txt"):
    register_gridworld(mapfile, cls=domains.BernoulliGridWorld)


def register_lifegame(rule, prefix, mapfile):
    name = prefix + mapfile.stem
    gym.envs.register(
        id=f"RLPyLifeGame{name}-v0",
        entry_point="rlpy.gym:lifegame",
        max_episode_steps=200,
        kwargs=dict(mapfile=mapfile, rule=rule, mode="image"),
        reward_threshold=1.0,
    )
    gym.envs.register(
        id=f"RLPyLifeGame{name}-v1",
        entry_point="rlpy.gym:lifegame",
        max_episode_steps=200,
        kwargs=dict(mapfile=mapfile, rule=rule, mode="binary-image"),
        reward_threshold=1.0,
    )


for mapfile in domains.LifeGameSurvival.DEFAULT_MAP_DIR.joinpath("life").glob("*.txt"):
    register_lifegame("life", "", mapfile)

for mapfile in domains.LifeGameSurvival.DEFAULT_MAP_DIR.joinpath("dry").glob("*.txt"):
    register_lifegame("dry", "Dry", mapfile)

for mapfile in domains.LifeGameSurvival.DEFAULT_MAP_DIR.joinpath("seeds").glob("*.txt"):
    register_lifegame("seeds", "Seeds", mapfile)

for size in range(4, 40, 2):
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
        kwargs=dict(noise=0.01, cfg=cfgfile),
        reward_threshold=8000,
    )
    gym.envs.register(
        id=f"RLPyPinball{name}-v2",
        entry_point="rlpy.gym:pinball",
        max_episode_steps=1000,
        kwargs=dict(noise=0.05, cfg=cfgfile),
        reward_threshold=8000,
    )
