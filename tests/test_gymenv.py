import gym
import numpy as np
import pytest

from rlpy import gym as rlpy_gym  # noqa


def grid4x5_v0_goal():
    res = np.zeros(20, dtype=np.float32)
    res[4 * 4 - 1] = 1.0
    return res


def grid4x5_v1_goal():
    return np.array([3.0, 3.0], dtype=np.float32)


def grid4x5_v2_goal():
    map_ = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [2, 4, 4, 5, 0]]
    )
    return map_.astype(np.float32).reshape(1, 4, 5)


@pytest.mark.parametrize(
    "version, goal_fn",
    [("v0", grid4x5_v0_goal), ("v1", grid4x5_v1_goal), ("v2", grid4x5_v2_goal)],
)
def test_gridworld(version, goal_fn):
    env = gym.make("RLPyGridWorld4x5-" + version, noise=0.0)
    state = env.reset()
    assert env.action_space.n == 4
    assert state.shape == env.observation_space.shape
    for act in [0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 2]:
        state, reward, terminal, _ = env.step(act)
    assert terminal
    assert reward == 1.0
    np.testing.assert_array_almost_equal(state, goal_fn())


@pytest.mark.parametrize("version", ["v0", "v1", "v2"])
def test_deepsea(version):
    env = gym.make("RLPyDeepSea20-" + version, noise=0.0)
    state = env.reset()
    assert env.action_space.n == 2
    assert state.shape == env.observation_space.shape
    for _ in range(20):
        state, reward, terminal, _ = env.step(1)
    assert terminal
    assert reward > 0.9
