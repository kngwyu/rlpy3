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


def grid4x5_v3_goal():
    layers = []
    layers.append(
        np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 1]])
    )
    layers.append(
        np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    )
    layers.append(
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    )
    layers.append(
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]])
    )
    layers.append(
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0]])
    )
    layers.append(
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]])
    )
    return np.stack(layers).astype(np.float32)


@pytest.mark.parametrize(
    "version, goal_fn",
    [
        (0, grid4x5_v0_goal),
        (1, grid4x5_v1_goal),
        (2, grid4x5_v2_goal),
        (3, grid4x5_v3_goal),
    ],
)
def test_gridworld(version, goal_fn):
    env = gym.make(f"RLPyGridWorld4x5-v{version}", noise=0.0)
    assert env.unwrapped.domain.episode_cap == 18
    state = env.reset()
    assert env.action_space.n == 4
    assert state.shape == env.observation_space.shape
    for act in [0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 2]:
        state, reward, terminal, _ = env.step(act)
    assert terminal
    assert reward == 1.0
    np.testing.assert_array_almost_equal(state, goal_fn())


@pytest.mark.parametrize("version", [0, 1])
def test_randomgoal_gridworld(version):
    env = gym.make(f"RLPyGridWorld11x11-4Rooms-RandomGoal-v{version}", noise=0.0)
    state = env.reset()
    assert env.action_space.n == 4
    assert state.shape == env.observation_space.shape
    state, reward, terminal, info = env.step(0)
    # Test raw obs
    raw_space = env.raw_observation_space
    assert raw_space.dtype == np.int64
    for raw in info["raw_obs"], env.initial_raw_obs():
        raw = info["raw_obs"]
        assert raw.shape == (3,)
        assert raw[:2].tolist() == [0, 0]
        assert 0 <= raw[2] <= 2


@pytest.mark.parametrize("version", [0, 1, 2])
def test_deepsea(version):
    env = gym.make(f"RLPyDeepSea20-v{version}", noise=0.0)
    state = env.reset()
    assert env.action_space.n == 2
    assert state.shape == env.observation_space.shape
    for _ in range(20):
        state, reward, terminal, _ = env.step(1)
    assert terminal
    assert reward > 0.9


@pytest.mark.parametrize("version", [0, 1])
def test_lifegame(version):
    env = gym.make(f"RLPyLifeGame7x7ever-v{version}")
    state = env.reset()
    assert env.action_space.n == 4
    assert state.shape == env.observation_space.shape
    state, reward, terminal, _ = env.step(1)
    assert not terminal
    assert reward == 0.01


@pytest.mark.parametrize(
    "name, version", [("Box", 0), ("Box", 1), ("Medium", 0), ("Medium", 1)]
)
def test_pinball(name, version):
    env = gym.make(f"RLPyPinball{name}-v{version}")
    state = env.reset()
    assert env.action_space.n == 5
    assert state.shape == env.observation_space.shape
