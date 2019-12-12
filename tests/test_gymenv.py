import gym
import pytest

from rlpy import gym as rlpy_gym  # noqa


@pytest.mark.parametrize("envname", ["GridWorld4x5-v0", "GridWorld4x5-v1"])
def test_gridworld(envname):
    env = gym.make(envname, noise=0.0)
    state = env.reset()
    assert state.shape == env.observation_space.shape
    for act in [0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 2]:
        _, reward, terminal, _ = env.step(act)
    assert terminal
    assert reward == 1.0


@pytest.mark.parametrize("envname", ["DeepSea20-v0", "DeepSea20-v1"])
def test_deepsea(envname):
    env = gym.make(envname, noise=0.0)
    state = env.reset()
    assert state.shape == env.observation_space.shape
    for _ in range(20):
        state, reward, terminal, _ = env.step(1)
    assert terminal
    assert reward > 0.9
