import numpy as np
from rlpy.domains import FiniteCartPoleBalanceOriginal, InfCartPoleBalance
from rlpy.Tools import __rlpy_location__
import os
import pytest


def get_file(fname):
    return os.path.join(__rlpy_location__, "..", "tests", fname)


@pytest.mark.parametrize(
    "domain_class, filename",
    [
        (InfCartPoleBalance, get_file("traj_InfiniteCartpoleBalance.npy")),
        (
            FiniteCartPoleBalanceOriginal,
            get_file("traj_FiniteCartpoleBalanceOriginal.npy"),
        ),
    ],
)
def test_trajectory(domain_class, filename):
    traj = np.load(filename, allow_pickle=True)
    traj_now = sample_random_trajectory(domain_class)
    for e1, e2 in zip(traj_now, traj):
        # State
        assert np.allclose(e1[0], e2[0]), "now: {}, saved: {}".format(e1[0], e2[0])
        # Reward
        assert e1[-1] == e2[-1], "now: {}, saved: {}".format(e1[-1], e2[-1])
        # Terminal
        assert e1[1] == e2[1], "now: {}, saved: {}".format(e1[1], e2[1])
        # Actions
        assert len(e1[2]) == len(e2[2])
        # p_actions
        assert np.all([a == b for a, b in zip(e1[2], e2[2])])


def sample_random_trajectory(domain_class):
    """
    sample a trajectory of 1000 steps
    """
    traj = []
    np.random.seed(1)
    domain = domain_class()
    domain.random_state = np.random.RandomState(1)
    terminal = True
    steps = 0
    T = 1000
    r = 0
    while steps < T:
        if terminal:
            s, terminal, p_actions = domain.s0()
        elif steps % domain.episodeCap == 0:
            s, terminal, p_actions = domain.s0()
        a = np.random.choice(p_actions)
        traj.append((s, terminal, p_actions, a, r))
        r, s, terminal, p_actions = domain.step(a)
        steps += 1
    return traj
